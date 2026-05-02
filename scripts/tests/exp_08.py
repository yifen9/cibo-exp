from pathlib import Path
from copy import deepcopy
from collections import deque
import importlib.util
from statistics import mean, variance

import numpy as np

from cibo_pycore.io.csv import write_csv
from cibo_pycore.io.json import write_json
from cibo_pycore.io.yaml import read_yaml, write_yaml
from cibo_pycore.utils.audit import Audit
from cibo_pycore.utils.console import ConsoleSink
from cibo_pycore.utils.jlog import jline
from cibo_pycore.utils.logger import Logger
from cibo_pycore.utils.meta import build_meta
from cibo_pycore.utils.progress import Progress
from cibo_pycore.utils.versioner import build_version_dir


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_exp06_module(root: Path):
    path = root / "scripts" / "tests" / "exp_06.py"
    spec = importlib.util.spec_from_file_location(
        "cibo_exp_06_runtime_for_exp_08", str(path)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def draw_line(height: int, width: int) -> np.ndarray:
    target = np.zeros((height, width), dtype=np.uint8)
    target[height // 2, 6 : width - 6] = 1
    return target


def draw_boundary(height: int, width: int) -> np.ndarray:
    target = np.zeros((height, width), dtype=np.uint8)
    top = 8
    left = 8
    h = 16
    w = 16
    bottom = min(height, top + h)
    right = min(width, left + w)
    target[top, left:right] = 1
    target[bottom - 1, left:right] = 1
    target[top:bottom, left] = 1
    target[top:bottom, right - 1] = 1
    return target


def draw_connected_blob(height: int, width: int) -> np.ndarray:
    target = np.zeros((height, width), dtype=np.uint8)
    cx = height // 2
    cy = width // 2

    for x in range(height):
        for y in range(width):
            dx = (x - cx) / 7.0
            dy = (y - cy) / 9.0
            wave = 0.18 * np.sin(0.9 * x) + 0.12 * np.cos(0.7 * y)
            if dx * dx + dy * dy <= 1.0 + wave:
                target[x, y] = 1

    return target


def draw_irregular_glyph(height: int, width: int) -> np.ndarray:
    target = np.zeros((height, width), dtype=np.uint8)

    target[7:25, 9] = 1
    target[7, 9:22] = 1
    target[15, 9:21] = 1
    target[24, 9:23] = 1
    target[8:15, 22] = 1
    target[16:24, 23] = 1
    target[10:14, 15] = 1
    target[18:22, 16] = 1

    return target


def draw_maze(height: int, width: int) -> np.ndarray:
    target = np.zeros((height, width), dtype=np.uint8)
    path = [
        (5, 5),
        (5, 24),
        (9, 24),
        (9, 10),
        (13, 10),
        (13, 27),
        (18, 27),
        (18, 7),
        (23, 7),
        (23, 25),
        (27, 25),
    ]

    for (x0, y0), (x1, y1) in zip(path, path[1:]):
        if x0 == x1:
            a = min(y0, y1)
            b = max(y0, y1)
            target[x0, a : b + 1] = 1
        elif y0 == y1:
            a = min(x0, x1)
            b = max(x0, x1)
            target[a : b + 1, y0] = 1

    return target


def draw_disconnected_components(height: int, width: int) -> np.ndarray:
    target = np.zeros((height, width), dtype=np.uint8)
    boxes = [
        (6, 6, 4, 4),
        (6, 22, 4, 4),
        (22, 6, 4, 4),
        (22, 22, 4, 4),
        (14, 14, 5, 5),
    ]

    for top, left, h, w in boxes:
        target[top : top + h, left : left + w] = 1

    return target


def draw_sparse_dots(height: int, width: int) -> np.ndarray:
    target = np.zeros((height, width), dtype=np.uint8)
    cells = [
        (6, 6),
        (6, 16),
        (6, 25),
        (12, 10),
        (12, 21),
        (17, 5),
        (17, 16),
        (17, 27),
        (23, 9),
        (23, 20),
        (27, 14),
        (27, 25),
    ]

    for x, y in cells:
        target[x, y] = 1

    return target


def make_target_exp08(height: int, width: int, cfg: dict) -> np.ndarray:
    kind = str(cfg["type"])

    if kind == "line":
        return draw_line(height, width)

    if kind == "boundary":
        return draw_boundary(height, width)

    if kind == "connected_blob":
        return draw_connected_blob(height, width)

    if kind == "irregular_glyph":
        return draw_irregular_glyph(height, width)

    if kind == "maze":
        return draw_maze(height, width)

    if kind == "disconnected_components":
        return draw_disconnected_components(height, width)

    if kind == "sparse_dots":
        return draw_sparse_dots(height, width)

    raise ValueError(f"unknown exp_08 morphology type: {kind}")


def apply_morphology_damage(canvas: np.ndarray, target: np.ndarray, cfg: dict) -> int:
    cells = np.argwhere(target == 1)
    p = float(cfg.get("damage_fraction", 0.20))
    seed = int(cfg.get("damage_seed", 0))
    rng = np.random.default_rng(seed)
    count = max(1, int(round(len(cells) * p)))
    selected = cells[rng.choice(len(cells), size=min(count, len(cells)), replace=False)]

    for x, y in selected:
        canvas[int(x), int(y)] = 0

    return int(len(selected))


def connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros_like(mask, dtype=bool)
    comps = []

    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            if not bool(mask[x, y]) or bool(seen[x, y]):
                continue

            q = deque([(x, y)])
            seen[x, y] = True
            comp = []

            while q:
                i, j = q.popleft()
                comp.append((i, j))

                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ni = i + di
                    nj = j + dj

                    if 0 <= ni < mask.shape[0] and 0 <= nj < mask.shape[1]:
                        if bool(mask[ni, nj]) and not bool(seen[ni, nj]):
                            seen[ni, nj] = True
                            q.append((ni, nj))

            comps.append(comp)

    return comps


def local_coherence(target: np.ndarray, radius: int) -> float:
    cells = np.argwhere(target == 1)

    if len(cells) == 0:
        return 0.0

    scores = []

    for x, y in cells:
        x = int(x)
        y = int(y)
        x0 = max(0, x - radius)
        x1 = min(target.shape[0], x + radius + 1)
        y0 = max(0, y - radius)
        y1 = min(target.shape[1], y + radius + 1)
        window = target[x0:x1, y0:y1]
        total = max(1, window.size - 1)
        active = int(window.sum()) - int(target[x, y])
        scores.append(active / total)

    return float(sum(scores) / len(scores))


def morphology_static_metrics(target: np.ndarray, radius: int) -> dict:
    comps = connected_components(target == 1)
    sizes = [len(c) for c in comps]

    return {
        "target_size": int(target.sum()),
        "component_count": int(len(comps)),
        "largest_component_size": int(max(sizes)) if sizes else 0,
        "smallest_component_size": int(min(sizes)) if sizes else 0,
        "local_coherence": local_coherence(target, radius),
    }


def condition_to_exp06_condition(cfg: dict, morphology: dict) -> dict:
    repair = cfg["repair"]

    return {
        "id": str(morphology["id"]),
        "group": str(morphology["class"]),
        "operator": str(repair["operator"]),
        "mechanism": str(repair["mechanism"]),
        "theta_repair": float(repair["theta_repair"]),
        "k": int(repair["k"]),
        "m": int(repair["m"]),
        "gamma": float(repair["gamma"]),
        "budget": bool(repair["budget"]),
    }


def build_condition_cfg(cfg: dict, morphology: dict, seed: int) -> dict:
    out = deepcopy(cfg)
    out["target"] = {"type": str(morphology["type"])}
    out["perturbation"] = {
        "schedule": [
            {
                "type": "random_noise",
                "time": int(cfg["perturbation"]["noise_time"]),
                "strength": float(cfg["perturbation"]["noise_strength"]),
                "scope": str(cfg["perturbation"]["noise_scope"]),
            },
            {
                "type": "local_gap_attack",
                "time": int(cfg["perturbation"]["damage_time"]),
                "damage_fraction": float(cfg["perturbation"]["damage_fraction"]),
                "damage_seed": int(seed * 1009 + len(str(morphology["id"]))),
            },
        ]
    }
    return out


def is_strong_success(summary: dict, cfg: dict) -> bool:
    return float(summary["final_target_true_positive_rate"]) > float(
        cfg["metrics"]["strong_tpr_threshold"]
    ) and float(summary["final_outside_false_positive_rate"]) < float(
        cfg["metrics"]["strong_fpr_threshold"]
    )


def is_strict_success(summary: dict, cfg: dict) -> bool:
    return float(summary["final_target_true_positive_rate"]) > float(
        cfg["metrics"]["strict_tpr_threshold"]
    ) and float(summary["final_outside_false_positive_rate"]) < float(
        cfg["metrics"]["strict_fpr_threshold"]
    )


def is_failure(summary: dict) -> bool:
    return (
        summary.get("saturation_time") is not None
        or summary.get("target_damage_time") is not None
    )


def failure_mode(summary: dict, cfg: dict) -> str:
    tpr = float(summary["final_target_true_positive_rate"])
    fpr = float(summary["final_outside_false_positive_rate"])
    strong_tpr = float(cfg["metrics"]["strong_tpr_threshold"])
    strong_fpr = float(cfg["metrics"]["strong_fpr_threshold"])

    if tpr > strong_tpr and fpr < strong_fpr:
        return "success"

    if tpr <= strong_tpr and fpr < strong_fpr:
        return "target_under_repair"

    if tpr > strong_tpr and fpr >= strong_fpr:
        return "outside_saturation"

    return "mixed_failure"


def safe_var(xs: list[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    return float(variance(xs))


def aggregate_by_morphology(summaries: list[dict], cfg: dict) -> list[dict]:
    groups = {}

    for s in summaries:
        groups.setdefault(str(s["morphology_id"]), []).append(s)

    rows = []

    for morphology_id, items in groups.items():
        balanced = [float(x["final_balanced_integrity"]) for x in items]
        tpr = [float(x["final_target_true_positive_rate"]) for x in items]
        fpr = [float(x["final_outside_false_positive_rate"]) for x in items]
        expansion = [float(x["expansion_total"]) for x in items]
        gap = [float(x["gap_closure_total"]) for x in items]
        repair_precision = [float(x["repair_precision"]) for x in items]
        false_repair = [float(x["false_repair_rate"]) for x in items]
        strong = [1 if is_strong_success(x, cfg) else 0 for x in items]
        strict = [1 if is_strict_success(x, cfg) else 0 for x in items]
        failures = [1 if is_failure(x) else 0 for x in items]
        first = items[0]
        modes = {}

        for item in items:
            mode = failure_mode(item, cfg)
            modes[mode] = modes.get(mode, 0) + 1

        dominant_mode = sorted(modes.items(), key=lambda x: (-x[1], x[0]))[0][0]

        rows.append(
            {
                "morphology_id": morphology_id,
                "morphology_class": str(first["morphology_class"]),
                "morphology_type": str(first["morphology_type"]),
                "coherence_group": str(first["coherence_group"]),
                "seed_count": len(items),
                "strong_success_rate": float(sum(strong) / len(strong)),
                "strict_success_rate": float(sum(strict) / len(strict)),
                "failure_rate": float(sum(failures) / len(failures)),
                "dominant_failure_mode": dominant_mode,
                "mean_final_balanced_integrity": float(mean(balanced)),
                "var_final_balanced_integrity": safe_var(balanced),
                "worst_final_balanced_integrity": float(min(balanced)),
                "best_final_balanced_integrity": float(max(balanced)),
                "mean_final_tpr": float(mean(tpr)),
                "worst_final_tpr": float(min(tpr)),
                "mean_final_fpr": float(mean(fpr)),
                "worst_final_fpr": float(max(fpr)),
                "mean_expansion_total": float(mean(expansion)),
                "mean_gap_closure_total": float(mean(gap)),
                "mean_repair_precision": float(mean(repair_precision)),
                "mean_false_repair_rate": float(mean(false_repair)),
                "target_size": int(first["target_size"]),
                "component_count": int(first["component_count"]),
                "largest_component_size": int(first["largest_component_size"]),
                "smallest_component_size": int(first["smallest_component_size"]),
                "local_coherence": float(first["local_coherence"]),
            }
        )

    rows.sort(key=lambda x: str(x["morphology_class"]))
    return rows


def morphology_sensitivity(rows: list[dict]) -> dict:
    high = [
        float(x["mean_final_balanced_integrity"])
        for x in rows
        if str(x["coherence_group"]) == "high"
    ]
    low = [
        float(x["mean_final_balanced_integrity"])
        for x in rows
        if str(x["coherence_group"]) == "low"
    ]

    return {
        "high_coherence_mean_balanced": float(mean(high)) if high else 0.0,
        "low_coherence_mean_balanced": float(mean(low)) if low else 0.0,
        "morphology_sensitivity": float(mean(high) - mean(low))
        if high and low
        else 0.0,
    }


def make_aggregate_plots(rows: list[dict], figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [str(x["morphology_class"]) for x in rows]
    balanced = [float(x["mean_final_balanced_integrity"]) for x in rows]
    tpr = [float(x["mean_final_tpr"]) for x in rows]
    fpr = [float(x["mean_final_fpr"]) for x in rows]
    coherence = [float(x["local_coherence"]) for x in rows]

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(labels, balanced)
    ax.set_title("Experiment 08 mean final balanced integrity by morphology")
    ax.set_xlabel("morphology")
    ax.set_ylabel("mean final balanced integrity")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(str(figures_dir / "mean_final_balanced_by_morphology.png"), dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(1, 1, 1)

    for item in rows:
        ax.scatter(float(item["mean_final_fpr"]), float(item["mean_final_tpr"]))
        ax.text(
            float(item["mean_final_fpr"]),
            float(item["mean_final_tpr"]),
            str(item["morphology_class"]),
            fontsize=8,
        )

    ax.set_title("Experiment 08 morphology TPR-FPR plane")
    ax.set_xlabel("mean final outside FPR")
    ax.set_ylabel("mean final TPR")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(str(figures_dir / "morphology_tpr_fpr_plane.png"), dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(1, 1, 1)

    for item in rows:
        ax.scatter(
            float(item["local_coherence"]), float(item["mean_final_balanced_integrity"])
        )
        ax.text(
            float(item["local_coherence"]),
            float(item["mean_final_balanced_integrity"]),
            str(item["morphology_class"]),
            fontsize=8,
        )

    ax.set_title("Experiment 08 local coherence vs performance")
    ax.set_xlabel("local coherence")
    ax.set_ylabel("mean final balanced integrity")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(str(figures_dir / "coherence_vs_balanced.png"), dpi=160)
    plt.close(fig)


def make_gallery(cfg: dict, figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    morphologies = list(cfg["morphologies"])
    height = int(cfg["canvas"]["height"])
    width = int(cfg["canvas"]["width"])

    fig = plt.figure(figsize=(12, 4))

    for i, morphology in enumerate(morphologies):
        target = make_target_exp08(height, width, {"type": str(morphology["type"])})
        ax = fig.add_subplot(1, len(morphologies), i + 1)
        ax.imshow(target, interpolation="nearest")
        ax.set_title(str(morphology["class"]), fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(str(figures_dir / "morphology_gallery.png"), dpi=160)
    plt.close(fig)


def main() -> int:
    root = repo_root()
    config_path = root / "config" / "tests" / "exp_08.yaml"
    cfg = read_yaml(str(config_path))
    exp06 = load_exp06_module(root)
    exp06.make_target = make_target_exp08
    exp06.apply_local_gap_attack = apply_morphology_damage

    meta = build_meta(
        params=cfg,
        env=str(root / "uv.lock"),
        script=str(Path(__file__).resolve()),
        src=str(root / "src"),
    )

    output_root = root / cfg["output"]["root"]
    run_dir = Path(build_version_dir(str(output_root), meta))
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    write_yaml(str(run_dir / "config.yaml"), cfg)

    audit = Audit.create(str(run_dir), meta)
    logger = Logger(sinks=[ConsoleSink(transient=False), audit])
    logger.info(jline("run", cfg["name"], "start", run_dir=str(run_dir)))

    seeds = [int(x) for x in cfg["run"]["seeds"]]
    morphologies = list(cfg["morphologies"])
    total_units = (int(cfg["run"]["steps"]) + 1) * len(seeds) * len(morphologies)
    progress = Progress(logger=logger, name=cfg["name"], total=total_units)
    progress.start()

    all_runs = []
    height = int(cfg["canvas"]["height"])
    width = int(cfg["canvas"]["width"])
    radius = int(cfg["operator"]["radius"])

    for morphology in morphologies:
        target = make_target_exp08(height, width, {"type": str(morphology["type"])})
        static = morphology_static_metrics(target, radius)

        for seed in seeds:
            exp06_condition = condition_to_exp06_condition(cfg, morphology)
            condition_cfg = build_condition_cfg(cfg, morphology, seed)
            logger.info(
                jline("morphology", str(morphology["id"]), "start", seed=int(seed))
            )
            result = exp06.run_condition(
                condition_cfg, exp06_condition, seed, run_dir, logger, progress
            )
            result["summary"]["morphology_id"] = str(morphology["id"])
            result["summary"]["morphology_class"] = str(morphology["class"])
            result["summary"]["morphology_type"] = str(morphology["type"])
            result["summary"]["coherence_group"] = str(morphology["coherence_group"])
            result["summary"].update(static)
            all_runs.append(result)

    progress.finish()

    summaries = [item["summary"] for item in all_runs]
    morphology_rows = aggregate_by_morphology(summaries, cfg)
    sensitivity = morphology_sensitivity(morphology_rows)

    run_header = [
        "morphology_id",
        "morphology_class",
        "morphology_type",
        "coherence_group",
        "seed",
        "operator",
        "mechanism",
        "theta_preserve",
        "theta_repair",
        "k",
        "m",
        "gamma",
        "budget",
        "target_size",
        "component_count",
        "largest_component_size",
        "smallest_component_size",
        "local_coherence",
        "final_mismatch_rate",
        "best_mismatch_rate",
        "mean_mismatch_rate",
        "final_balanced_integrity",
        "best_balanced_integrity",
        "mean_balanced_integrity",
        "final_target_true_positive_rate",
        "final_target_false_negative_rate",
        "final_outside_false_positive_rate",
        "final_outside_true_negative_rate",
        "max_outside_false_positive_rate",
        "min_target_true_positive_rate",
        "saturation_time",
        "target_damage_time",
        "recovery_time",
        "proposal_total",
        "applied_total",
        "write_0_total",
        "write_1_total",
        "repair_pass_total",
        "repair_reject_total",
        "true_repair_total",
        "false_repair_total",
        "true_suppression_total",
        "false_suppression_total",
        "repair_precision",
        "false_repair_rate",
        "suppression_precision",
        "false_suppression_rate",
        "repair_activation_rate",
        "repair_rejection_rate",
        "repair_to_suppression_ratio",
        "expansion_total",
        "gap_closure_total",
        "metrics_path",
        "events_path",
    ]

    morphology_header = [
        "morphology_id",
        "morphology_class",
        "morphology_type",
        "coherence_group",
        "seed_count",
        "strong_success_rate",
        "strict_success_rate",
        "failure_rate",
        "dominant_failure_mode",
        "mean_final_balanced_integrity",
        "var_final_balanced_integrity",
        "worst_final_balanced_integrity",
        "best_final_balanced_integrity",
        "mean_final_tpr",
        "worst_final_tpr",
        "mean_final_fpr",
        "worst_final_fpr",
        "mean_expansion_total",
        "mean_gap_closure_total",
        "mean_repair_precision",
        "mean_false_repair_rate",
        "target_size",
        "component_count",
        "largest_component_size",
        "smallest_component_size",
        "local_coherence",
    ]

    write_csv(
        str(run_dir / "runs_summary.csv"),
        [[s.get(k) for k in run_header] for s in summaries],
        header=run_header,
    )
    write_json(str(run_dir / "runs_summary.json"), summaries)
    write_csv(
        str(run_dir / "morphology_summary.csv"),
        [[s.get(k) for k in morphology_header] for s in morphology_rows],
        header=morphology_header,
    )
    write_json(str(run_dir / "morphology_summary.json"), morphology_rows)

    if bool(cfg["output"]["make_plot"]):
        make_gallery(cfg, figures_dir)
        make_aggregate_plots(morphology_rows, figures_dir)

    run_summary = {
        "name": cfg["name"],
        "run_dir": str(run_dir),
        "fingerprint": meta.get("fingerprint"),
        "morphology_count": len(morphologies),
        "seed_count": len(seeds),
        "run_count": len(summaries),
        "runs_summary_path": "runs_summary.csv",
        "runs_summary_json_path": "runs_summary.json",
        "morphology_summary_path": "morphology_summary.csv",
        "morphology_summary_json_path": "morphology_summary.json",
        "morphology_sensitivity": sensitivity,
        "figures": [
            "figures/morphology_gallery.png",
            "figures/mean_final_balanced_by_morphology.png",
            "figures/morphology_tpr_fpr_plane.png",
            "figures/coherence_vs_balanced.png",
        ]
        if bool(cfg["output"]["make_plot"])
        else [],
        "success": True,
    }

    write_json(str(run_dir / "summary.json"), run_summary)

    logger.info(jline("run", cfg["name"], "finish", run_dir=str(run_dir)))
    audit.finish_success()

    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
