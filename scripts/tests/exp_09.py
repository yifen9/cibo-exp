from pathlib import Path
from copy import deepcopy
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
        "cibo_exp_06_runtime_for_exp_09", str(path)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def condition_to_exp06_condition(cfg: dict, condition: dict) -> dict:
    repair = cfg["repair"]

    return {
        "id": str(condition["id"]),
        "group": str(condition["group"]),
        "operator": str(repair["operator"]),
        "mechanism": str(repair["mechanism"]),
        "theta_repair": float(repair["theta_repair"]),
        "k": int(repair["k"]),
        "m": int(repair["m"]),
        "gamma": float(repair["gamma"]),
        "budget": bool(repair["budget"]),
    }


def random_target_delete(
    canvas: np.ndarray, target: np.ndarray, fraction: float, seed: int
) -> int:
    rng = np.random.default_rng(seed)
    cells = np.argwhere(target == 1)

    if len(cells) == 0:
        return 0

    count = max(1, int(round(len(cells) * fraction)))
    selected = cells[rng.choice(len(cells), size=min(count, len(cells)), replace=False)]

    for x, y in selected:
        canvas[int(x), int(y)] = 0

    return int(len(selected))


def outside_activate(
    canvas: np.ndarray, target: np.ndarray, fraction: float, seed: int
) -> int:
    rng = np.random.default_rng(seed)
    cells = np.argwhere(target == 0)

    if len(cells) == 0:
        return 0

    count = max(1, int(round(len(cells) * fraction)))
    selected = cells[rng.choice(len(cells), size=min(count, len(cells)), replace=False)]

    for x, y in selected:
        canvas[int(x), int(y)] = 1

    return int(len(selected))


def block_target_occlusion(
    canvas: np.ndarray, target: np.ndarray, center: list[int], size: int
) -> int:
    cx = int(center[0])
    cy = int(center[1])
    r = int(size // 2)
    changed = 0

    for x in range(max(0, cx - r), min(canvas.shape[0], cx + r + 1)):
        for y in range(max(0, cy - r), min(canvas.shape[1], cy + r + 1)):
            if int(target[x, y]) == 1:
                if int(canvas[x, y]) != 0:
                    changed += 1
                canvas[x, y] = 0

    return int(changed)


def adversarial_false_positive(
    canvas: np.ndarray, target: np.ndarray, cfg: dict
) -> int:
    top = int(cfg["false_top"])
    left = int(cfg["false_left"])
    h = int(cfg["false_height"])
    w = int(cfg["false_width"])
    bottom = min(canvas.shape[0], top + h)
    right = min(canvas.shape[1], left + w)
    changed = 0
    cells = []

    for y in range(left, right):
        cells.append((top, y))
        cells.append((bottom - 1, y))

    for x in range(top, bottom):
        cells.append((x, left))
        cells.append((x, right - 1))

    for x, y in cells:
        if (
            0 <= x < canvas.shape[0]
            and 0 <= y < canvas.shape[1]
            and int(target[x, y]) == 0
        ):
            if int(canvas[x, y]) != 1:
                changed += 1
            canvas[x, y] = 1

    return int(changed)


def apply_exp09_perturbation(canvas: np.ndarray, target: np.ndarray, cfg: dict) -> int:
    kind = str(cfg["kind"])
    seed = int(cfg.get("seed", 0))

    if kind == "random_target_deletion":
        return random_target_delete(
            canvas, target, float(cfg["target_delete_fraction"]), seed
        )

    if kind == "block_target_occlusion":
        return block_target_occlusion(
            canvas, target, list(cfg["block_center"]), int(cfg["block_size"])
        )

    if kind == "outside_false_positive_activation":
        return outside_activate(canvas, target, float(cfg["outside_fraction"]), seed)

    if kind == "salt_and_pepper_mixed_noise":
        a = random_target_delete(
            canvas, target, float(cfg["target_delete_fraction"]), seed
        )
        b = outside_activate(canvas, target, float(cfg["outside_fraction"]), seed + 17)
        return int(a + b)

    if kind == "drifting_damage_region":
        return block_target_occlusion(
            canvas, target, list(cfg["block_center"]), int(cfg["block_size"])
        )

    if kind == "adversarial_structural_false_positives":
        return adversarial_false_positive(canvas, target, cfg)

    raise ValueError(f"unknown exp_09 perturbation kind: {kind}")


def build_schedule(condition: dict, seed: int) -> list[dict]:
    kind = str(condition["perturbation_type"])

    if kind == "random_target_deletion":
        return [
            {
                "type": "local_gap_attack",
                "time": int(condition["target_delete_time"]),
                "kind": kind,
                "target_delete_fraction": float(condition["target_delete_fraction"]),
                "seed": int(seed * 1009 + 1),
            }
        ]

    if kind == "block_target_occlusion":
        return [
            {
                "type": "local_gap_attack",
                "time": int(condition["block_time"]),
                "kind": kind,
                "block_size": int(condition["block_size"]),
                "block_center": list(condition["block_center"]),
                "seed": int(seed * 1009 + 2),
            }
        ]

    if kind == "outside_false_positive_activation":
        return [
            {
                "type": "local_gap_attack",
                "time": int(condition["outside_time"]),
                "kind": kind,
                "outside_fraction": float(condition["outside_fraction"]),
                "seed": int(seed * 1009 + 3),
            }
        ]

    if kind == "salt_and_pepper_mixed_noise":
        return [
            {
                "type": "local_gap_attack",
                "time": int(condition["mixed_time"]),
                "kind": kind,
                "target_delete_fraction": float(condition["target_delete_fraction"]),
                "outside_fraction": float(condition["outside_fraction"]),
                "seed": int(seed * 1009 + 4),
            }
        ]

    if kind == "repeated_partial_damage":
        return [
            {
                "type": "local_gap_attack",
                "time": int(t),
                "kind": "random_target_deletion",
                "target_delete_fraction": float(condition["target_delete_fraction"]),
                "seed": int(seed * 1009 + i + 5),
            }
            for i, t in enumerate(condition["repeated_times"])
        ]

    if kind == "drifting_damage_region":
        return [
            {
                "type": "local_gap_attack",
                "time": int(t),
                "kind": kind,
                "block_size": int(condition["block_size"]),
                "block_center": list(condition["drift_centers"][i]),
                "seed": int(seed * 1009 + i + 11),
            }
            for i, t in enumerate(condition["drifting_times"])
        ]

    if kind == "adversarial_structural_false_positives":
        return [
            {
                "type": "local_gap_attack",
                "time": int(condition["adversarial_time"]),
                "kind": kind,
                "false_shape": str(condition["false_shape"]),
                "false_top": int(condition["false_top"]),
                "false_left": int(condition["false_left"]),
                "false_height": int(condition["false_height"]),
                "false_width": int(condition["false_width"]),
                "seed": int(seed * 1009 + 29),
            }
        ]

    raise ValueError(f"unknown perturbation type: {kind}")


def build_condition_cfg(cfg: dict, condition: dict, seed: int) -> dict:
    out = deepcopy(cfg)
    out["perturbation"] = {"schedule": build_schedule(condition, seed)}
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


def aggregate_by_perturbation(summaries: list[dict], cfg: dict) -> list[dict]:
    groups = {}

    for s in summaries:
        groups.setdefault(str(s["condition_id"]), []).append(s)

    rows = []

    for condition_id, items in groups.items():
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
        modes = {}

        for item in items:
            mode = failure_mode(item, cfg)
            modes[mode] = modes.get(mode, 0) + 1

        first = items[0]
        dominant_mode = sorted(modes.items(), key=lambda x: (-x[1], x[0]))[0][0]

        rows.append(
            {
                "condition_id": condition_id,
                "group": str(first["group"]),
                "perturbation_type": str(first["perturbation_type"]),
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
            }
        )

    rows.sort(key=lambda x: str(x["group"]))
    return rows


def perturbation_robustness_score(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    return float(mean([float(x["mean_final_balanced_integrity"]) for x in rows]))


def make_aggregate_plots(rows: list[dict], figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [str(x["group"]) for x in rows]
    balanced = [float(x["mean_final_balanced_integrity"]) for x in rows]
    success = [float(x["strong_success_rate"]) for x in rows]

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(labels, balanced)
    ax.set_title("Experiment 09 mean final balanced integrity by perturbation")
    ax.set_xlabel("perturbation class")
    ax.set_ylabel("mean final balanced integrity")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(str(figures_dir / "mean_final_balanced_by_perturbation.png"), dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(labels, success)
    ax.set_title("Experiment 09 strong success rate by perturbation")
    ax.set_xlabel("perturbation class")
    ax.set_ylabel("strong success rate")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(str(figures_dir / "strong_success_rate_by_perturbation.png"), dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(1, 1, 1)

    for item in rows:
        ax.scatter(float(item["mean_final_fpr"]), float(item["mean_final_tpr"]))
        ax.text(
            float(item["mean_final_fpr"]),
            float(item["mean_final_tpr"]),
            str(item["group"]),
            fontsize=8,
        )

    ax.set_title("Experiment 09 perturbation TPR-FPR plane")
    ax.set_xlabel("mean final outside FPR")
    ax.set_ylabel("mean final TPR")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(str(figures_dir / "perturbation_tpr_fpr_plane.png"), dpi=160)
    plt.close(fig)


def main() -> int:
    root = repo_root()
    config_path = root / "config" / "tests" / "exp_09.yaml"
    cfg = read_yaml(str(config_path))
    exp06 = load_exp06_module(root)
    exp06.apply_local_gap_attack = apply_exp09_perturbation

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
    conditions = list(cfg["conditions"])
    total_units = (int(cfg["run"]["steps"]) + 1) * len(seeds) * len(conditions)
    progress = Progress(logger=logger, name=cfg["name"], total=total_units)
    progress.start()

    all_runs = []

    for condition in conditions:
        for seed in seeds:
            exp06_condition = condition_to_exp06_condition(cfg, condition)
            condition_cfg = build_condition_cfg(cfg, condition, seed)
            logger.info(
                jline("perturbation", str(condition["id"]), "start", seed=int(seed))
            )
            result = exp06.run_condition(
                condition_cfg, exp06_condition, seed, run_dir, logger, progress
            )
            result["summary"]["perturbation_type"] = str(condition["perturbation_type"])
            all_runs.append(result)

    progress.finish()

    summaries = [item["summary"] for item in all_runs]
    perturbation_rows = aggregate_by_perturbation(summaries, cfg)
    robustness_score = perturbation_robustness_score(perturbation_rows)

    run_header = [
        "condition_id",
        "group",
        "perturbation_type",
        "seed",
        "operator",
        "mechanism",
        "theta_preserve",
        "theta_repair",
        "k",
        "m",
        "gamma",
        "budget",
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

    perturbation_header = [
        "condition_id",
        "group",
        "perturbation_type",
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
    ]

    write_csv(
        str(run_dir / "runs_summary.csv"),
        [[s.get(k) for k in run_header] for s in summaries],
        header=run_header,
    )
    write_json(str(run_dir / "runs_summary.json"), summaries)
    write_csv(
        str(run_dir / "perturbation_summary.csv"),
        [[s.get(k) for k in perturbation_header] for s in perturbation_rows],
        header=perturbation_header,
    )
    write_json(str(run_dir / "perturbation_summary.json"), perturbation_rows)

    if bool(cfg["output"]["make_plot"]):
        make_aggregate_plots(perturbation_rows, figures_dir)

    run_summary = {
        "name": cfg["name"],
        "run_dir": str(run_dir),
        "fingerprint": meta.get("fingerprint"),
        "condition_count": len(conditions),
        "seed_count": len(seeds),
        "run_count": len(summaries),
        "perturbation_robustness_score": robustness_score,
        "runs_summary_path": "runs_summary.csv",
        "runs_summary_json_path": "runs_summary.json",
        "perturbation_summary_path": "perturbation_summary.csv",
        "perturbation_summary_json_path": "perturbation_summary.json",
        "figures": [
            "figures/mean_final_balanced_by_perturbation.png",
            "figures/strong_success_rate_by_perturbation.png",
            "figures/perturbation_tpr_fpr_plane.png",
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
