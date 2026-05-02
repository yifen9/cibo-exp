from pathlib import Path
import math
from collections import defaultdict

import numpy as np

from cibo_pycore.io.csv import write_csv
from cibo_pycore.io.json import write_json
from cibo_pycore.io.jsonl import write_jsonl
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


def make_target(height: int, width: int, cfg: dict) -> np.ndarray:
    target = np.zeros((height, width), dtype=np.uint8)
    kind = cfg["type"]

    if kind == "horizontal_line":
        row = int(cfg.get("row", height // 2))
        thickness = int(cfg.get("thickness", 1))
        start = max(0, row - thickness // 2)
        end = min(height, start + thickness)
        target[start:end, :] = 1
        return target

    if kind == "boundary":
        top = int(cfg["top"])
        left = int(cfg["left"])
        h = int(cfg["height"])
        w = int(cfg["width"])
        thickness = int(cfg.get("thickness", 1))
        bottom = min(height, top + h)
        right = min(width, left + w)
        target[top : min(bottom, top + thickness), left:right] = 1
        target[max(top, bottom - thickness) : bottom, left:right] = 1
        target[top:bottom, left : min(right, left + thickness)] = 1
        target[top:bottom, max(left, right - thickness) : right] = 1
        return target

    raise ValueError(f"unknown target type: {kind}")


def init_operators(
    count: int, height: int, width: int, rng: np.random.Generator
) -> np.ndarray:
    xs = rng.integers(0, height, size=count)
    ys = rng.integers(0, width, size=count)
    return np.stack([xs, ys], axis=1)


def move_operators(
    positions: np.ndarray, height: int, width: int, rng: np.random.Generator
) -> None:
    delta = rng.integers(-1, 2, size=positions.shape)
    positions += delta
    positions[:, 0] = np.clip(positions[:, 0], 0, height - 1)
    positions[:, 1] = np.clip(positions[:, 1], 0, width - 1)


def window_bounds(
    array: np.ndarray, x: int, y: int, radius: int
) -> tuple[int, int, int, int]:
    height, width = array.shape
    return (
        max(0, x - radius),
        min(height, x + radius + 1),
        max(0, y - radius),
        min(width, y + radius + 1),
    )


def local_canvas_stats(
    canvas: np.ndarray, trace: np.ndarray, x: int, y: int, radius: int
) -> dict:
    x0, x1, y0, y1 = window_bounds(canvas, x, y, radius)
    cw = canvas[x0:x1, y0:y1]
    tw = trace[x0:x1, y0:y1]
    center = int(canvas[x, y])
    active_neighbors = int(cw.sum()) - center
    positive_trace = float(np.maximum(tw, 0.0).sum())
    negative_trace = float(np.maximum(-tw, 0.0).sum())
    return {
        "center": center,
        "active_neighbors": active_neighbors,
        "positive_trace": positive_trace,
        "negative_trace": negative_trace,
    }


def choose_action(
    canvas: np.ndarray,
    trace: np.ndarray,
    memory: int,
    x: int,
    y: int,
    radius: int,
    operator_type: str,
    action_space: str,
    suppression: float,
    cfg: dict,
    rng: np.random.Generator,
) -> tuple[str, int, bool]:
    stats = local_canvas_stats(canvas, trace, x, y, radius)
    center = stats["center"]
    active = stats["active_neighbors"]
    positive_trace = stats["positive_trace"]
    negative_trace = stats["negative_trace"]

    birth_threshold = int(cfg["birth_threshold"])
    survive_min = int(cfg["survive_min"])
    survive_max = int(cfg["survive_max"])

    unsupported_active = center == 1 and (active < survive_min or active > survive_max)
    supported_inactive = center == 0 and active >= birth_threshold
    trace_suppressed = negative_trace > positive_trace
    suppression_active = False

    if operator_type == "l1":
        if supported_inactive:
            return ("write_1" if action_space == "write" else "flip", 1, False)
        if unsupported_active and active == 0:
            return ("write_0" if action_space == "write" else "flip", -1, False)
        return "stay", 0, False

    if operator_type in {"l1s", "l1c", "l2s"}:
        effective_suppression = suppression

        if operator_type == "l2s" and memory > 0:
            effective_suppression = min(1.0, suppression + 0.25)

        if unsupported_active and rng.random() < effective_suppression:
            suppression_active = True
            return (
                "write_0" if action_space == "write" else "flip",
                -1,
                suppression_active,
            )

        if (
            supported_inactive
            and not trace_suppressed
            and rng.random() < max(0.0, 1.0 - 0.5 * effective_suppression)
        ):
            return ("write_1" if action_space == "write" else "flip", 1, False)

        if center == 1 and trace_suppressed and rng.random() < effective_suppression:
            suppression_active = True
            return (
                "write_0" if action_space == "write" else "flip",
                -1,
                suppression_active,
            )

        return "stay", 0, False

    raise ValueError(f"unknown operator type: {operator_type}")


def apply_random_noise(
    canvas: np.ndarray,
    strength: float,
    scope: str,
    target: np.ndarray,
    rng: np.random.Generator,
) -> int:
    if scope == "target":
        cells = np.argwhere(target == 1)
    elif scope == "outside":
        cells = np.argwhere(target == 0)
    elif scope == "all":
        cells = np.argwhere(np.ones_like(canvas, dtype=bool))
    else:
        raise ValueError(f"unknown perturbation scope: {scope}")

    count = max(1, int(math.ceil(len(cells) * strength)))
    selected = cells[rng.choice(len(cells), size=count, replace=False)]

    for x, y in selected:
        canvas[x, y] = 1 - canvas[x, y]

    return int(count)


def resolve_flip(canvas: np.ndarray, proposals: dict) -> dict:
    stats = {
        "applied_actions": 0,
        "flips_total": 0,
        "write_0_total": 0,
        "write_1_total": 0,
        "conflict_attempts": 0,
    }

    for cell, values in proposals.items():
        if len(values) > 1:
            stats["conflict_attempts"] += len(values) - 1
        if len(values) % 2 == 1:
            x, y = cell
            canvas[x, y] = 1 - canvas[x, y]
            stats["applied_actions"] += 1
            stats["flips_total"] += 1

    return stats


def resolve_write(canvas: np.ndarray, proposals: dict) -> dict:
    stats = {
        "applied_actions": 0,
        "flips_total": 0,
        "write_0_total": 0,
        "write_1_total": 0,
        "conflict_attempts": 0,
    }

    for cell, values in proposals.items():
        write_1 = sum(1 for v in values if v > 0)
        write_0 = sum(1 for v in values if v < 0)

        if write_1 > 0 and write_0 > 0:
            stats["conflict_attempts"] += min(write_1, write_0)

        if write_1 > write_0:
            x, y = cell
            if canvas[x, y] != 1:
                stats["applied_actions"] += 1
            canvas[x, y] = 1
            stats["write_1_total"] += 1
        elif write_0 > write_1:
            x, y = cell
            if canvas[x, y] != 0:
                stats["applied_actions"] += 1
            canvas[x, y] = 0
            stats["write_0_total"] += 1

    return stats


def compute_action_quality(
    before: np.ndarray, after: np.ndarray, target: np.ndarray
) -> tuple[int, int, int]:
    before_err = before != target
    after_err = after != target
    useful = int((before_err & ~after_err).sum())
    harmful = int((~before_err & after_err).sum())
    neutral = int((before_err == after_err).sum())
    return useful, harmful, neutral


def update_trace(
    trace: np.ndarray, proposals: dict, duration: int, decay: float
) -> None:
    trace *= float(decay)

    for (x, y), values in proposals.items():
        if len(values) == 0:
            continue
        score = sum(values)
        if score > 0:
            trace[x, y] = float(duration)
        elif score < 0:
            trace[x, y] = -float(duration)


def compute_metrics(canvas: np.ndarray, target: np.ndarray) -> dict:
    mismatch = canvas != target
    target_mask = target == 1
    outside_mask = target == 0

    target_total = int(target_mask.sum())
    outside_total = int(outside_mask.sum())

    tp = int(((canvas == 1) & target_mask).sum())
    fn = int(((canvas == 0) & target_mask).sum())
    fp = int(((canvas == 1) & outside_mask).sum())
    tn = int(((canvas == 0) & outside_mask).sum())

    tpr = float(tp / target_total) if target_total else 0.0
    fnr = float(fn / target_total) if target_total else 0.0
    fpr = float(fp / outside_total) if outside_total else 0.0
    tnr = float(tn / outside_total) if outside_total else 0.0

    return {
        "mismatch_rate": float(mismatch.mean()),
        "target_true_positive_rate": tpr,
        "target_false_negative_rate": fnr,
        "outside_false_positive_rate": fpr,
        "outside_true_negative_rate": tnr,
        "balanced_integrity": float(0.5 * (tpr + tnr)),
    }


def save_snapshot(
    path: Path,
    t: int,
    canvas: np.ndarray,
    target: np.ndarray,
    trace: np.ndarray,
    label: str,
) -> None:
    write_json(
        str(path),
        {
            "t": int(t),
            "label": label,
            "height": int(canvas.shape[0]),
            "width": int(canvas.shape[1]),
            "canvas": canvas.astype(int).tolist(),
            "target": target.astype(int).tolist(),
            "trace": trace.astype(float).tolist(),
        },
    )


def summarize_metrics(
    rows: list[list],
    header: list[str],
    perturbation_time: int,
    saturation_threshold: float,
    target_damage_threshold: float,
    recovery_threshold: float,
) -> dict:
    idx = {name: i for i, name in enumerate(header)}
    mismatch_values = [float(row[idx["mismatch_rate"]]) for row in rows]
    balanced_values = [float(row[idx["balanced_integrity"]]) for row in rows]
    outside_values = [float(row[idx["outside_false_positive_rate"]]) for row in rows]
    target_values = [float(row[idx["target_true_positive_rate"]]) for row in rows]

    saturation_time = None
    for row in rows:
        if float(row[idx["outside_false_positive_rate"]]) >= saturation_threshold:
            saturation_time = int(row[idx["t"]])
            break

    target_damage_time = None
    for row in rows:
        if float(row[idx["target_true_positive_rate"]]) <= target_damage_threshold:
            target_damage_time = int(row[idx["t"]])
            break

    recovery_time = None
    for row in rows:
        t = int(row[idx["t"]])
        if (
            t >= perturbation_time
            and float(row[idx["balanced_integrity"]]) >= recovery_threshold
        ):
            recovery_time = t - perturbation_time
            break

    return {
        "final_mismatch_rate": float(rows[-1][idx["mismatch_rate"]]),
        "best_mismatch_rate": float(min(mismatch_values)),
        "mean_mismatch_rate": float(sum(mismatch_values) / len(mismatch_values)),
        "final_balanced_integrity": float(rows[-1][idx["balanced_integrity"]]),
        "best_balanced_integrity": float(max(balanced_values)),
        "mean_balanced_integrity": float(sum(balanced_values) / len(balanced_values)),
        "final_target_true_positive_rate": float(
            rows[-1][idx["target_true_positive_rate"]]
        ),
        "final_target_false_negative_rate": float(
            rows[-1][idx["target_false_negative_rate"]]
        ),
        "final_outside_false_positive_rate": float(
            rows[-1][idx["outside_false_positive_rate"]]
        ),
        "final_outside_true_negative_rate": float(
            rows[-1][idx["outside_true_negative_rate"]]
        ),
        "max_outside_false_positive_rate": float(max(outside_values)),
        "min_target_true_positive_rate": float(min(target_values)),
        "saturation_time": saturation_time,
        "target_damage_time": target_damage_time,
        "recovery_time": recovery_time,
    }


def make_line_plot(
    all_runs: list[dict], figures_dir: Path, metric: str, filename: str, title: str
) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 5.5))
    ax = fig.add_subplot(1, 1, 1)

    for item in all_runs:
        header = item["header"]
        rows = item["rows"]
        t_idx = header.index("t")
        y_idx = header.index(metric)
        xs = [int(row[t_idx]) for row in rows]
        ys = [float(row[y_idx]) for row in rows]
        ax.plot(xs, ys, label=f"{item['condition_id']}/s{item['seed']}")

    ax.set_title(title)
    ax.set_xlabel("timestep")
    ax.set_ylabel(metric)
    ax.grid(True)
    ax.legend(fontsize=6, ncol=3)
    fig.tight_layout()
    fig.savefig(str(figures_dir / filename), dpi=160)
    plt.close(fig)


def make_pareto_plot(summaries: list[dict], figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(1, 1, 1)

    for s in summaries:
        ax.scatter(
            float(s["final_outside_false_positive_rate"]),
            float(s["final_target_true_positive_rate"]),
        )
        ax.text(
            float(s["final_outside_false_positive_rate"]),
            float(s["final_target_true_positive_rate"]),
            str(s["condition_id"]),
            fontsize=7,
        )

    ax.set_title("Experiment 03 preservation-suppression plane")
    ax.set_xlabel("outside false positive rate")
    ax.set_ylabel("target true positive rate")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(str(figures_dir / "pareto_tpr_fpr.png"), dpi=160)
    plt.close(fig)


def run_condition(
    cfg: dict,
    condition: dict,
    seed: int,
    run_dir: Path,
    logger: Logger,
    progress: Progress,
) -> dict:
    height = int(cfg["canvas"]["height"])
    width = int(cfg["canvas"]["width"])
    steps = int(cfg["run"]["steps"])
    rng = np.random.default_rng(seed)

    condition_id = str(condition["id"])
    condition_dir = run_dir / "conditions" / f"{condition_id}_seed_{seed}"
    snapshots_dir = condition_dir / "snapshots"
    condition_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    target = make_target(height, width, cfg["target"])
    canvas = target.copy()
    trace = np.zeros((height, width), dtype=np.float64)

    operator_count = int(cfg["operator"]["count"])
    radius = int(cfg["operator"]["radius"])
    operators = init_operators(operator_count, height, width, rng)
    memory = np.zeros(operator_count, dtype=np.int32)

    operator_type = str(condition["operator"])
    action_space = str(condition["action_space"])
    suppression = float(condition["suppression"])
    perturbation_time = int(cfg["perturbation"]["time"])

    events = []
    rows = []
    header = [
        "t",
        "mismatch_rate",
        "target_true_positive_rate",
        "target_false_negative_rate",
        "outside_false_positive_rate",
        "outside_true_negative_rate",
        "balanced_integrity",
        "actions_total",
        "proposal_total",
        "applied_actions",
        "flips_total",
        "write_0_total",
        "write_1_total",
        "useful_cells",
        "harmful_cells",
        "neutral_cells",
        "conflict_attempts",
        "suppression_triggers",
        "positive_trace_mass",
        "negative_trace_mass",
        "active_l2_memory",
        "perturbed",
    ]

    events.append(
        {
            "event": "condition_start",
            "condition_id": condition_id,
            "seed": int(seed),
            "operator": operator_type,
            "action_space": action_space,
            "suppression": suppression,
        }
    )

    if bool(cfg["output"]["save_initial_snapshot"]):
        save_snapshot(
            snapshots_dir / "t_0000_initial.json", 0, canvas, target, trace, "initial"
        )
        events.append(
            {"event": "snapshot_saved", "t": 0, "path": "snapshots/t_0000_initial.json"}
        )

    for t in range(steps + 1):
        perturbed = 0

        if t == perturbation_time:
            changed = apply_random_noise(
                canvas,
                float(cfg["perturbation"]["strength"]),
                str(cfg["perturbation"]["scope"]),
                target,
                rng,
            )
            perturbed = 1
            events.append(
                {"event": "perturbation_applied", "t": int(t), "changed_cells": changed}
            )

            if bool(cfg["output"]["save_perturbation_snapshot"]):
                snapshot_name = f"t_{t:04d}_after_perturbation.json"
                save_snapshot(
                    snapshots_dir / snapshot_name,
                    t,
                    canvas,
                    target,
                    trace,
                    "after_perturbation",
                )
                events.append(
                    {
                        "event": "snapshot_saved",
                        "t": int(t),
                        "path": f"snapshots/{snapshot_name}",
                    }
                )

        proposals = defaultdict(list)
        suppression_triggers = 0

        if t < steps:
            if str(cfg["operator"]["move"]) == "random_walk":
                move_operators(operators, height, width, rng)

            memory = np.maximum(memory - 1, 0)

            for i, pos in enumerate(operators):
                x = int(pos[0])
                y = int(pos[1])
                action, delta, suppressed = choose_action(
                    canvas,
                    trace,
                    int(memory[i]),
                    x,
                    y,
                    radius,
                    operator_type,
                    action_space,
                    suppression,
                    cfg["operator"],
                    rng,
                )

                if action != "stay":
                    proposals[(x, y)].append(delta)

                if suppressed:
                    suppression_triggers += 1

            before = canvas.copy()

            if action_space == "write":
                action_stats = resolve_write(canvas, proposals)
            else:
                action_stats = resolve_flip(canvas, proposals)

            useful_cells, harmful_cells, neutral_cells = compute_action_quality(
                before, canvas, target
            )
            update_trace(
                trace,
                proposals,
                int(cfg["trace"]["duration"]),
                float(cfg["trace"]["decay"]),
            )

            if operator_type == "l2s":
                for i, pos in enumerate(operators):
                    x = int(pos[0])
                    y = int(pos[1])
                    if before[x, y] != canvas[x, y]:
                        memory[i] = int(cfg["operator"]["l2_memory_steps"])
        else:
            action_stats = {
                "applied_actions": 0,
                "flips_total": 0,
                "write_0_total": 0,
                "write_1_total": 0,
                "conflict_attempts": 0,
            }
            useful_cells = 0
            harmful_cells = 0
            neutral_cells = int(canvas.size)

        m = compute_metrics(canvas, target)

        rows.append(
            [
                int(t),
                m["mismatch_rate"],
                m["target_true_positive_rate"],
                m["target_false_negative_rate"],
                m["outside_false_positive_rate"],
                m["outside_true_negative_rate"],
                m["balanced_integrity"],
                int(operator_count),
                int(sum(len(v) for v in proposals.values())),
                int(action_stats["applied_actions"]),
                int(action_stats["flips_total"]),
                int(action_stats["write_0_total"]),
                int(action_stats["write_1_total"]),
                int(useful_cells),
                int(harmful_cells),
                int(neutral_cells),
                int(action_stats["conflict_attempts"]),
                int(suppression_triggers),
                float(np.maximum(trace, 0.0).sum()),
                float(np.maximum(-trace, 0.0).sum()),
                int((memory > 0).sum()),
                int(perturbed),
            ]
        )

        progress.step(1)

    if bool(cfg["output"]["save_final_snapshot"]):
        snapshot_name = f"t_{steps:04d}_final.json"
        save_snapshot(
            snapshots_dir / snapshot_name, steps, canvas, target, trace, "final"
        )
        events.append(
            {
                "event": "snapshot_saved",
                "t": int(steps),
                "path": f"snapshots/{snapshot_name}",
            }
        )

    summary = summarize_metrics(
        rows,
        header,
        perturbation_time,
        float(cfg["metrics"]["saturation_threshold"]),
        float(cfg["metrics"]["target_damage_threshold"]),
        float(cfg["metrics"]["recovery_threshold"]),
    )

    idx = {name: i for i, name in enumerate(header)}
    proposal_total = sum(int(row[idx["proposal_total"]]) for row in rows)
    applied_total = sum(int(row[idx["applied_actions"]]) for row in rows)
    suppression_total = sum(int(row[idx["suppression_triggers"]]) for row in rows)
    write_0_total = sum(int(row[idx["write_0_total"]]) for row in rows)
    write_1_total = sum(int(row[idx["write_1_total"]]) for row in rows)
    flips_total = sum(int(row[idx["flips_total"]]) for row in rows)
    conflict_total = sum(int(row[idx["conflict_attempts"]]) for row in rows)

    summary.update(
        {
            "condition_id": condition_id,
            "group": str(condition["group"]),
            "seed": int(seed),
            "operator": operator_type,
            "action_space": action_space,
            "suppression": suppression,
            "condition_dir": str(condition_dir.relative_to(run_dir)),
            "metrics_path": str((condition_dir / "metrics.csv").relative_to(run_dir)),
            "events_path": str((condition_dir / "events.jsonl").relative_to(run_dir)),
            "proposal_total": int(proposal_total),
            "applied_total": int(applied_total),
            "suppression_total": int(suppression_total),
            "suppression_activation_rate": float(suppression_total / proposal_total)
            if proposal_total
            else 0.0,
            "flip_total": int(flips_total),
            "write_0_total": int(write_0_total),
            "write_1_total": int(write_1_total),
            "conflict_total": int(conflict_total),
        }
    )

    events.append({"event": "condition_finished", "t": int(steps), "summary": summary})

    write_csv(str(condition_dir / "metrics.csv"), rows, header=header)
    write_jsonl(str(condition_dir / "events.jsonl"), events)
    write_json(str(condition_dir / "summary.json"), summary)

    logger.info(
        jline(
            "condition",
            condition_id,
            "finish",
            seed=int(seed),
            final_balanced_integrity=summary["final_balanced_integrity"],
            final_target_true_positive_rate=summary["final_target_true_positive_rate"],
            final_outside_false_positive_rate=summary[
                "final_outside_false_positive_rate"
            ],
        )
    )

    return {
        "condition_id": condition_id,
        "seed": int(seed),
        "header": header,
        "rows": rows,
        "summary": summary,
    }


def main() -> int:
    root = repo_root()
    config_path = root / "config" / "tests" / "exp_03.yaml"
    cfg = read_yaml(str(config_path))

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
            logger.info(
                jline("condition", str(condition["id"]), "start", seed=int(seed))
            )
            all_runs.append(
                run_condition(cfg, condition, seed, run_dir, logger, progress)
            )

    progress.finish()

    summary_header = [
        "condition_id",
        "group",
        "seed",
        "operator",
        "action_space",
        "suppression",
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
        "suppression_total",
        "suppression_activation_rate",
        "flip_total",
        "write_0_total",
        "write_1_total",
        "conflict_total",
        "metrics_path",
        "events_path",
    ]

    summaries = [item["summary"] for item in all_runs]
    summary_rows = [[s.get(k) for k in summary_header] for s in summaries]

    write_csv(
        str(run_dir / "conditions_summary.csv"), summary_rows, header=summary_header
    )
    write_json(str(run_dir / "conditions_summary.json"), summaries)

    if bool(cfg["output"]["make_plot"]):
        make_line_plot(
            all_runs,
            figures_dir,
            "balanced_integrity",
            "balanced_integrity.png",
            "Experiment 03 balanced integrity",
        )
        make_line_plot(
            all_runs,
            figures_dir,
            "target_true_positive_rate",
            "target_true_positive_rate.png",
            "Experiment 03 target true positive rate",
        )
        make_line_plot(
            all_runs,
            figures_dir,
            "outside_false_positive_rate",
            "outside_false_positive_rate.png",
            "Experiment 03 outside false positive rate",
        )
        make_line_plot(
            all_runs,
            figures_dir,
            "suppression_triggers",
            "suppression_triggers.png",
            "Experiment 03 suppression triggers",
        )
        make_pareto_plot(summaries, figures_dir)

    run_summary = {
        "name": cfg["name"],
        "run_dir": str(run_dir),
        "fingerprint": meta.get("fingerprint"),
        "condition_count": len(conditions),
        "seed_count": len(seeds),
        "conditions_summary_path": "conditions_summary.csv",
        "conditions_summary_json_path": "conditions_summary.json",
        "figures": [
            "figures/balanced_integrity.png",
            "figures/target_true_positive_rate.png",
            "figures/outside_false_positive_rate.png",
            "figures/suppression_triggers.png",
            "figures/pareto_tpr_fpr.png",
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
