from pathlib import Path
import math

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


def local_window(canvas: np.ndarray, x: int, y: int, radius: int) -> np.ndarray:
    height, width = canvas.shape
    x0 = max(0, x - radius)
    x1 = min(height, x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(width, y + radius + 1)
    return canvas[x0:x1, y0:y1]


def action_to_delta(action: str) -> int:
    if action == "write_1":
        return 1
    if action == "write_0":
        return -1
    if action == "flip_to_1":
        return 1
    if action == "flip_to_0":
        return -1
    return 0


def choose_l1_action(
    canvas: np.ndarray, x: int, y: int, radius: int, action_space: str, cfg: dict
) -> str:
    window = local_window(canvas, x, y, radius)
    center = int(canvas[x, y])
    active = int(window.sum()) - center

    if center == 0 and active >= int(cfg["birth_threshold"]):
        return "write_1" if action_space == "write" else "flip_to_1"

    if center == 1 and active <= int(cfg["death_threshold"]):
        return "write_0" if action_space == "write" else "flip_to_0"

    return "stay"


def choose_l1s_action(
    canvas: np.ndarray, x: int, y: int, radius: int, action_space: str, cfg: dict
) -> str:
    window = local_window(canvas, x, y, radius)
    center = int(canvas[x, y])
    active = int(window.sum()) - center

    birth_min = int(cfg["birth_min"])
    birth_max = int(cfg["birth_max"])
    survive_min = int(cfg["survive_min"])
    survive_max = int(cfg["survive_max"])

    if center == 0 and birth_min <= active <= birth_max:
        return "write_1" if action_space == "write" else "flip_to_1"

    if center == 1 and (active < survive_min or active > survive_max):
        return "write_0" if action_space == "write" else "flip_to_0"

    return "stay"


def choose_l2_action(
    canvas: np.ndarray,
    x: int,
    y: int,
    radius: int,
    action_space: str,
    cfg: dict,
    refractory: int,
) -> str:
    if refractory > 0:
        return "stay"

    return choose_l1s_action(canvas, x, y, radius, action_space, cfg)


def apply_action(canvas: np.ndarray, x: int, y: int, action: str) -> None:
    if action == "write_1":
        canvas[x, y] = 1
    elif action == "write_0":
        canvas[x, y] = 0
    elif action == "flip_to_1" or action == "flip_to_0":
        canvas[x, y] = 1 - canvas[x, y]


def apply_random_noise(
    canvas: np.ndarray,
    strength: float,
    scope: str,
    target: np.ndarray,
    rng: np.random.Generator,
) -> int:
    if scope == "target":
        cells = np.argwhere(target == 1)
    elif scope == "all":
        cells = np.argwhere(np.ones_like(canvas, dtype=bool))
    else:
        raise ValueError(f"unknown perturbation scope: {scope}")

    count = max(1, int(math.ceil(len(cells) * strength)))
    selected = cells[rng.choice(len(cells), size=count, replace=False)]

    for x, y in selected:
        canvas[x, y] = 1 - canvas[x, y]

    return int(count)


def decay_trace(trace: np.ndarray, permanent: bool) -> np.ndarray:
    if permanent:
        return trace
    return np.maximum(trace - 1, 0)


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
            "trace": trace.astype(int).tolist(),
        },
    )


def summarize_metrics(
    rows: list[list],
    header: list[str],
    perturbation_time: int,
    saturation_threshold: float,
    recovery_threshold: float,
) -> dict:
    idx = {name: i for i, name in enumerate(header)}
    mismatch_values = [float(row[idx["mismatch_rate"]]) for row in rows]
    balanced_values = [float(row[idx["balanced_integrity"]]) for row in rows]
    outside_values = [float(row[idx["outside_false_positive_rate"]]) for row in rows]

    saturation_time = None
    for row in rows:
        if float(row[idx["outside_false_positive_rate"]]) >= saturation_threshold:
            saturation_time = int(row[idx["t"]])
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
        "saturation_time": saturation_time,
        "recovery_time": recovery_time,
    }


def make_plot(
    all_runs: list[dict], figures_dir: Path, metric: str, filename: str, title: str
) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(1, 1, 1)

    for item in all_runs:
        header = item["header"]
        rows = item["rows"]
        t_idx = header.index("t")
        y_idx = header.index(metric)
        xs = [int(row[t_idx]) for row in rows]
        ys = [float(row[y_idx]) for row in rows]
        ax.plot(xs, ys, label=f"{item['condition_id']}/seed{item['seed']}")

    ax.set_title(title)
    ax.set_xlabel("timestep")
    ax.set_ylabel(metric)
    ax.grid(True)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(str(figures_dir / filename), dpi=160)
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
    trace = np.zeros((height, width), dtype=np.int32)
    last_delta = np.zeros((height, width), dtype=np.int8)

    operator_count = int(cfg["operator"]["count"])
    radius = int(cfg["operator"]["radius"])
    operators = init_operators(operator_count, height, width, rng)
    refractory = np.zeros(operator_count, dtype=np.int32)

    trace_key = str(condition["trace"])
    trace_duration = int(cfg["trace"]["durations"][trace_key])
    permanent_trace = trace_duration < 0
    operator_type = str(condition["operator"])
    action_space = str(condition["action_space"])
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
        "applied_actions",
        "flips_total",
        "write_0_total",
        "write_1_total",
        "useful_actions",
        "harmful_actions",
        "neutral_actions",
        "conflict_attempts",
        "blocked_by_trace",
        "perturbed",
    ]

    events.append(
        {
            "event": "condition_start",
            "condition_id": condition_id,
            "seed": int(seed),
            "trace": trace_key,
            "operator": operator_type,
            "action_space": action_space,
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

        applied_actions = 0
        flips_total = 0
        write_0_total = 0
        write_1_total = 0
        useful_actions = 0
        harmful_actions = 0
        neutral_actions = 0
        conflict_attempts = 0
        blocked_by_trace = 0

        if t < steps:
            if str(cfg["operator"]["move"]) == "random_walk":
                move_operators(operators, height, width, rng)

            refractory = np.maximum(refractory - 1, 0)

            for i, pos in enumerate(operators):
                x = int(pos[0])
                y = int(pos[1])

                if operator_type == "l1":
                    action = choose_l1_action(
                        canvas, x, y, radius, action_space, cfg["operator"]["l1"]
                    )
                elif operator_type == "l1s":
                    action = choose_l1s_action(
                        canvas, x, y, radius, action_space, cfg["operator"]["l1s"]
                    )
                elif operator_type == "l2":
                    action = choose_l2_action(
                        canvas,
                        x,
                        y,
                        radius,
                        action_space,
                        cfg["operator"]["l2"],
                        int(refractory[i]),
                    )
                else:
                    raise ValueError(f"unknown operator type: {operator_type}")

                delta = action_to_delta(action)

                if action != "stay" and trace[x, y] > 0:
                    if (
                        delta != 0
                        and last_delta[x, y] != 0
                        and delta != int(last_delta[x, y])
                    ):
                        conflict_attempts += 1
                    blocked_by_trace += 1
                    continue

                if action == "stay":
                    continue

                before = int(canvas[x, y] != target[x, y])
                apply_action(canvas, x, y, action)
                after = int(canvas[x, y] != target[x, y])

                if after < before:
                    useful_actions += 1
                elif after > before:
                    harmful_actions += 1
                else:
                    neutral_actions += 1

                applied_actions += 1

                if action == "flip_to_1" or action == "flip_to_0":
                    flips_total += 1
                elif action == "write_0":
                    write_0_total += 1
                elif action == "write_1":
                    write_1_total += 1

                if trace_duration != 0:
                    trace[x, y] = 1 if permanent_trace else trace_duration
                    last_delta[x, y] = delta

                if operator_type == "l2":
                    refractory[i] = int(cfg["operator"]["l2"]["refractory_steps"])

            trace = decay_trace(trace, permanent_trace)

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
                int(applied_actions),
                int(flips_total),
                int(write_0_total),
                int(write_1_total),
                int(useful_actions),
                int(harmful_actions),
                int(neutral_actions),
                int(conflict_attempts),
                int(blocked_by_trace),
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

    saturation_threshold = float(cfg["metrics"]["saturation_threshold"])
    recovery_threshold = float(cfg["metrics"]["recovery_threshold"])
    summary = summarize_metrics(
        rows, header, perturbation_time, saturation_threshold, recovery_threshold
    )

    summary.update(
        {
            "condition_id": condition_id,
            "seed": int(seed),
            "trace": trace_key,
            "trace_duration": int(trace_duration),
            "operator": operator_type,
            "action_space": action_space,
            "condition_dir": str(condition_dir.relative_to(run_dir)),
            "metrics_path": str((condition_dir / "metrics.csv").relative_to(run_dir)),
            "events_path": str((condition_dir / "events.jsonl").relative_to(run_dir)),
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
    config_path = root / "config" / "tests" / "exp_02.yaml"
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
            result = run_condition(cfg, condition, seed, run_dir, logger, progress)
            all_runs.append(result)

    progress.finish()

    summary_header = [
        "condition_id",
        "seed",
        "trace",
        "trace_duration",
        "operator",
        "action_space",
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
        "saturation_time",
        "recovery_time",
        "metrics_path",
        "events_path",
    ]

    summary_rows = []

    for item in all_runs:
        s = item["summary"]
        summary_rows.append([s.get(k) for k in summary_header])

    write_csv(
        str(run_dir / "conditions_summary.csv"), summary_rows, header=summary_header
    )
    write_json(
        str(run_dir / "conditions_summary.json"), [item["summary"] for item in all_runs]
    )

    if bool(cfg["output"]["make_plot"]):
        make_plot(
            all_runs,
            figures_dir,
            "mismatch_rate",
            "mismatch_rate.png",
            "Experiment 02 mismatch rate",
        )
        make_plot(
            all_runs,
            figures_dir,
            "outside_false_positive_rate",
            "outside_false_positive_rate.png",
            "Experiment 02 outside false positive rate",
        )
        make_plot(
            all_runs,
            figures_dir,
            "balanced_integrity",
            "balanced_integrity.png",
            "Experiment 02 balanced integrity",
        )

    run_summary = {
        "name": cfg["name"],
        "run_dir": str(run_dir),
        "fingerprint": meta.get("fingerprint"),
        "condition_count": len(conditions),
        "seed_count": len(seeds),
        "conditions_summary_path": "conditions_summary.csv",
        "conditions_summary_json_path": "conditions_summary.json",
        "figures": [
            "figures/mismatch_rate.png",
            "figures/outside_false_positive_rate.png",
            "figures/balanced_integrity.png",
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
