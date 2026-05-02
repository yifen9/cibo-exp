from pathlib import Path
import sys
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


def choose_reactive_action(
    canvas: np.ndarray,
    x: int,
    y: int,
    radius: int,
    birth_threshold: int,
    death_threshold: int,
) -> str:
    window = local_window(canvas, x, y, radius)
    center = int(canvas[x, y])
    active = int(window.sum()) - center

    if center == 0 and active >= birth_threshold:
        return "flip"

    if center == 1 and active <= death_threshold:
        return "flip"

    return "stay"


def apply_random_noise(
    canvas: np.ndarray,
    strength: float,
    scope: str,
    target: np.ndarray,
    rng: np.random.Generator,
) -> int:
    height, width = canvas.shape

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


def compute_metrics(canvas: np.ndarray, target: np.ndarray) -> dict:
    mismatch = canvas != target
    target_mask = target == 1
    outside_mask = target == 0

    target_total = int(target_mask.sum())
    outside_total = int(outside_mask.sum())

    target_true_positive = int(((canvas == 1) & target_mask).sum())
    outside_false_positive = int(((canvas == 1) & outside_mask).sum())

    return {
        "mismatch_rate": float(mismatch.mean()),
        "target_true_positive_rate": float(target_true_positive / target_total)
        if target_total
        else 0.0,
        "outside_false_positive_rate": float(outside_false_positive / outside_total)
        if outside_total
        else 0.0,
    }


def save_snapshot(
    path: Path, t: int, canvas: np.ndarray, target: np.ndarray, label: str
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
        },
    )


def make_plot(metrics_rows: list[list], header: list[str], path: Path) -> bool:
    import matplotlib.pyplot as plt

    t_idx = header.index("t")
    mismatch_idx = header.index("mismatch_rate")

    xs = [int(row[t_idx]) for row in metrics_rows]
    ys = [float(row[mismatch_idx]) for row in metrics_rows]

    fig = plt.figure(figsize=(8, 4.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(xs, ys)
    ax.set_title("Experiment 01 mismatch rate")
    ax.set_xlabel("timestep")
    ax.set_ylabel("mismatch rate")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(str(path), dpi=160)
    plt.close(fig)
    return True


def main() -> int:
    root = repo_root()
    config_path = root / "config" / "tests" / "exp_01.yaml"
    cfg = read_yaml(str(config_path))

    env_path = root / "uv.lock"

    meta = build_meta(
        params=cfg,
        env=str(env_path),
        script=str(Path(__file__).resolve()),
        src=str(root / "src"),
    )

    output_root = root / cfg["output"]["root"]
    run_dir = Path(build_version_dir(str(output_root), meta))
    snapshots_dir = run_dir / "snapshots"
    figures_dir = run_dir / "figures"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    write_yaml(str(run_dir / "config.yaml"), cfg)

    audit = Audit.create(str(run_dir), meta)
    logger = Logger(sinks=[ConsoleSink(transient=False), audit])
    logger.info(jline("run", cfg["name"], "start", run_dir=str(run_dir)))

    events = []
    metrics_rows = []
    metrics_header = [
        "t",
        "mismatch_rate",
        "target_true_positive_rate",
        "outside_false_positive_rate",
        "actions_total",
        "flips_total",
        "useful_flips",
        "harmful_flips",
        "perturbed",
    ]

    seed = int(cfg["run"]["seed"])
    steps = int(cfg["run"]["steps"])
    rng = np.random.default_rng(seed)

    height = int(cfg["canvas"]["height"])
    width = int(cfg["canvas"]["width"])

    target = make_target(height, width, cfg["target"])
    canvas = target.copy()
    trace = np.zeros((height, width), dtype=np.int32)

    operator_count = int(cfg["operator"]["count"])
    radius = int(cfg["operator"]["radius"])
    birth_threshold = int(cfg["operator"]["birth_threshold"])
    death_threshold = int(cfg["operator"]["death_threshold"])
    operators = init_operators(operator_count, height, width, rng)

    trace_duration = int(cfg["trace"]["duration"])
    perturbation_time = int(cfg["perturbation"]["time"])

    events.append({"event": "run_start", "t": 0, "seed": seed, "run_dir": str(run_dir)})
    events.append(
        {"event": "target_initialized", "t": 0, "target_type": cfg["target"]["type"]}
    )

    if bool(cfg["output"].get("save_initial_snapshot", True)):
        save_snapshot(
            snapshots_dir / "t_0000_initial.json", 0, canvas, target, "initial"
        )
        events.append(
            {"event": "snapshot_saved", "t": 0, "path": "snapshots/t_0000_initial.json"}
        )

    progress = Progress(logger=logger, name=cfg["name"], total=steps + 1)
    progress.start()

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

            if bool(cfg["output"].get("save_perturbation_snapshot", True)):
                snapshot_name = f"t_{t:04d}_after_perturbation.json"
                save_snapshot(
                    snapshots_dir / snapshot_name,
                    t,
                    canvas,
                    target,
                    "after_perturbation",
                )
                events.append(
                    {
                        "event": "snapshot_saved",
                        "t": int(t),
                        "path": f"snapshots/{snapshot_name}",
                    }
                )

        actions_total = int(operator_count)
        flips_total = 0
        useful_flips = 0
        harmful_flips = 0

        if t < steps:
            if cfg["operator"].get("move", "random_walk") == "random_walk":
                move_operators(operators, height, width, rng)

            for x, y in operators:
                action = choose_reactive_action(
                    canvas, int(x), int(y), radius, birth_threshold, death_threshold
                )

                if action == "flip" and trace[int(x), int(y)] == 0:
                    before = int(canvas[int(x), int(y)] != target[int(x), int(y)])
                    canvas[int(x), int(y)] = 1 - canvas[int(x), int(y)]
                    after = int(canvas[int(x), int(y)] != target[int(x), int(y)])
                    trace[int(x), int(y)] = trace_duration
                    flips_total += 1

                    if after < before:
                        useful_flips += 1
                    elif after > before:
                        harmful_flips += 1

            trace = np.maximum(trace - 1, 0)

        m = compute_metrics(canvas, target)
        metrics_rows.append(
            [
                int(t),
                m["mismatch_rate"],
                m["target_true_positive_rate"],
                m["outside_false_positive_rate"],
                actions_total,
                flips_total,
                useful_flips,
                harmful_flips,
                perturbed,
            ]
        )

        progress.step(1)

    progress.finish()

    final_snapshot_name = f"t_{steps:04d}_final.json"

    if bool(cfg["output"].get("save_final_snapshot", True)):
        save_snapshot(
            snapshots_dir / final_snapshot_name, steps, canvas, target, "final"
        )
        events.append(
            {
                "event": "snapshot_saved",
                "t": int(steps),
                "path": f"snapshots/{final_snapshot_name}",
            }
        )

    write_csv(str(run_dir / "metrics.csv"), metrics_rows, header=metrics_header)

    mismatch_idx = metrics_header.index("mismatch_rate")
    target_idx = metrics_header.index("target_true_positive_rate")
    outside_idx = metrics_header.index("outside_false_positive_rate")

    mismatch_values = [float(row[mismatch_idx]) for row in metrics_rows]

    figure_written = False

    if bool(cfg["output"].get("make_plot", True)):
        figure_written = make_plot(
            metrics_rows, metrics_header, figures_dir / "mismatch_rate.png"
        )

    summary = {
        "name": cfg["name"],
        "run_dir": str(run_dir),
        "fingerprint": meta.get("fingerprint"),
        "seed": seed,
        "steps": steps,
        "final_mismatch_rate": float(metrics_rows[-1][mismatch_idx]),
        "best_mismatch_rate": float(min(mismatch_values)),
        "mean_mismatch_rate": float(sum(mismatch_values) / len(mismatch_values)),
        "final_target_true_positive_rate": float(metrics_rows[-1][target_idx]),
        "final_outside_false_positive_rate": float(metrics_rows[-1][outside_idx]),
        "perturbation_time": perturbation_time,
        "metrics_path": "metrics.csv",
        "events_path": "events.jsonl",
        "figure_path": "figures/mismatch_rate.png" if figure_written else None,
        "success": True,
    }

    events.append({"event": "run_finished", "t": int(steps), "summary": summary})

    write_jsonl(str(run_dir / "events.jsonl"), events)
    write_json(str(run_dir / "summary.json"), summary)

    logger.info(jline("run", cfg["name"], "finish", run_dir=str(run_dir)))
    audit.finish_success()

    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
