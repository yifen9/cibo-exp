from pathlib import Path
from collections import defaultdict, deque
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


def get_cell(canvas: np.ndarray, x: int, y: int) -> int:
    h, w = canvas.shape
    if x < 0 or x >= h or y < 0 or y >= w:
        return 0
    return int(canvas[x, y])


def local_window(canvas: np.ndarray, x: int, y: int, radius: int) -> np.ndarray:
    h, w = canvas.shape
    x0 = max(0, x - radius)
    x1 = min(h, x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(w, y + radius + 1)
    return canvas[x0:x1, y0:y1]


def neighbor_count_support(canvas: np.ndarray, x: int, y: int, radius: int) -> float:
    w = local_window(canvas, x, y, radius)
    center = int(canvas[x, y])
    total = int(w.sum()) - center
    denom = max(1, w.size - 1)
    return float(total / denom)


def line_continuity_support(canvas: np.ndarray, x: int, y: int) -> float:
    left = get_cell(canvas, x, y - 1)
    right = get_cell(canvas, x, y + 1)
    up = get_cell(canvas, x - 1, y)
    down = get_cell(canvas, x + 1, y)
    ul = get_cell(canvas, x - 1, y - 1)
    dr = get_cell(canvas, x + 1, y + 1)
    ur = get_cell(canvas, x - 1, y + 1)
    dl = get_cell(canvas, x + 1, y - 1)
    return float(max(left + right, up + down, ul + dr, ur + dl) / 2.0)


def boundary_consistency_support(canvas: np.ndarray, x: int, y: int) -> float:
    left = get_cell(canvas, x, y - 1)
    right = get_cell(canvas, x, y + 1)
    up = get_cell(canvas, x - 1, y)
    down = get_cell(canvas, x + 1, y)
    straight = max(left + right, up + down) / 2.0
    corner = max(left + up, left + down, right + up, right + down) / 2.0
    return float(max(straight, corner))


def connected_component_support(
    canvas: np.ndarray, x: int, y: int, radius: int
) -> float:
    x0 = max(0, x - radius)
    x1 = min(canvas.shape[0], x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(canvas.shape[1], y + radius + 1)
    w = canvas[x0:x1, y0:y1]
    cx = x - x0
    cy = y - y0

    if int(w[cx, cy]) == 0:
        return float(min(1.0, int(w.sum()) / max(1, radius + 1)))

    seen = np.zeros_like(w, dtype=bool)
    q = deque([(cx, cy)])
    seen[cx, cy] = True
    size = 0

    while q:
        i, j = q.popleft()
        size += 1
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni = i + di
            nj = j + dj
            if (
                0 <= ni < w.shape[0]
                and 0 <= nj < w.shape[1]
                and not seen[ni, nj]
                and int(w[ni, nj]) == 1
            ):
                seen[ni, nj] = True
                q.append((ni, nj))

    return float(min(1.0, size / max(1, radius * 2 + 1)))


def structural_support(canvas: np.ndarray, x: int, y: int, radius: int) -> float:
    a = neighbor_count_support(canvas, x, y, radius)
    b = line_continuity_support(canvas, x, y)
    c = boundary_consistency_support(canvas, x, y)
    d = connected_component_support(canvas, x, y, radius)
    return float(0.15 * a + 0.30 * b + 0.35 * c + 0.20 * d)


def gap_evidence(canvas: np.ndarray, x: int, y: int) -> float:
    left = get_cell(canvas, x, y - 1)
    right = get_cell(canvas, x, y + 1)
    up = get_cell(canvas, x - 1, y)
    down = get_cell(canvas, x + 1, y)
    horizontal_gap = left == 1 and right == 1
    vertical_gap = up == 1 and down == 1
    corner_gap = (left or right) and (up or down)

    if horizontal_gap or vertical_gap:
        return 1.0

    if corner_gap:
        return 0.75

    return 0.0


def repair_evidence(
    canvas: np.ndarray, trace: np.ndarray, x: int, y: int, radius: int
) -> float:
    if int(canvas[x, y]) == 1:
        return structural_support(canvas, x, y, radius)

    old = int(canvas[x, y])
    canvas[x, y] = 1
    s = structural_support(canvas, x, y, radius)
    canvas[x, y] = old

    g = gap_evidence(canvas, x, y)
    x0 = max(0, x - radius)
    x1 = min(canvas.shape[0], x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(canvas.shape[1], y + radius + 1)
    tw = trace[x0:x1, y0:y1]
    positive = float(np.maximum(tw, 0.0).sum())
    negative = float(np.maximum(-tw, 0.0).sum())
    trace_score = 1.0 if positive > negative and positive > 0 else 0.0

    return float(0.45 * s + 0.40 * g + 0.15 * trace_score)


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


def apply_local_gap_attack(canvas: np.ndarray, target: np.ndarray, cfg: dict) -> int:
    cells = np.argwhere(target == 1)
    side = str(cfg.get("side", "top"))
    length = int(cfg.get("length", 8))

    if side == "top":
        row = int(cells[:, 0].min())
        line = cells[cells[:, 0] == row]
        line = line[np.argsort(line[:, 1])]
    elif side == "bottom":
        row = int(cells[:, 0].max())
        line = cells[cells[:, 0] == row]
        line = line[np.argsort(line[:, 1])]
    elif side == "left":
        col = int(cells[:, 1].min())
        line = cells[cells[:, 1] == col]
        line = line[np.argsort(line[:, 0])]
    elif side == "right":
        col = int(cells[:, 1].max())
        line = cells[cells[:, 1] == col]
        line = line[np.argsort(line[:, 0])]
    else:
        raise ValueError(f"unknown local gap attack side: {side}")

    start = max(0, (len(line) - length) // 2)
    selected = line[start : start + length]

    for x, y in selected:
        canvas[int(x), int(y)] = 0

    return int(len(selected))


def resolve_write(canvas: np.ndarray, proposals: dict) -> dict:
    stats = {
        "applied_actions": 0,
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
            if int(canvas[x, y]) != 1:
                stats["applied_actions"] += 1
            canvas[x, y] = 1
            stats["write_1_total"] += 1
        elif write_0 > write_1:
            x, y = cell
            if int(canvas[x, y]) != 0:
                stats["applied_actions"] += 1
            canvas[x, y] = 0
            stats["write_0_total"] += 1

    return stats


def update_trace(
    trace: np.ndarray, proposals: dict, duration: int, decay: float
) -> None:
    trace *= float(decay)

    for (x, y), values in proposals.items():
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


def classify_repair_action(
    before: np.ndarray, target: np.ndarray, x: int, y: int
) -> str:
    if int(before[x, y]) == 0 and int(target[x, y]) == 1:
        return "true_repair"
    if int(before[x, y]) == 0 and int(target[x, y]) == 0:
        return "false_repair"
    return "non_repair"


def classify_suppression_action(
    before: np.ndarray, target: np.ndarray, x: int, y: int
) -> str:
    if int(before[x, y]) == 1 and int(target[x, y]) == 0:
        return "true_suppression"
    if int(before[x, y]) == 1 and int(target[x, y]) == 1:
        return "false_suppression"
    return "non_suppression"


def compute_action_quality(
    before: np.ndarray, after: np.ndarray, target: np.ndarray
) -> tuple[int, int, int]:
    before_err = before != target
    after_err = after != target
    useful = int((before_err & ~after_err).sum())
    harmful = int((~before_err & after_err).sum())
    neutral = int((before_err == after_err).sum())
    return useful, harmful, neutral


def expansion_count(before: np.ndarray, after: np.ndarray, target: np.ndarray) -> int:
    outside_new = (before == 0) & (after == 1) & (target == 0)
    count = 0

    for x, y in np.argwhere(outside_new):
        x = int(x)
        y = int(y)
        if (
            get_cell(before, x - 1, y) == 1
            or get_cell(before, x + 1, y) == 1
            or get_cell(before, x, y - 1) == 1
            or get_cell(before, x, y + 1) == 1
        ):
            count += 1

    return int(count)


def gap_closure_count(before: np.ndarray, after: np.ndarray, target: np.ndarray) -> int:
    return int(((before == 0) & (after == 1) & (target == 1)).sum())


def update_temporal(
    mechanism: str,
    evidence: float,
    theta: float,
    x: int,
    y: int,
    t: int,
    k: int,
    m: int,
    gamma: float,
    consecutive: np.ndarray,
    history: np.ndarray,
    decayed: np.ndarray,
) -> tuple[bool, float]:
    high = evidence >= theta

    if mechanism == "consecutive":
        consecutive[x, y] = consecutive[x, y] + 1 if high else 0
        return int(consecutive[x, y]) >= k, float(consecutive[x, y])

    if mechanism == "sliding":
        slot = t % max(1, k)
        history[x, y, slot] = 1 if high else 0
        score = int(history[x, y, :k].sum())
        return score >= m, float(score / max(1, k))

    if mechanism == "decay":
        decayed[x, y] = gamma * decayed[x, y] + (1.0 - gamma) * evidence
        return float(decayed[x, y]) >= theta, float(decayed[x, y])

    raise ValueError(f"unknown temporal mechanism: {mechanism}")


def choose_action(
    canvas: np.ndarray,
    trace: np.ndarray,
    memory: int,
    temporal_pass: bool,
    x: int,
    y: int,
    theta_preserve: float,
    cfg: dict,
    rng: np.random.Generator,
) -> dict:
    radius = int(cfg["radius"])
    center = int(canvas[x, y])
    support = structural_support(canvas, x, y, radius)
    evidence = repair_evidence(canvas, trace, x, y, radius)
    suppression = float(cfg["suppression"])
    preserve_pass = support >= theta_preserve

    if center == 1:
        if not preserve_pass and rng.random() < suppression:
            return {
                "action": "write_0",
                "delta": -1,
                "support": support,
                "evidence": evidence,
                "repair_pass": False,
                "preserve_pass": False,
            }

        return {
            "action": "stay",
            "delta": 0,
            "support": support,
            "evidence": evidence,
            "repair_pass": False,
            "preserve_pass": True,
        }

    if temporal_pass:
        return {
            "action": "write_1",
            "delta": 1,
            "support": support,
            "evidence": evidence,
            "repair_pass": True,
            "preserve_pass": False,
        }

    return {
        "action": "stay",
        "delta": 0,
        "support": support,
        "evidence": evidence,
        "repair_pass": False,
        "preserve_pass": False,
    }


def save_snapshot(
    path: Path,
    t: int,
    canvas: np.ndarray,
    target: np.ndarray,
    trace: np.ndarray,
    temporal_score: np.ndarray,
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
            "temporal_score": temporal_score.astype(float).tolist(),
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

    ax.set_title("Experiment 06 TPR-FPR plane")
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
    consecutive = np.zeros((height, width), dtype=np.int32)
    history = np.zeros((height, width, 4), dtype=np.uint8)
    decayed = np.zeros((height, width), dtype=np.float64)
    memory = np.zeros(int(cfg["operator"]["count"]), dtype=np.int32)

    operator_count = int(cfg["operator"]["count"])
    operators = init_operators(operator_count, height, width, rng)

    mechanism = str(condition["mechanism"])
    operator_type = str(condition["operator"])
    theta_repair = float(condition["theta_repair"])
    theta_preserve = float(cfg["operator"]["theta_preserve"])
    k = int(condition["k"])
    m = int(condition["m"])
    gamma = float(condition["gamma"])
    use_budget = bool(condition["budget"])
    perturbation_times = [int(p["time"]) for p in cfg["perturbation"]["schedule"]]

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
        "write_0_total",
        "write_1_total",
        "useful_cells",
        "harmful_cells",
        "neutral_cells",
        "conflict_attempts",
        "repair_pass_total",
        "repair_reject_total",
        "true_repair_total",
        "false_repair_total",
        "true_suppression_total",
        "false_suppression_total",
        "expansion_total",
        "gap_closure_total",
        "support_mean",
        "repair_evidence_mean",
        "temporal_score_mean",
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
            "mechanism": mechanism,
            "theta_repair": theta_repair,
            "k": k,
            "m": m,
            "gamma": gamma,
            "budget": use_budget,
        }
    )

    if bool(cfg["output"]["save_initial_snapshot"]):
        save_snapshot(
            snapshots_dir / "t_0000_initial.json",
            0,
            canvas,
            target,
            trace,
            decayed,
            "initial",
        )
        events.append(
            {"event": "snapshot_saved", "t": 0, "path": "snapshots/t_0000_initial.json"}
        )

    for t in range(steps + 1):
        perturbed = 0

        for p in cfg["perturbation"]["schedule"]:
            if t == int(p["time"]):
                if str(p["type"]) == "random_noise":
                    changed = apply_random_noise(
                        canvas, float(p["strength"]), str(p["scope"]), target, rng
                    )
                elif str(p["type"]) == "local_gap_attack":
                    changed = apply_local_gap_attack(canvas, target, p)
                else:
                    raise ValueError(f"unknown perturbation type: {p['type']}")

                perturbed = 1
                events.append(
                    {
                        "event": "perturbation_applied",
                        "t": int(t),
                        "type": str(p["type"]),
                        "changed_cells": int(changed),
                    }
                )

                if bool(cfg["output"]["save_perturbation_snapshot"]):
                    snapshot_name = f"t_{t:04d}_{p['type']}.json"
                    save_snapshot(
                        snapshots_dir / snapshot_name,
                        t,
                        canvas,
                        target,
                        trace,
                        decayed,
                        str(p["type"]),
                    )
                    events.append(
                        {
                            "event": "snapshot_saved",
                            "t": int(t),
                            "path": f"snapshots/{snapshot_name}",
                        }
                    )

        proposals = defaultdict(list)
        proposal_meta = defaultdict(list)

        repair_pass_total = 0
        repair_reject_total = 0
        support_values = []
        evidence_values = []
        temporal_values = []

        if t < steps:
            if str(cfg["operator"]["move"]) == "random_walk":
                move_operators(operators, height, width, rng)

            memory = np.maximum(memory - 1, 0)

            for i, pos in enumerate(operators):
                x = int(pos[0])
                y = int(pos[1])
                evidence = repair_evidence(
                    canvas, trace, x, y, int(cfg["operator"]["radius"])
                )

                temporal_pass, temporal_score = update_temporal(
                    mechanism,
                    evidence,
                    theta_repair,
                    x,
                    y,
                    t,
                    max(1, k),
                    max(1, m),
                    gamma,
                    consecutive,
                    history,
                    decayed,
                )

                if operator_type in {"l2sgt", "l2sgtd"} and int(memory[i]) > 0:
                    temporal_pass = temporal_pass and evidence >= theta_repair

                result = choose_action(
                    canvas,
                    trace,
                    int(memory[i]),
                    temporal_pass,
                    x,
                    y,
                    theta_preserve,
                    cfg["operator"],
                    rng,
                )

                support_values.append(float(result["support"]))
                evidence_values.append(float(result["evidence"]))
                temporal_values.append(float(temporal_score))

                if int(canvas[x, y]) == 0:
                    if bool(result["repair_pass"]):
                        repair_pass_total += 1
                    else:
                        repair_reject_total += 1

                if result["action"] != "stay":
                    proposals[(x, y)].append(int(result["delta"]))
                    proposal_meta[(x, y)].append(result)

            if use_budget:
                repair_cells = [
                    (cell, metas)
                    for cell, metas in proposal_meta.items()
                    if any(int(m["delta"]) > 0 for m in metas)
                ]
                budget = max(
                    1,
                    int(
                        math.ceil(
                            float(cfg["operator"]["repair_budget_ratio"])
                            * operator_count
                        )
                    ),
                )
                ranked = sorted(
                    repair_cells,
                    key=lambda item: max(float(m["evidence"]) for m in item[1]),
                    reverse=True,
                )
                allowed = {cell for cell, _ in ranked[:budget]}

                for cell, metas in list(proposal_meta.items()):
                    if cell not in allowed:
                        filtered = [m for m in metas if int(m["delta"]) <= 0]
                        removed = len(metas) - len(filtered)
                        repair_pass_total -= removed
                        repair_reject_total += removed
                        proposal_meta[cell] = filtered
                        proposals[cell] = [int(m["delta"]) for m in filtered]

            before = canvas.copy()
            action_stats = resolve_write(canvas, proposals)

            true_repair_total = 0
            false_repair_total = 0
            true_suppression_total = 0
            false_suppression_total = 0

            for cell, metas in proposal_meta.items():
                x, y = cell
                for item in metas:
                    if int(item["delta"]) > 0:
                        label = classify_repair_action(before, target, x, y)
                        if label == "true_repair":
                            true_repair_total += 1
                        elif label == "false_repair":
                            false_repair_total += 1
                    elif int(item["delta"]) < 0:
                        label = classify_suppression_action(before, target, x, y)
                        if label == "true_suppression":
                            true_suppression_total += 1
                        elif label == "false_suppression":
                            false_suppression_total += 1

            useful_cells, harmful_cells, neutral_cells = compute_action_quality(
                before, canvas, target
            )
            expansion_total = expansion_count(before, canvas, target)
            gap_closure_total = gap_closure_count(before, canvas, target)

            update_trace(
                trace,
                proposals,
                int(cfg["trace"]["duration"]),
                float(cfg["trace"]["decay"]),
            )

            if operator_type in {"l2sgt", "l2sgtd"}:
                for i, pos in enumerate(operators):
                    x = int(pos[0])
                    y = int(pos[1])
                    if before[x, y] != canvas[x, y]:
                        memory[i] = int(cfg["operator"]["l2_memory_steps"])

        else:
            action_stats = {
                "applied_actions": 0,
                "write_0_total": 0,
                "write_1_total": 0,
                "conflict_attempts": 0,
            }
            useful_cells = 0
            harmful_cells = 0
            neutral_cells = int(canvas.size)
            true_repair_total = 0
            false_repair_total = 0
            true_suppression_total = 0
            false_suppression_total = 0
            expansion_total = 0
            gap_closure_total = 0

        metrics = compute_metrics(canvas, target)

        rows.append(
            [
                int(t),
                metrics["mismatch_rate"],
                metrics["target_true_positive_rate"],
                metrics["target_false_negative_rate"],
                metrics["outside_false_positive_rate"],
                metrics["outside_true_negative_rate"],
                metrics["balanced_integrity"],
                int(operator_count),
                int(sum(len(v) for v in proposals.values())),
                int(action_stats["applied_actions"]),
                int(action_stats["write_0_total"]),
                int(action_stats["write_1_total"]),
                int(useful_cells),
                int(harmful_cells),
                int(neutral_cells),
                int(action_stats["conflict_attempts"]),
                int(repair_pass_total),
                int(repair_reject_total),
                int(true_repair_total),
                int(false_repair_total),
                int(true_suppression_total),
                int(false_suppression_total),
                int(expansion_total),
                int(gap_closure_total),
                float(sum(support_values) / len(support_values))
                if support_values
                else 0.0,
                float(sum(evidence_values) / len(evidence_values))
                if evidence_values
                else 0.0,
                float(sum(temporal_values) / len(temporal_values))
                if temporal_values
                else 0.0,
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
            snapshots_dir / snapshot_name,
            steps,
            canvas,
            target,
            trace,
            decayed,
            "final",
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
        min(perturbation_times),
        float(cfg["metrics"]["saturation_threshold"]),
        float(cfg["metrics"]["target_damage_threshold"]),
        float(cfg["metrics"]["recovery_threshold"]),
    )

    idx = {name: i for i, name in enumerate(header)}
    proposal_total = sum(int(row[idx["proposal_total"]]) for row in rows)
    applied_total = sum(int(row[idx["applied_actions"]]) for row in rows)
    write_0_total = sum(int(row[idx["write_0_total"]]) for row in rows)
    write_1_total = sum(int(row[idx["write_1_total"]]) for row in rows)
    repair_pass_total = sum(int(row[idx["repair_pass_total"]]) for row in rows)
    repair_reject_total = sum(int(row[idx["repair_reject_total"]]) for row in rows)
    true_repair_total = sum(int(row[idx["true_repair_total"]]) for row in rows)
    false_repair_total = sum(int(row[idx["false_repair_total"]]) for row in rows)
    true_suppression_total = sum(
        int(row[idx["true_suppression_total"]]) for row in rows
    )
    false_suppression_total = sum(
        int(row[idx["false_suppression_total"]]) for row in rows
    )
    expansion_total = sum(int(row[idx["expansion_total"]]) for row in rows)
    gap_closure_total = sum(int(row[idx["gap_closure_total"]]) for row in rows)
    repair_action_total = true_repair_total + false_repair_total
    suppression_action_total = true_suppression_total + false_suppression_total

    summary.update(
        {
            "condition_id": condition_id,
            "group": str(condition["group"]),
            "seed": int(seed),
            "operator": operator_type,
            "mechanism": mechanism,
            "theta_preserve": theta_preserve,
            "theta_repair": theta_repair,
            "k": int(k),
            "m": int(m),
            "gamma": float(gamma),
            "budget": bool(use_budget),
            "condition_dir": str(condition_dir.relative_to(run_dir)),
            "metrics_path": str((condition_dir / "metrics.csv").relative_to(run_dir)),
            "events_path": str((condition_dir / "events.jsonl").relative_to(run_dir)),
            "proposal_total": int(proposal_total),
            "applied_total": int(applied_total),
            "write_0_total": int(write_0_total),
            "write_1_total": int(write_1_total),
            "repair_pass_total": int(repair_pass_total),
            "repair_reject_total": int(repair_reject_total),
            "true_repair_total": int(true_repair_total),
            "false_repair_total": int(false_repair_total),
            "true_suppression_total": int(true_suppression_total),
            "false_suppression_total": int(false_suppression_total),
            "repair_precision": float(true_repair_total / repair_action_total)
            if repair_action_total
            else 0.0,
            "false_repair_rate": float(false_repair_total / repair_action_total)
            if repair_action_total
            else 0.0,
            "suppression_precision": float(
                true_suppression_total / suppression_action_total
            )
            if suppression_action_total
            else 0.0,
            "false_suppression_rate": float(
                false_suppression_total / suppression_action_total
            )
            if suppression_action_total
            else 0.0,
            "repair_activation_rate": float(
                repair_pass_total / max(1, repair_pass_total + repair_reject_total)
            ),
            "repair_rejection_rate": float(
                repair_reject_total / max(1, repair_pass_total + repair_reject_total)
            ),
            "repair_to_suppression_ratio": float(write_1_total / max(1, write_0_total)),
            "expansion_total": int(expansion_total),
            "gap_closure_total": int(gap_closure_total),
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
            repair_precision=summary["repair_precision"],
            false_repair_rate=summary["false_repair_rate"],
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
    config_path = root / "config" / "tests" / "exp_06.yaml"
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
            "Experiment 06 balanced integrity",
        )
        make_line_plot(
            all_runs,
            figures_dir,
            "target_true_positive_rate",
            "target_true_positive_rate.png",
            "Experiment 06 target true positive rate",
        )
        make_line_plot(
            all_runs,
            figures_dir,
            "outside_false_positive_rate",
            "outside_false_positive_rate.png",
            "Experiment 06 outside false positive rate",
        )
        make_line_plot(
            all_runs,
            figures_dir,
            "repair_pass_total",
            "repair_pass_total.png",
            "Experiment 06 repair pass total",
        )
        make_line_plot(
            all_runs,
            figures_dir,
            "temporal_score_mean",
            "temporal_score_mean.png",
            "Experiment 06 temporal score mean",
        )
        make_line_plot(
            all_runs,
            figures_dir,
            "expansion_total",
            "expansion_total.png",
            "Experiment 06 expansion total",
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
            "figures/repair_pass_total.png",
            "figures/temporal_score_mean.png",
            "figures/expansion_total.png",
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
