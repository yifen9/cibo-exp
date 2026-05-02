from pathlib import Path
from collections import defaultdict, deque
from statistics import mean, variance
import math

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


def make_target(height: int, width: int, cfg: dict) -> np.ndarray:
    target = np.zeros((height, width), dtype=np.uint8)
    kind = str(cfg["type"])

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

    if kind == "maze":
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

    raise ValueError(f"unknown target type: {kind}")


def get_cell(canvas: np.ndarray, x: int, y: int) -> int:
    if x < 0 or x >= canvas.shape[0] or y < 0 or y >= canvas.shape[1]:
        return 0
    return int(canvas[x, y])


def local_window(canvas: np.ndarray, x: int, y: int, radius: int) -> np.ndarray:
    x0 = max(0, x - radius)
    x1 = min(canvas.shape[0], x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(canvas.shape[1], y + radius + 1)
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
            if 0 <= ni < w.shape[0] and 0 <= nj < w.shape[1]:
                if int(w[ni, nj]) == 1 and not bool(seen[ni, nj]):
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

    if left == 1 and right == 1:
        return 1.0

    if up == 1 and down == 1:
        return 1.0

    if (left or right) and (up or down):
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


def init_operators(
    count: int, height: int, width: int, rng: np.random.Generator
) -> np.ndarray:
    xs = rng.integers(0, height, size=count)
    ys = rng.integers(0, width, size=count)
    return np.stack([xs, ys], axis=1)


def move_operators(
    positions: np.ndarray, canvas: np.ndarray, topology: str, rng: np.random.Generator
) -> None:
    height, width = canvas.shape

    if topology == "random_local":
        delta = rng.integers(-2, 3, size=positions.shape)
    else:
        delta = rng.integers(-1, 2, size=positions.shape)

    if topology == "small_world":
        jump = rng.random(size=positions.shape[0]) < 0.05
        positions[jump, 0] = rng.integers(0, height, size=int(jump.sum()))
        positions[jump, 1] = rng.integers(0, width, size=int(jump.sum()))

    if topology == "component_local":
        for i in range(positions.shape[0]):
            x = int(positions[i, 0])
            y = int(positions[i, 1])
            candidates = []
            for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
                nx = min(max(0, x + dx), height - 1)
                ny = min(max(0, y + dy), width - 1)
                if int(canvas[nx, ny]) == 1:
                    candidates.append((nx, ny))
            if candidates and rng.random() < 0.7:
                nx, ny = candidates[rng.integers(0, len(candidates))]
                positions[i, 0] = nx
                positions[i, 1] = ny
            else:
                positions[i] += delta[i]
    else:
        positions += delta

    positions[:, 0] = np.clip(positions[:, 0], 0, height - 1)
    positions[:, 1] = np.clip(positions[:, 1], 0, width - 1)


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


def apply_perturbation(
    canvas: np.ndarray,
    target: np.ndarray,
    cfg: dict,
    condition: dict,
    seed: int,
    t: int,
) -> int:
    kind = str(condition["perturbation"])

    if kind == "random_target_deletion" and t == int(
        cfg["perturbation"]["random_delete_time"]
    ):
        return random_target_delete(
            canvas,
            target,
            float(cfg["perturbation"]["random_delete_fraction"]),
            int(seed * 1009 + 1),
        )

    if kind == "drifting_damage":
        times = [int(x) for x in cfg["perturbation"]["drift_times"]]
        centers = list(cfg["perturbation"]["drift_centers"])
        if t in times:
            i = times.index(t)
            return block_target_occlusion(
                canvas,
                target,
                list(centers[i]),
                int(cfg["perturbation"]["drift_block_size"]),
            )

    return 0


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


def update_sliding(
    history: np.ndarray,
    evidence: float,
    theta: float,
    x: int,
    y: int,
    t: int,
    k: int,
    m: int,
) -> tuple[bool, float]:
    slot = t % k
    history[x, y, slot] = 1 if evidence >= theta else 0
    score = int(history[x, y, :k].sum())
    return score >= m, float(score / k)


def choose_action(
    observed: np.ndarray,
    trace: np.ndarray,
    temporal_pass: bool,
    x: int,
    y: int,
    radius: int,
    theta_preserve: float,
    suppression: float,
    rng: np.random.Generator,
) -> dict:
    center = int(observed[x, y])
    support = structural_support(observed, x, y, radius)
    evidence = repair_evidence(observed, trace, x, y, radius)
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


def observed_position(
    position: np.ndarray, canvas: np.ndarray, topology: str, rng: np.random.Generator
) -> tuple[int, int]:
    x = int(position[0])
    y = int(position[1])

    if topology == "random_global":
        return int(rng.integers(0, canvas.shape[0])), int(
            rng.integers(0, canvas.shape[1])
        )

    return x, y


def active_operator_indices(
    count: int, mode: str, async_fraction: float, rng: np.random.Generator
) -> np.ndarray:
    if mode == "random_async":
        n = max(1, int(round(count * async_fraction)))
        return rng.choice(count, size=n, replace=False)

    return np.arange(count)


def summarize_metrics(rows: list[list], header: list[str], cfg: dict) -> dict:
    idx = {name: i for i, name in enumerate(header)}
    mismatch_values = [float(row[idx["mismatch_rate"]]) for row in rows]
    balanced_values = [float(row[idx["balanced_integrity"]]) for row in rows]
    outside_values = [float(row[idx["outside_false_positive_rate"]]) for row in rows]
    target_values = [float(row[idx["target_true_positive_rate"]]) for row in rows]

    saturation_time = None
    for row in rows:
        if float(row[idx["outside_false_positive_rate"]]) >= float(
            cfg["metrics"]["saturation_threshold"]
        ):
            saturation_time = int(row[idx["t"]])
            break

    target_damage_time = None
    for row in rows:
        if float(row[idx["target_true_positive_rate"]]) <= float(
            cfg["metrics"]["target_damage_threshold"]
        ):
            target_damage_time = int(row[idx["t"]])
            break

    recovery_time = None
    for row in rows:
        t = int(row[idx["t"]])
        if t >= int(cfg["perturbation"]["random_delete_time"]) and float(
            row[idx["balanced_integrity"]]
        ) >= float(cfg["metrics"]["recovery_threshold"]):
            recovery_time = t - int(cfg["perturbation"]["random_delete_time"])
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
    condition_dir.mkdir(parents=True, exist_ok=True)

    target = make_target(height, width, cfg["target"])
    canvas = target.copy()
    trace = np.zeros((height, width), dtype=np.float64)

    radius = int(condition["radius"])
    density = float(condition["density"])
    operator_count = max(1, int(round(height * width * density)))
    operators = init_operators(operator_count, height, width, rng)

    k = int(cfg["mechanism"]["k"])
    m = int(cfg["mechanism"]["m"])
    theta_repair = float(cfg["mechanism"]["theta_repair"])
    theta_preserve = float(cfg["mechanism"]["theta_preserve"])
    suppression = float(cfg["mechanism"]["suppression"])
    history = np.zeros((height, width, k), dtype=np.uint8)

    delay = int(condition["feedback_delay"])
    topology = str(condition["topology"])
    update_mode = str(condition["update_mode"])
    async_fraction = float(
        condition.get("async_fraction", cfg["default"]["async_fraction"])
    )

    canvas_history = [canvas.copy() for _ in range(delay + 1)]

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
        "coverage_rate",
        "action_redundancy",
        "conflict_rate",
        "evidence_stability",
        "positive_trace_mass",
        "negative_trace_mass",
        "perturbed",
    ]

    for t in range(steps + 1):
        perturbed = apply_perturbation(canvas, target, cfg, condition, seed, t)

        proposals = defaultdict(list)
        proposal_meta = defaultdict(list)
        observed_cells = set()
        evidence_high_total = 0
        candidate_total = 0
        repair_pass_total = 0
        repair_reject_total = 0

        if t < steps:
            move_operators(operators, canvas, topology, rng)
            observed = canvas_history[0] if delay > 0 else canvas.copy()
            indices = active_operator_indices(
                operator_count, update_mode, async_fraction, rng
            )

            if update_mode == "sequential_async":
                before = canvas.copy()

                for i in indices:
                    ax = int(operators[i, 0])
                    ay = int(operators[i, 1])
                    ox, oy = observed_position(operators[i], observed, topology, rng)
                    ox = min(max(0, ox), height - 1)
                    oy = min(max(0, oy), width - 1)
                    evidence = repair_evidence(observed, trace, ox, oy, radius)
                    temporal_pass, temporal_score = update_sliding(
                        history, evidence, theta_repair, ax, ay, t, k, m
                    )
                    result = choose_action(
                        observed,
                        trace,
                        temporal_pass,
                        ox,
                        oy,
                        radius,
                        theta_preserve,
                        suppression,
                        rng,
                    )
                    observed_cells.add((ax, ay))

                    if evidence >= theta_repair:
                        evidence_high_total += 1
                    candidate_total += 1

                    if int(canvas[ax, ay]) == 0:
                        if bool(result["repair_pass"]):
                            repair_pass_total += 1
                        else:
                            repair_reject_total += 1

                    if result["action"] != "stay":
                        if int(result["delta"]) > 0:
                            canvas[ax, ay] = 1
                        elif int(result["delta"]) < 0:
                            canvas[ax, ay] = 0
                        proposals[(ax, ay)].append(int(result["delta"]))
                        proposal_meta[(ax, ay)].append(result)

                action_stats = {
                    "applied_actions": int(sum(len(v) for v in proposals.values())),
                    "write_0_total": int(
                        sum(1 for values in proposals.values() for v in values if v < 0)
                    ),
                    "write_1_total": int(
                        sum(1 for values in proposals.values() for v in values if v > 0)
                    ),
                    "conflict_attempts": 0,
                }
            else:
                before = canvas.copy()

                for i in indices:
                    ax = int(operators[i, 0])
                    ay = int(operators[i, 1])
                    ox, oy = observed_position(operators[i], observed, topology, rng)
                    ox = min(max(0, ox), height - 1)
                    oy = min(max(0, oy), width - 1)
                    evidence = repair_evidence(observed, trace, ox, oy, radius)
                    temporal_pass, temporal_score = update_sliding(
                        history, evidence, theta_repair, ax, ay, t, k, m
                    )
                    result = choose_action(
                        observed,
                        trace,
                        temporal_pass,
                        ox,
                        oy,
                        radius,
                        theta_preserve,
                        suppression,
                        rng,
                    )
                    observed_cells.add((ax, ay))

                    if evidence >= theta_repair:
                        evidence_high_total += 1
                    candidate_total += 1

                    if int(canvas[ax, ay]) == 0:
                        if bool(result["repair_pass"]):
                            repair_pass_total += 1
                        else:
                            repair_reject_total += 1

                    if result["action"] != "stay":
                        proposals[(ax, ay)].append(int(result["delta"]))
                        proposal_meta[(ax, ay)].append(result)

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
                int(cfg["mechanism"]["trace_duration"]),
                float(cfg["mechanism"]["trace_decay"]),
            )
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
        target_cells = set((int(x), int(y)) for x, y in np.argwhere(target == 1))
        covered_target = len(target_cells.intersection(observed_cells))
        coverage_rate = float(covered_target / max(1, len(target_cells)))
        proposal_count = int(sum(len(v) for v in proposals.values()))
        touched_cells = max(1, len(proposals))
        action_redundancy = (
            float(proposal_count / touched_cells) if proposal_count else 0.0
        )
        conflict_rate = float(
            action_stats["conflict_attempts"] / max(1, proposal_count)
        )
        evidence_stability = float(evidence_high_total / max(1, candidate_total))

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
                int(proposal_count),
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
                float(coverage_rate),
                float(action_redundancy),
                float(conflict_rate),
                float(evidence_stability),
                float(np.maximum(trace, 0.0).sum()),
                float(np.maximum(-trace, 0.0).sum()),
                int(perturbed > 0),
            ]
        )

        canvas_history.append(canvas.copy())
        if len(canvas_history) > delay + 1:
            canvas_history.pop(0)

        progress.step(1)

    summary = summarize_metrics(rows, header, cfg)
    idx = {name: i for i, name in enumerate(header)}

    proposal_total = sum(int(row[idx["proposal_total"]]) for row in rows)
    applied_total = sum(int(row[idx["applied_actions"]]) for row in rows)
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
            "variable": str(condition["variable"]),
            "seed": int(seed),
            "radius": int(radius),
            "density": float(density),
            "operator_count": int(operator_count),
            "update_mode": update_mode,
            "feedback_delay": int(delay),
            "topology": topology,
            "perturbation": str(condition["perturbation"]),
            "condition_dir": str(condition_dir.relative_to(run_dir)),
            "metrics_path": str((condition_dir / "metrics.csv").relative_to(run_dir)),
            "proposal_total": int(proposal_total),
            "applied_total": int(applied_total),
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
            "expansion_total": int(expansion_total),
            "gap_closure_total": int(gap_closure_total),
            "mean_coverage_rate": float(
                mean([float(row[idx["coverage_rate"]]) for row in rows])
            ),
            "mean_action_redundancy": float(
                mean([float(row[idx["action_redundancy"]]) for row in rows])
            ),
            "mean_conflict_rate": float(
                mean([float(row[idx["conflict_rate"]]) for row in rows])
            ),
            "mean_evidence_stability": float(
                mean([float(row[idx["evidence_stability"]]) for row in rows])
            ),
        }
    )

    write_csv(str(condition_dir / "metrics.csv"), rows, header=header)
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


def safe_var(xs: list[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    return float(variance(xs))


def failure_mode(summary: dict, cfg: dict) -> str:
    tpr = float(summary["final_target_true_positive_rate"])
    fpr = float(summary["final_outside_false_positive_rate"])

    if is_strong_success(summary, cfg):
        return "success"

    if tpr <= float(cfg["metrics"]["strong_tpr_threshold"]) and fpr < float(
        cfg["metrics"]["strong_fpr_threshold"]
    ):
        return "target_under_repair"

    if tpr > float(cfg["metrics"]["strong_tpr_threshold"]) and fpr >= float(
        cfg["metrics"]["strong_fpr_threshold"]
    ):
        return "outside_saturation"

    return "mixed_failure"


def aggregate_by_condition(summaries: list[dict], cfg: dict) -> list[dict]:
    groups = {}

    for s in summaries:
        groups.setdefault(str(s["condition_id"]), []).append(s)

    rows = []

    for condition_id, items in groups.items():
        balanced = [float(x["final_balanced_integrity"]) for x in items]
        tpr = [float(x["final_target_true_positive_rate"]) for x in items]
        fpr = [float(x["final_outside_false_positive_rate"]) for x in items]
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
                "variable": str(first["variable"]),
                "radius": int(first["radius"]),
                "density": float(first["density"]),
                "operator_count": int(first["operator_count"]),
                "update_mode": str(first["update_mode"]),
                "feedback_delay": int(first["feedback_delay"]),
                "topology": str(first["topology"]),
                "perturbation": str(first["perturbation"]),
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
                "mean_repair_precision": float(
                    mean([float(x["repair_precision"]) for x in items])
                ),
                "mean_false_repair_rate": float(
                    mean([float(x["false_repair_rate"]) for x in items])
                ),
                "mean_coverage_rate": float(
                    mean([float(x["mean_coverage_rate"]) for x in items])
                ),
                "mean_action_redundancy": float(
                    mean([float(x["mean_action_redundancy"]) for x in items])
                ),
                "mean_conflict_rate": float(
                    mean([float(x["mean_conflict_rate"]) for x in items])
                ),
                "mean_evidence_stability": float(
                    mean([float(x["mean_evidence_stability"]) for x in items])
                ),
                "mean_expansion_total": float(
                    mean([float(x["expansion_total"]) for x in items])
                ),
                "mean_gap_closure_total": float(
                    mean([float(x["gap_closure_total"]) for x in items])
                ),
            }
        )

    rows.sort(key=lambda x: (str(x["group"]), str(x["condition_id"])))
    return rows


def make_plots(rows: list[dict], figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [str(x["condition_id"]) for x in rows]
    balanced = [float(x["mean_final_balanced_integrity"]) for x in rows]
    success = [float(x["strong_success_rate"]) for x in rows]

    fig = plt.figure(figsize=(14, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(labels, balanced)
    ax.set_title("Experiment 10 mean final balanced integrity")
    ax.set_xlabel("condition")
    ax.set_ylabel("mean final balanced integrity")
    ax.tick_params(axis="x", rotation=75)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(str(figures_dir / "mean_final_balanced_integrity.png"), dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(14, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(labels, success)
    ax.set_title("Experiment 10 strong success rate")
    ax.set_xlabel("condition")
    ax.set_ylabel("strong success rate")
    ax.tick_params(axis="x", rotation=75)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(str(figures_dir / "strong_success_rate.png"), dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(1, 1, 1)

    for item in rows:
        ax.scatter(float(item["mean_final_fpr"]), float(item["mean_final_tpr"]))
        ax.text(
            float(item["mean_final_fpr"]),
            float(item["mean_final_tpr"]),
            str(item["condition_id"]),
            fontsize=6,
        )

    ax.set_title("Experiment 10 TPR-FPR plane")
    ax.set_xlabel("mean final outside FPR")
    ax.set_ylabel("mean final TPR")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(str(figures_dir / "tpr_fpr_plane.png"), dpi=160)
    plt.close(fig)


def main() -> int:
    root = repo_root()
    config_path = root / "config" / "tests" / "exp_10.yaml"
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

    summaries = [item["summary"] for item in all_runs]
    condition_rows = aggregate_by_condition(summaries, cfg)

    run_header = [
        "condition_id",
        "group",
        "variable",
        "seed",
        "radius",
        "density",
        "operator_count",
        "update_mode",
        "feedback_delay",
        "topology",
        "perturbation",
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
        "expansion_total",
        "gap_closure_total",
        "mean_coverage_rate",
        "mean_action_redundancy",
        "mean_conflict_rate",
        "mean_evidence_stability",
        "metrics_path",
    ]

    condition_header = [
        "condition_id",
        "group",
        "variable",
        "radius",
        "density",
        "operator_count",
        "update_mode",
        "feedback_delay",
        "topology",
        "perturbation",
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
        "mean_repair_precision",
        "mean_false_repair_rate",
        "mean_coverage_rate",
        "mean_action_redundancy",
        "mean_conflict_rate",
        "mean_evidence_stability",
        "mean_expansion_total",
        "mean_gap_closure_total",
    ]

    write_csv(
        str(run_dir / "runs_summary.csv"),
        [[s.get(k) for k in run_header] for s in summaries],
        header=run_header,
    )
    write_json(str(run_dir / "runs_summary.json"), summaries)
    write_csv(
        str(run_dir / "structure_summary.csv"),
        [[s.get(k) for k in condition_header] for s in condition_rows],
        header=condition_header,
    )
    write_json(str(run_dir / "structure_summary.json"), condition_rows)

    if bool(cfg["output"]["make_plot"]):
        make_plots(condition_rows, figures_dir)

    default_rows = [
        x
        for x in condition_rows
        if str(x["condition_id"]) in {"A_R1", "D_DELAY0", "E_GRID"}
    ]
    random_global_rows = [
        x for x in condition_rows if str(x["condition_id"]) == "E_RANDOM_GLOBAL"
    ]
    delay0 = [x for x in condition_rows if str(x["condition_id"]) == "D_DELAY0"]
    delay8 = [x for x in condition_rows if str(x["condition_id"]) == "D_DELAY8"]

    topology_sensitivity = None
    if default_rows and random_global_rows:
        topology_sensitivity = float(
            default_rows[0]["mean_final_balanced_integrity"]
        ) - float(random_global_rows[0]["mean_final_balanced_integrity"])

    delay_sensitivity = None
    if delay0 and delay8:
        delay_sensitivity = float(delay0[0]["mean_final_balanced_integrity"]) - float(
            delay8[0]["mean_final_balanced_integrity"]
        )

    run_summary = {
        "name": cfg["name"],
        "run_dir": str(run_dir),
        "fingerprint": meta.get("fingerprint"),
        "condition_count": len(conditions),
        "seed_count": len(seeds),
        "run_count": len(summaries),
        "topology_sensitivity": topology_sensitivity,
        "delay_sensitivity": delay_sensitivity,
        "runs_summary_path": "runs_summary.csv",
        "runs_summary_json_path": "runs_summary.json",
        "structure_summary_path": "structure_summary.csv",
        "structure_summary_json_path": "structure_summary.json",
        "figures": [
            "figures/mean_final_balanced_integrity.png",
            "figures/strong_success_rate.png",
            "figures/tpr_fpr_plane.png",
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
