from pathlib import Path
from collections import defaultdict, deque
from statistics import mean, variance
import importlib.util

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


def load_exp11_module(root: Path):
    path = root / "scripts" / "tests" / "exp_11.py"
    spec = importlib.util.spec_from_file_location(
        "cibo_exp_11_runtime_for_exp_12", str(path)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def thicken(mask: np.ndarray, radius: int) -> np.ndarray:
    out = mask.copy()

    for x, y in np.argwhere(mask == 1):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx = int(x) + dx
                ny = int(y) + dy
                if 0 <= nx < mask.shape[0] and 0 <= ny < mask.shape[1]:
                    out[nx, ny] = 1

    return out


def bridge_center_cells(reference: np.ndarray, count: int) -> list[tuple[int, int]]:
    cells = np.argwhere(reference == 1)
    center = np.array([reference.shape[0] // 2, reference.shape[1] // 2])
    distances = np.sum((cells - center) ** 2, axis=1)
    order = np.argsort(distances)
    selected = cells[order[: min(count, len(cells))]]
    return [(int(x), int(y)) for x, y in selected]


def make_marker(
    reference: np.ndarray, cells: list[tuple[int, int]], radius: int, strength: float
) -> np.ndarray:
    marker = np.zeros_like(reference, dtype=np.float64)

    for x, y in cells:
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx = x + dx
                ny = y + dy
                if 0 <= nx < reference.shape[0] and 0 <= ny < reference.shape[1]:
                    d = abs(dx) + abs(dy)
                    marker[nx, ny] = max(marker[nx, ny], strength / max(1.0, d + 1.0))

    return marker


def make_direction(reference: np.ndarray) -> np.ndarray:
    direction = np.zeros((*reference.shape, 2), dtype=np.float64)

    for x, y in np.argwhere(reference == 1):
        x = int(x)
        y = int(y)
        vx = 0.0
        vy = 0.0

        if get_cell(reference, x - 1, y) == 1:
            vx -= 1.0
        if get_cell(reference, x + 1, y) == 1:
            vx += 1.0
        if get_cell(reference, x, y - 1) == 1:
            vy -= 1.0
        if get_cell(reference, x, y + 1) == 1:
            vy += 1.0

        n = max(1e-9, (vx * vx + vy * vy) ** 0.5)
        direction[x, y, 0] = vx / n
        direction[x, y, 1] = vy / n

    return direction


def apply_scaffold(exp11, cfg: dict, condition: dict):
    height = int(cfg["canvas"]["height"])
    width = int(cfg["canvas"]["width"])
    reference, start, goal = exp11.make_path_environment(
        height, width, str(condition["path_type"])
    )
    scaffold = str(condition["scaffold"])

    marker_cells = bridge_center_cells(reference, int(condition.get("cut_length", 5)))
    marker = np.zeros_like(reference, dtype=np.float64)

    if scaffold in {"width", "combined"}:
        reference = thicken(reference, int(condition.get("width_radius", 1)))

    if scaffold == "aligned_redundancy":
        reference, start, goal = exp11.make_path_environment(
            height, width, "double_path"
        )

    if scaffold in {"bridge_marker", "combined"}:
        marker = make_marker(
            reference,
            marker_cells,
            int(condition.get("marker_radius", 2)),
            float(condition.get("marker_strength", 1.0)),
        )

    if scaffold == "directional_trace":
        direction = make_direction(reference)
    elif scaffold == "combined":
        direction = make_direction(reference)
    else:
        direction = np.zeros((*reference.shape, 2), dtype=np.float64)

    return reference, start, goal, marker, direction


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


def local_density(canvas: np.ndarray, x: int, y: int, radius: int) -> float:
    w = local_window(canvas, x, y, radius)
    center = int(canvas[x, y])
    return float((int(w.sum()) - center) / max(1, w.size - 1))


def bridge_evidence(canvas: np.ndarray, x: int, y: int) -> float:
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


def directional_consistency(
    canvas: np.ndarray, direction: np.ndarray, x: int, y: int
) -> float:
    vx = float(direction[x, y, 0])
    vy = float(direction[x, y, 1])

    if abs(vx) + abs(vy) <= 1e-9:
        return 0.0

    score = 0.0

    if vx < 0 and get_cell(canvas, x - 1, y) == 1:
        score += abs(vx)
    if vx > 0 and get_cell(canvas, x + 1, y) == 1:
        score += abs(vx)
    if vy < 0 and get_cell(canvas, x, y - 1) == 1:
        score += abs(vy)
    if vy > 0 and get_cell(canvas, x, y + 1) == 1:
        score += abs(vy)

    return float(min(1.0, score))


def cost_penalty(canvas: np.ndarray, x: int, y: int, radius: int) -> float:
    old = int(canvas[x, y])
    canvas[x, y] = 1

    neighbors = (
        get_cell(canvas, x - 1, y)
        + get_cell(canvas, x + 1, y)
        + get_cell(canvas, x, y - 1)
        + get_cell(canvas, x, y + 1)
    )

    density = local_density(canvas, x, y, radius)
    canvas[x, y] = old

    isolated = 1.0 if neighbors <= 1 else 0.0
    bulky = max(0.0, density - 0.50)
    return float(min(1.0, 0.70 * isolated + 0.30 * bulky))


def scaffolded_support(
    canvas: np.ndarray,
    trace: np.ndarray,
    marker: np.ndarray,
    direction: np.ndarray,
    x: int,
    y: int,
    cfg: dict,
    condition: dict,
) -> float:
    radius = int(cfg["operator"]["radius"])
    repair = cfg["repair"]
    scaffold = str(condition["scaffold"])

    old = int(canvas[x, y])
    if old == 0:
        canvas[x, y] = 1

    local = local_density(canvas, x, y, radius)
    bridge = bridge_evidence(canvas, x, y)
    direct = directional_consistency(canvas, direction, x, y)
    mark = float(marker[x, y])
    cost = cost_penalty(canvas, x, y, radius)

    canvas[x, y] = old

    x0 = max(0, x - radius)
    x1 = min(canvas.shape[0], x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(canvas.shape[1], y + radius + 1)
    tw = trace[x0:x1, y0:y1]
    positive = float(np.maximum(tw, 0.0).sum())
    negative = float(np.maximum(-tw, 0.0).sum())
    trace_score = 1.0 if positive > negative and positive > 0 else 0.0

    score = (
        float(repair["w_local"]) * local
        + float(repair["w_bridge"]) * bridge
        + 0.10 * trace_score
    )

    if scaffold in {"bridge_marker", "combined"}:
        score += float(repair["w_marker"]) * mark

    if scaffold in {"directional_trace", "combined"}:
        score += float(repair["w_direction"]) * direct

    if scaffold in {"cost_sensitive", "combined"}:
        score -= float(repair["w_cost"]) * cost

    return float(max(0.0, min(1.0, score)))


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


def random_path_delete(
    canvas: np.ndarray, reference: np.ndarray, fraction: float, seed: int
) -> int:
    rng = np.random.default_rng(seed)
    cells = np.argwhere(reference == 1)

    if len(cells) == 0:
        return 0

    count = max(1, int(round(len(cells) * fraction)))
    selected = cells[rng.choice(len(cells), size=min(count, len(cells)), replace=False)]

    for x, y in selected:
        canvas[int(x), int(y)] = 0

    return int(len(selected))


def bridge_cut(canvas: np.ndarray, reference: np.ndarray, condition: dict) -> int:
    cells = bridge_center_cells(reference, int(condition.get("cut_length", 5)))
    changed = 0

    for x, y in cells:
        if int(canvas[x, y]) == 1:
            changed += 1
        canvas[x, y] = 0

    return int(changed)


def outside_opening(
    canvas: np.ndarray, reference: np.ndarray, fraction: float, seed: int
) -> int:
    rng = np.random.default_rng(seed)
    cells = np.argwhere(reference == 0)

    if len(cells) == 0:
        return 0

    count = max(1, int(round(len(cells) * fraction)))
    selected = cells[rng.choice(len(cells), size=min(count, len(cells)), replace=False)]

    for x, y in selected:
        canvas[int(x), int(y)] = 1

    return int(len(selected))


def drifting_blockage(
    canvas: np.ndarray, reference: np.ndarray, center: list[int], size: int
) -> int:
    cx = int(center[0])
    cy = int(center[1])
    r = size // 2
    changed = 0

    for x in range(max(0, cx - r), min(canvas.shape[0], cx + r + 1)):
        for y in range(max(0, cy - r), min(canvas.shape[1], cy + r + 1)):
            if int(reference[x, y]) == 1:
                if int(canvas[x, y]) == 1:
                    changed += 1
                canvas[x, y] = 0

    return int(changed)


def apply_perturbation(
    canvas: np.ndarray, reference: np.ndarray, condition: dict, seed: int, t: int
) -> int:
    kind = str(condition["perturbation_type"])

    if kind == "random_path_deletion":
        if t == int(condition["damage_time"]):
            return random_path_delete(
                canvas,
                reference,
                float(condition["delete_fraction"]),
                int(seed * 1009 + 1),
            )
        return 0

    if kind == "bridge_cut":
        if t == int(condition["damage_time"]):
            return bridge_cut(canvas, reference, condition)
        return 0

    if kind == "mixed_path_damage_outside_opening":
        if t == int(condition["damage_time"]):
            a = random_path_delete(
                canvas,
                reference,
                float(condition["delete_fraction"]),
                int(seed * 1009 + 2),
            )
            b = outside_opening(
                canvas,
                reference,
                float(condition["outside_fraction"]),
                int(seed * 1009 + 3),
            )
            return int(a + b)
        return 0

    if kind == "drifting_blockage":
        times = [int(x) for x in condition["drifting_times"]]
        if t in times:
            i = times.index(t)
            return drifting_blockage(
                canvas,
                reference,
                list(condition["drift_centers"][i]),
                int(condition["block_size"]),
            )
        return 0

    raise ValueError(f"unknown perturbation type: {kind}")


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
    canvas: np.ndarray,
    support: float,
    temporal_pass: bool,
    x: int,
    y: int,
    cfg: dict,
    rng: np.random.Generator,
) -> dict:
    center = int(canvas[x, y])

    if center == 1:
        if support < float(cfg["operator"]["theta_preserve"]) and rng.random() < float(
            cfg["operator"]["suppression"]
        ):
            return {"action": "close", "delta": -1, "repair_pass": False}

        return {"action": "stay", "delta": 0, "repair_pass": False}

    if temporal_pass:
        return {"action": "open", "delta": 1, "repair_pass": True}

    return {"action": "stay", "delta": 0, "repair_pass": False}


def resolve_actions(canvas: np.ndarray, proposals: dict) -> dict:
    stats = {
        "applied_actions": 0,
        "open_total": 0,
        "close_total": 0,
        "conflict_attempts": 0,
    }

    for cell, values in proposals.items():
        open_votes = sum(1 for v in values if v > 0)
        close_votes = sum(1 for v in values if v < 0)

        if open_votes > 0 and close_votes > 0:
            stats["conflict_attempts"] += min(open_votes, close_votes)

        if open_votes > close_votes:
            x, y = cell
            if int(canvas[x, y]) != 1:
                stats["applied_actions"] += 1
            canvas[x, y] = 1
            stats["open_total"] += 1
        elif close_votes > open_votes:
            x, y = cell
            if int(canvas[x, y]) != 0:
                stats["applied_actions"] += 1
            canvas[x, y] = 0
            stats["close_total"] += 1

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


def classify_open(before: np.ndarray, reference: np.ndarray, x: int, y: int) -> str:
    if int(before[x, y]) == 0 and int(reference[x, y]) == 1:
        return "true_open"
    if int(before[x, y]) == 0 and int(reference[x, y]) == 0:
        return "false_open"
    return "non_open"


def classify_close(before: np.ndarray, reference: np.ndarray, x: int, y: int) -> str:
    if int(before[x, y]) == 1 and int(reference[x, y]) == 0:
        return "true_close"
    if int(before[x, y]) == 1 and int(reference[x, y]) == 1:
        return "false_close"
    return "non_close"


def scaffold_utilization(
    x: int, y: int, marker: np.ndarray, direction: np.ndarray, condition: dict
) -> int:
    scaffold = str(condition["scaffold"])

    if scaffold in {"bridge_marker", "combined"} and float(marker[x, y]) > 0:
        return 1

    if (
        scaffold in {"directional_trace", "combined"}
        and abs(float(direction[x, y, 0])) + abs(float(direction[x, y, 1])) > 0
    ):
        return 1

    return 0


def compute_metrics(
    exp11,
    canvas: np.ndarray,
    reference: np.ndarray,
    start: list[tuple[int, int]],
    goal: list[tuple[int, int]],
    initial_open: int,
) -> dict:
    conn = exp11.connectivity(canvas, start, goal)
    sp = exp11.shortest_path_length(canvas, start, goal)
    open_cost = int(canvas.sum())
    outside_open = int(((canvas == 1) & (reference == 0)).sum())
    outside_total = int((reference == 0).sum())
    path_tpr = float(
        ((canvas == 1) & (reference == 1)).sum() / max(1, int(reference.sum()))
    )
    outside_open_rate = float(outside_open / max(1, outside_total))
    open_cost_factor = float(open_cost / max(1, initial_open))

    return {
        "connectivity": int(conn),
        "shortest_path_length": sp,
        "open_cost": int(open_cost),
        "open_cost_factor": open_cost_factor,
        "outside_open_rate": outside_open_rate,
        "path_tpr": path_tpr,
        "false_corridor_count": int(exp11.false_corridor_count(canvas, start, goal)),
    }


def summarize(
    rows: list[list],
    header: list[str],
    condition: dict,
    initial_connected: int,
    initial_open: int,
) -> dict:
    idx = {name: i for i, name in enumerate(header)}
    final = rows[-1]
    values = [int(row[idx["connectivity"]]) for row in rows]
    damage_time = int(
        condition.get("damage_time", condition.get("drifting_times", [0])[0])
    )
    post = [row for row in rows if int(row[idx["t"]]) >= damage_time]

    recovery_time = None

    for row in post:
        if int(row[idx["connectivity"]]) == 1:
            recovery_time = int(row[idx["t"]]) - damage_time
            break

    return {
        "initial_connected": int(initial_connected),
        "initial_open_cost": int(initial_open),
        "final_connectivity": int(final[idx["connectivity"]]),
        "connectivity_stability": float(sum(values) / max(1, len(values))),
        "post_damage_connectivity_stability": float(
            sum(int(row[idx["connectivity"]]) for row in post) / max(1, len(post))
        ),
        "recovery_time": recovery_time,
        "final_open_cost": int(final[idx["open_cost"]]),
        "final_open_cost_factor": float(final[idx["open_cost_factor"]]),
        "final_outside_open_rate": float(final[idx["outside_open_rate"]]),
        "final_path_tpr": float(final[idx["path_tpr"]]),
        "final_shortest_path_length": final[idx["shortest_path_length"]],
        "final_false_corridor_count": int(final[idx["false_corridor_count"]]),
        "mean_open_cost_factor": float(
            mean([float(row[idx["open_cost_factor"]]) for row in rows])
        ),
        "mean_outside_open_rate": float(
            mean([float(row[idx["outside_open_rate"]]) for row in rows])
        ),
    }


def run_condition(
    exp11,
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

    reference, start, goal, marker, direction = apply_scaffold(exp11, cfg, condition)
    canvas = reference.copy()
    trace = np.zeros((height, width), dtype=np.float64)
    history = np.zeros((height, width, int(cfg["repair"]["k"])), dtype=np.uint8)
    operators = init_operators(int(cfg["operator"]["count"]), height, width, rng)

    initial_connected = exp11.connectivity(canvas, start, goal)
    initial_open = int(canvas.sum())

    rows = []
    header = [
        "t",
        "connectivity",
        "shortest_path_length",
        "open_cost",
        "open_cost_factor",
        "outside_open_rate",
        "path_tpr",
        "false_corridor_count",
        "actions_total",
        "proposal_total",
        "applied_actions",
        "open_total",
        "close_total",
        "true_open_total",
        "false_open_total",
        "true_close_total",
        "false_close_total",
        "repair_pass_total",
        "repair_reject_total",
        "scaffold_utilized_total",
        "conflict_attempts",
        "support_mean",
        "positive_trace_mass",
        "negative_trace_mass",
        "perturbed",
    ]

    for t in range(steps + 1):
        perturbed = apply_perturbation(canvas, reference, condition, seed, t)
        proposals = defaultdict(list)
        proposal_meta = defaultdict(list)
        support_values = []
        repair_pass_total = 0
        repair_reject_total = 0
        scaffold_utilized_total = 0

        if t < steps:
            move_operators(operators, height, width, rng)

            for pos in operators:
                x = int(pos[0])
                y = int(pos[1])
                support = scaffolded_support(
                    canvas, trace, marker, direction, x, y, cfg, condition
                )
                temporal_pass, temporal_score = update_sliding(
                    history,
                    support,
                    float(cfg["repair"]["theta_repair"]),
                    x,
                    y,
                    t,
                    int(cfg["repair"]["k"]),
                    int(cfg["repair"]["m"]),
                )
                result = choose_action(canvas, support, temporal_pass, x, y, cfg, rng)
                support_values.append(float(support))

                if int(canvas[x, y]) == 0:
                    if bool(result["repair_pass"]):
                        repair_pass_total += 1
                    else:
                        repair_reject_total += 1

                if result["action"] != "stay":
                    proposals[(x, y)].append(int(result["delta"]))
                    proposal_meta[(x, y)].append(result)

            before = canvas.copy()
            action_stats = resolve_actions(canvas, proposals)

            true_open_total = 0
            false_open_total = 0
            true_close_total = 0
            false_close_total = 0

            for cell, metas in proposal_meta.items():
                x, y = cell
                scaffold_utilized_total += scaffold_utilization(
                    x, y, marker, direction, condition
                )

                for item in metas:
                    if int(item["delta"]) > 0:
                        label = classify_open(before, reference, x, y)
                        if label == "true_open":
                            true_open_total += 1
                        elif label == "false_open":
                            false_open_total += 1
                    elif int(item["delta"]) < 0:
                        label = classify_close(before, reference, x, y)
                        if label == "true_close":
                            true_close_total += 1
                        elif label == "false_close":
                            false_close_total += 1

            update_trace(
                trace,
                proposals,
                int(cfg["trace"]["duration"]),
                float(cfg["trace"]["decay"]),
            )
        else:
            action_stats = {
                "applied_actions": 0,
                "open_total": 0,
                "close_total": 0,
                "conflict_attempts": 0,
            }
            true_open_total = 0
            false_open_total = 0
            true_close_total = 0
            false_close_total = 0

        metrics = compute_metrics(exp11, canvas, reference, start, goal, initial_open)

        rows.append(
            [
                int(t),
                int(metrics["connectivity"]),
                metrics["shortest_path_length"],
                int(metrics["open_cost"]),
                float(metrics["open_cost_factor"]),
                float(metrics["outside_open_rate"]),
                float(metrics["path_tpr"]),
                int(metrics["false_corridor_count"]),
                int(cfg["operator"]["count"]),
                int(sum(len(v) for v in proposals.values())),
                int(action_stats["applied_actions"]),
                int(action_stats["open_total"]),
                int(action_stats["close_total"]),
                int(true_open_total),
                int(false_open_total),
                int(true_close_total),
                int(false_close_total),
                int(repair_pass_total),
                int(repair_reject_total),
                int(scaffold_utilized_total),
                int(action_stats["conflict_attempts"]),
                float(mean(support_values)) if support_values else 0.0,
                float(np.maximum(trace, 0.0).sum()),
                float(np.maximum(-trace, 0.0).sum()),
                int(perturbed > 0),
            ]
        )

        progress.step(1)

    summary = summarize(rows, header, condition, initial_connected, initial_open)
    idx = {name: i for i, name in enumerate(header)}

    true_open = sum(int(row[idx["true_open_total"]]) for row in rows)
    false_open = sum(int(row[idx["false_open_total"]]) for row in rows)
    true_close = sum(int(row[idx["true_close_total"]]) for row in rows)
    false_close = sum(int(row[idx["false_close_total"]]) for row in rows)
    open_total = true_open + false_open
    close_total = true_close + false_close
    scaffold_total = sum(int(row[idx["scaffold_utilized_total"]]) for row in rows)
    proposal_total = sum(int(row[idx["proposal_total"]]) for row in rows)

    summary.update(
        {
            "condition_id": condition_id,
            "group": str(condition["group"]),
            "seed": int(seed),
            "path_class": str(condition["path_class"]),
            "path_type": str(condition["path_type"]),
            "perturbation_type": str(condition["perturbation_type"]),
            "scaffold": str(condition["scaffold"]),
            "condition_dir": str(condition_dir.relative_to(run_dir)),
            "metrics_path": str((condition_dir / "metrics.csv").relative_to(run_dir)),
            "true_open_total": int(true_open),
            "false_open_total": int(false_open),
            "true_close_total": int(true_close),
            "false_close_total": int(false_close),
            "open_precision": float(true_open / open_total) if open_total else 0.0,
            "false_open_rate": float(false_open / open_total) if open_total else 0.0,
            "close_precision": float(true_close / close_total) if close_total else 0.0,
            "false_close_rate": float(false_close / close_total)
            if close_total
            else 0.0,
            "proposal_total": int(proposal_total),
            "applied_total": int(sum(int(row[idx["applied_actions"]]) for row in rows)),
            "repair_pass_total": int(
                sum(int(row[idx["repair_pass_total"]]) for row in rows)
            ),
            "repair_reject_total": int(
                sum(int(row[idx["repair_reject_total"]]) for row in rows)
            ),
            "scaffold_utilization_rate": float(scaffold_total / max(1, proposal_total)),
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
            final_connectivity=summary["final_connectivity"],
            final_open_cost_factor=summary["final_open_cost_factor"],
            final_outside_open_rate=summary["final_outside_open_rate"],
        )
    )

    return {
        "condition_id": condition_id,
        "seed": int(seed),
        "header": header,
        "rows": rows,
        "summary": summary,
    }


def weak_success(summary: dict) -> bool:
    return int(summary["final_connectivity"]) == 1


def strong_success(summary: dict, cfg: dict) -> bool:
    return int(summary["final_connectivity"]) == 1 and float(
        summary["final_open_cost_factor"]
    ) <= float(cfg["metrics"]["strong_cost_factor"])


def strict_success(summary: dict, cfg: dict) -> bool:
    return (
        int(summary["final_connectivity"]) == 1
        and float(summary["final_open_cost_factor"])
        <= float(cfg["metrics"]["strict_cost_factor"])
        and float(summary["final_outside_open_rate"])
        <= float(cfg["metrics"]["strict_outside_open_rate"])
    )


def failure_mode(summary: dict, cfg: dict) -> str:
    if strong_success(summary, cfg):
        return "functional_success"

    if int(summary["final_connectivity"]) == 0:
        if str(summary["path_class"]) == "F6":
            return "irreducible_sparse_failure"
        return "connectivity_under_repair"

    if float(summary["final_open_cost_factor"]) > float(
        cfg["metrics"]["strong_cost_factor"]
    ):
        return "over_expansion"

    if int(summary["final_false_corridor_count"]) > 1:
        return "false_corridor_formation"

    return "mixed_failure"


def safe_var(xs: list[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    return float(variance(xs))


def aggregate_by_condition(summaries: list[dict], cfg: dict) -> list[dict]:
    groups = {}

    for s in summaries:
        groups.setdefault(str(s["condition_id"]), []).append(s)

    rows = []

    for condition_id, items in groups.items():
        weak = [1 if weak_success(x) else 0 for x in items]
        strong = [1 if strong_success(x, cfg) else 0 for x in items]
        strict = [1 if strict_success(x, cfg) else 0 for x in items]
        cost = [float(x["final_open_cost_factor"]) for x in items]
        outside = [float(x["final_outside_open_rate"]) for x in items]
        stability = [float(x["post_damage_connectivity_stability"]) for x in items]
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
                "path_class": str(first["path_class"]),
                "path_type": str(first["path_type"]),
                "perturbation_type": str(first["perturbation_type"]),
                "scaffold": str(first["scaffold"]),
                "seed_count": len(items),
                "weak_success_rate": float(sum(weak) / len(weak)),
                "strong_success_rate": float(sum(strong) / len(strong)),
                "strict_success_rate": float(sum(strict) / len(strict)),
                "dominant_failure_mode": dominant_mode,
                "mean_final_open_cost_factor": float(mean(cost)),
                "var_final_open_cost_factor": safe_var(cost),
                "worst_final_open_cost_factor": float(max(cost)),
                "mean_final_outside_open_rate": float(mean(outside)),
                "worst_final_outside_open_rate": float(max(outside)),
                "mean_post_damage_connectivity_stability": float(mean(stability)),
                "mean_final_path_tpr": float(
                    mean([float(x["final_path_tpr"]) for x in items])
                ),
                "mean_false_corridor_count": float(
                    mean([float(x["final_false_corridor_count"]) for x in items])
                ),
                "mean_open_precision": float(
                    mean([float(x["open_precision"]) for x in items])
                ),
                "mean_false_open_rate": float(
                    mean([float(x["false_open_rate"]) for x in items])
                ),
                "mean_scaffold_utilization_rate": float(
                    mean([float(x["scaffold_utilization_rate"]) for x in items])
                ),
                "mean_recovery_time": float(
                    mean(
                        [
                            float(x["recovery_time"])
                            for x in items
                            if x["recovery_time"] is not None
                        ]
                    )
                )
                if any(x["recovery_time"] is not None for x in items)
                else None,
            }
        )

    rows.sort(key=lambda x: str(x["group"]))
    return rows


def make_plots(rows: list[dict], figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [str(x["group"]) for x in rows]
    strong = [float(x["strong_success_rate"]) for x in rows]
    cost = [float(x["mean_final_open_cost_factor"]) for x in rows]
    util = [float(x["mean_scaffold_utilization_rate"]) for x in rows]

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(labels, strong)
    ax.set_title("Experiment 12 strong functional success rate")
    ax.set_xlabel("condition")
    ax.set_ylabel("strong success rate")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(str(figures_dir / "strong_functional_success_rate.png"), dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(labels, cost)
    ax.set_title("Experiment 12 mean final open cost factor")
    ax.set_xlabel("condition")
    ax.set_ylabel("open cost factor")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(str(figures_dir / "mean_open_cost_factor.png"), dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(labels, util)
    ax.set_title("Experiment 12 scaffold utilization rate")
    ax.set_xlabel("condition")
    ax.set_ylabel("scaffold utilization rate")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(str(figures_dir / "scaffold_utilization_rate.png"), dpi=160)
    plt.close(fig)


def main() -> int:
    root = repo_root()
    config_path = root / "config" / "tests" / "exp_12.yaml"
    cfg = read_yaml(str(config_path))
    exp11 = load_exp11_module(root)

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
                run_condition(exp11, cfg, condition, seed, run_dir, logger, progress)
            )

    progress.finish()

    summaries = [item["summary"] for item in all_runs]
    condition_rows = aggregate_by_condition(summaries, cfg)

    run_header = [
        "condition_id",
        "group",
        "seed",
        "path_class",
        "path_type",
        "perturbation_type",
        "scaffold",
        "initial_connected",
        "initial_open_cost",
        "final_connectivity",
        "connectivity_stability",
        "post_damage_connectivity_stability",
        "recovery_time",
        "final_open_cost",
        "final_open_cost_factor",
        "final_outside_open_rate",
        "final_path_tpr",
        "final_shortest_path_length",
        "final_false_corridor_count",
        "mean_open_cost_factor",
        "mean_outside_open_rate",
        "true_open_total",
        "false_open_total",
        "true_close_total",
        "false_close_total",
        "open_precision",
        "false_open_rate",
        "close_precision",
        "false_close_rate",
        "proposal_total",
        "applied_total",
        "repair_pass_total",
        "repair_reject_total",
        "scaffold_utilization_rate",
        "metrics_path",
    ]

    condition_header = [
        "condition_id",
        "group",
        "path_class",
        "path_type",
        "perturbation_type",
        "scaffold",
        "seed_count",
        "weak_success_rate",
        "strong_success_rate",
        "strict_success_rate",
        "dominant_failure_mode",
        "mean_final_open_cost_factor",
        "var_final_open_cost_factor",
        "worst_final_open_cost_factor",
        "mean_final_outside_open_rate",
        "worst_final_outside_open_rate",
        "mean_post_damage_connectivity_stability",
        "mean_final_path_tpr",
        "mean_false_corridor_count",
        "mean_open_precision",
        "mean_false_open_rate",
        "mean_scaffold_utilization_rate",
        "mean_recovery_time",
    ]

    write_csv(
        str(run_dir / "runs_summary.csv"),
        [[s.get(k) for k in run_header] for s in summaries],
        header=run_header,
    )
    write_json(str(run_dir / "runs_summary.json"), summaries)
    write_csv(
        str(run_dir / "scaffold_summary.csv"),
        [[s.get(k) for k in condition_header] for s in condition_rows],
        header=condition_header,
    )
    write_json(str(run_dir / "scaffold_summary.json"), condition_rows)

    if bool(cfg["output"]["make_plot"]):
        make_plots(condition_rows, figures_dir)

    run_summary = {
        "name": cfg["name"],
        "run_dir": str(run_dir),
        "fingerprint": meta.get("fingerprint"),
        "condition_count": len(conditions),
        "seed_count": len(seeds),
        "run_count": len(summaries),
        "runs_summary_path": "runs_summary.csv",
        "runs_summary_json_path": "runs_summary.json",
        "scaffold_summary_path": "scaffold_summary.csv",
        "scaffold_summary_json_path": "scaffold_summary.json",
        "figures": [
            "figures/strong_functional_success_rate.png",
            "figures/mean_open_cost_factor.png",
            "figures/scaffold_utilization_rate.png",
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
