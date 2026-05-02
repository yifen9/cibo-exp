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


def draw_segment(
    canvas: np.ndarray, a: tuple[int, int], b: tuple[int, int], width: int = 1
) -> None:
    x0, y0 = a
    x1, y1 = b

    if x0 == x1:
        lo = min(y0, y1)
        hi = max(y0, y1)
        for x in range(x0 - width // 2, x0 + width // 2 + 1):
            if 0 <= x < canvas.shape[0]:
                canvas[x, lo : hi + 1] = 1
    elif y0 == y1:
        lo = min(x0, x1)
        hi = max(x0, x1)
        for y in range(y0 - width // 2, y0 + width // 2 + 1):
            if 0 <= y < canvas.shape[1]:
                canvas[lo : hi + 1, y] = 1


def make_single_narrow(
    height: int, width: int
) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    canvas = np.zeros((height, width), dtype=np.uint8)
    path = [(16, 3), (16, 10), (12, 10), (12, 18), (20, 18), (20, 28)]

    for a, b in zip(path, path[1:]):
        draw_segment(canvas, a, b, 1)

    start = [(16, 3), (16, 4)]
    goal = [(20, 27), (20, 28)]
    return canvas, start, goal


def make_thick_corridor(
    height: int, width: int
) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    canvas, start, goal = make_single_narrow(height, width)
    thick = np.zeros_like(canvas)

    for x, y in np.argwhere(canvas == 1):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nx = int(x) + dx
                ny = int(y) + dy
                if 0 <= nx < height and 0 <= ny < width:
                    thick[nx, ny] = 1

    return thick, start, goal


def make_double_path(
    height: int, width: int
) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    canvas = np.zeros((height, width), dtype=np.uint8)

    top_path = [(16, 3), (10, 3), (10, 14), (10, 24), (20, 24), (20, 28)]
    bottom_path = [(16, 3), (22, 3), (22, 14), (22, 24), (20, 24), (20, 28)]

    for path in (top_path, bottom_path):
        for a, b in zip(path, path[1:]):
            draw_segment(canvas, a, b, 1)

    draw_segment(canvas, (10, 14), (22, 14), 1)

    start = [(16, 3), (15, 3), (17, 3)]
    goal = [(20, 27), (20, 28)]
    return canvas, start, goal


def make_braided_path(
    height: int, width: int
) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    canvas = np.zeros((height, width), dtype=np.uint8)

    paths = [
        [(16, 3), (9, 7), (9, 16), (13, 22), (20, 28)],
        [(16, 3), (16, 8), (12, 14), (18, 20), (20, 28)],
        [(16, 3), (23, 7), (23, 16), (19, 22), (20, 28)],
    ]

    for path in paths:
        for a, b in zip(path, path[1:]):
            x0, y0 = a
            x1, y1 = b
            x, y = x0, y0
            canvas[x, y] = 1
            while x != x1 or y != y1:
                if x < x1:
                    x += 1
                elif x > x1:
                    x -= 1
                if y < y1:
                    y += 1
                elif y > y1:
                    y -= 1
                canvas[x, y] = 1

    for y in range(8, 24, 4):
        xs = np.argwhere(canvas[:, y] == 1).flatten()
        if len(xs) >= 2:
            draw_segment(canvas, (int(xs.min()), y), (int(xs.max()), y), 1)

    start = [(16, 3), (15, 3), (17, 3)]
    goal = [(20, 27), (20, 28)]
    return canvas, start, goal


def make_maze_like_path(
    height: int, width: int
) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    canvas = np.zeros((height, width), dtype=np.uint8)
    main = [(16, 3), (16, 8), (8, 8), (8, 18), (14, 18), (14, 25), (20, 25), (20, 28)]

    for a, b in zip(main, main[1:]):
        draw_segment(canvas, a, b, 1)

    dead_ends = [
        ((8, 12), (4, 12)),
        ((12, 18), (12, 12)),
        ((14, 22), (23, 22)),
        ((16, 6), (24, 6)),
        ((20, 25), (26, 25)),
    ]

    for a, b in dead_ends:
        draw_segment(canvas, a, b, 1)

    start = [(16, 3), (16, 4)]
    goal = [(20, 27), (20, 28)]
    return canvas, start, goal


def make_sparse_stepping_stone(
    height: int, width: int
) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    canvas = np.zeros((height, width), dtype=np.uint8)
    cells = [
        (16, 3),
        (16, 6),
        (15, 9),
        (13, 12),
        (12, 15),
        (13, 18),
        (16, 21),
        (18, 24),
        (20, 27),
        (20, 28),
    ]

    for x, y in cells:
        canvas[x, y] = 1

    start = [(16, 3)]
    goal = [(20, 28)]
    return canvas, start, goal


def make_path_environment(
    height: int, width: int, path_type: str
) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    if path_type == "single_narrow":
        return make_single_narrow(height, width)

    if path_type == "thick_corridor":
        return make_thick_corridor(height, width)

    if path_type == "double_path":
        return make_double_path(height, width)

    if path_type == "braided_path":
        return make_braided_path(height, width)

    if path_type == "maze_like_path":
        return make_maze_like_path(height, width)

    if path_type == "sparse_stepping_stone":
        return make_sparse_stepping_stone(height, width)

    raise ValueError(f"unknown path type: {path_type}")


def get_cell(canvas: np.ndarray, x: int, y: int) -> int:
    if x < 0 or x >= canvas.shape[0] or y < 0 or y >= canvas.shape[1]:
        return 0
    return int(canvas[x, y])


def connectivity(
    canvas: np.ndarray, start: list[tuple[int, int]], goal: list[tuple[int, int]]
) -> int:
    goal_set = set(goal)
    seen = set()
    q = deque()

    for cell in start:
        x, y = cell
        if (
            0 <= x < canvas.shape[0]
            and 0 <= y < canvas.shape[1]
            and int(canvas[x, y]) == 1
        ):
            q.append(cell)
            seen.add(cell)

    while q:
        x, y = q.popleft()

        if (x, y) in goal_set:
            return 1

        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx = x + dx
            ny = y + dy

            if 0 <= nx < canvas.shape[0] and 0 <= ny < canvas.shape[1]:
                if int(canvas[nx, ny]) == 1 and (nx, ny) not in seen:
                    seen.add((nx, ny))
                    q.append((nx, ny))

    return 0


def shortest_path_length(
    canvas: np.ndarray, start: list[tuple[int, int]], goal: list[tuple[int, int]]
) -> int | None:
    goal_set = set(goal)
    seen = set()
    q = deque()

    for cell in start:
        x, y = cell
        if (
            0 <= x < canvas.shape[0]
            and 0 <= y < canvas.shape[1]
            and int(canvas[x, y]) == 1
        ):
            q.append((cell, 0))
            seen.add(cell)

    while q:
        (x, y), d = q.popleft()

        if (x, y) in goal_set:
            return int(d + 1)

        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx = x + dx
            ny = y + dy

            if 0 <= nx < canvas.shape[0] and 0 <= ny < canvas.shape[1]:
                if int(canvas[nx, ny]) == 1 and (nx, ny) not in seen:
                    seen.add((nx, ny))
                    q.append(((nx, ny), d + 1))

    return None


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


def false_corridor_count(
    canvas: np.ndarray, start: list[tuple[int, int]], goal: list[tuple[int, int]]
) -> int:
    comps = connected_components(canvas == 1)
    start_set = set(start)
    goal_set = set(goal)
    count = 0

    for comp in comps:
        s = set(comp)
        if not (s & start_set and s & goal_set):
            count += 1

    return int(count)


def local_window(canvas: np.ndarray, x: int, y: int, radius: int) -> np.ndarray:
    x0 = max(0, x - radius)
    x1 = min(canvas.shape[0], x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(canvas.shape[1], y + radius + 1)
    return canvas[x0:x1, y0:y1]


def local_density(canvas: np.ndarray, x: int, y: int, radius: int) -> float:
    w = local_window(canvas, x, y, radius)
    center = int(canvas[x, y])
    total = int(w.sum()) - center
    denom = max(1, w.size - 1)
    return float(total / denom)


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


def path_support(
    canvas: np.ndarray, trace: np.ndarray, x: int, y: int, radius: int, alpha: float
) -> float:
    old = int(canvas[x, y])

    if old == 0:
        canvas[x, y] = 1

    density = local_density(canvas, x, y, radius)
    bridge = bridge_evidence(canvas, x, y)

    canvas[x, y] = old

    x0 = max(0, x - radius)
    x1 = min(canvas.shape[0], x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(canvas.shape[1], y + radius + 1)
    tw = trace[x0:x1, y0:y1]
    positive = float(np.maximum(tw, 0.0).sum())
    negative = float(np.maximum(-tw, 0.0).sum())
    trace_score = 1.0 if positive > negative and positive > 0 else 0.0

    return float(alpha * density + (1.0 - alpha) * bridge + 0.10 * trace_score)


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


def bridge_cut(canvas: np.ndarray, reference: np.ndarray, cut_length: int) -> int:
    cells = np.argwhere(reference == 1)
    center = np.array([canvas.shape[0] // 2, canvas.shape[1] // 2])
    distances = np.sum((cells - center) ** 2, axis=1)
    order = np.argsort(distances)
    selected = cells[order[: min(cut_length, len(cells))]]

    changed = 0
    for x, y in selected:
        if int(canvas[int(x), int(y)]) == 1:
            changed += 1
        canvas[int(x), int(y)] = 0

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
            return bridge_cut(canvas, reference, int(condition["cut_length"]))
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
    trace: np.ndarray,
    temporal_pass: bool,
    x: int,
    y: int,
    cfg: dict,
    repair: dict,
    rng: np.random.Generator,
) -> dict:
    radius = int(cfg["operator"]["radius"])
    alpha = float(repair["alpha"])
    center = int(canvas[x, y])
    support = path_support(canvas, trace, x, y, radius, alpha)

    if center == 1:
        if support < float(cfg["operator"]["theta_preserve"]) and rng.random() < float(
            cfg["operator"]["suppression"]
        ):
            return {
                "action": "close",
                "delta": -1,
                "support": support,
                "repair_pass": False,
            }

        return {
            "action": "stay",
            "delta": 0,
            "support": support,
            "repair_pass": False,
        }

    if temporal_pass:
        return {
            "action": "open",
            "delta": 1,
            "support": support,
            "repair_pass": True,
        }

    return {
        "action": "stay",
        "delta": 0,
        "support": support,
        "repair_pass": False,
    }


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


def compute_metrics(
    canvas: np.ndarray,
    reference: np.ndarray,
    start: list[tuple[int, int]],
    goal: list[tuple[int, int]],
    initial_open: int,
) -> dict:
    conn = connectivity(canvas, start, goal)
    sp = shortest_path_length(canvas, start, goal)
    open_cost = int(canvas.sum())
    outside_open = int(((canvas == 1) & (reference == 0)).sum())
    outside_total = int((reference == 0).sum())
    reference_open = int(reference.sum())
    path_tpr = float(((canvas == 1) & (reference == 1)).sum() / max(1, reference_open))
    outside_open_rate = float(outside_open / max(1, outside_total))
    open_cost_factor = float(open_cost / max(1, initial_open))

    return {
        "connectivity": int(conn),
        "shortest_path_length": sp,
        "open_cost": int(open_cost),
        "open_cost_factor": open_cost_factor,
        "outside_open_rate": outside_open_rate,
        "path_tpr": path_tpr,
        "false_corridor_count": int(false_corridor_count(canvas, start, goal)),
    }


def classify_open_action(
    before: np.ndarray, reference: np.ndarray, x: int, y: int
) -> str:
    if int(before[x, y]) == 0 and int(reference[x, y]) == 1:
        return "true_open"
    if int(before[x, y]) == 0 and int(reference[x, y]) == 0:
        return "false_open"
    return "non_open"


def classify_close_action(
    before: np.ndarray, reference: np.ndarray, x: int, y: int
) -> str:
    if int(before[x, y]) == 1 and int(reference[x, y]) == 0:
        return "true_close"
    if int(before[x, y]) == 1 and int(reference[x, y]) == 1:
        return "false_close"
    return "non_close"


def summarize(
    rows: list[list],
    header: list[str],
    condition: dict,
    initial_connected: int,
    initial_open: int,
) -> dict:
    idx = {name: i for i, name in enumerate(header)}
    final = rows[-1]
    connectivity_values = [int(row[idx["connectivity"]]) for row in rows]
    post_damage = [
        row
        for row in rows
        if int(row[idx["t"]])
        >= int(condition.get("damage_time", condition.get("drifting_times", [0])[0]))
    ]

    recovery_time = None
    for row in post_damage:
        if int(row[idx["connectivity"]]) == 1:
            recovery_time = int(row[idx["t"]]) - int(
                condition.get("damage_time", condition.get("drifting_times", [0])[0])
            )
            break

    return {
        "initial_connected": int(initial_connected),
        "initial_open_cost": int(initial_open),
        "final_connectivity": int(final[idx["connectivity"]]),
        "connectivity_stability": float(
            sum(connectivity_values) / max(1, len(connectivity_values))
        ),
        "post_damage_connectivity_stability": float(
            sum(int(row[idx["connectivity"]]) for row in post_damage)
            / max(1, len(post_damage))
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

    reference, start, goal = make_path_environment(
        height, width, str(condition["path_type"])
    )
    canvas = reference.copy()
    trace = np.zeros((height, width), dtype=np.float64)
    history = np.zeros((height, width, int(cfg["repair"]["k"])), dtype=np.uint8)

    initial_connected = connectivity(canvas, start, goal)
    initial_open = int(canvas.sum())

    operators = init_operators(int(cfg["operator"]["count"]), height, width, rng)

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

        if t < steps:
            move_operators(operators, height, width, rng)

            for pos in operators:
                x = int(pos[0])
                y = int(pos[1])
                evidence = path_support(
                    canvas,
                    trace,
                    x,
                    y,
                    int(cfg["operator"]["radius"]),
                    float(cfg["repair"]["alpha"]),
                )
                temporal_pass, temporal_score = update_sliding(
                    history,
                    evidence,
                    float(cfg["repair"]["theta_repair"]),
                    x,
                    y,
                    t,
                    int(cfg["repair"]["k"]),
                    int(cfg["repair"]["m"]),
                )
                result = choose_action(
                    canvas, trace, temporal_pass, x, y, cfg, cfg["repair"], rng
                )
                support_values.append(float(result["support"]))

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
                for item in metas:
                    if int(item["delta"]) > 0:
                        label = classify_open_action(before, reference, x, y)
                        if label == "true_open":
                            true_open_total += 1
                        elif label == "false_open":
                            false_open_total += 1
                    elif int(item["delta"]) < 0:
                        label = classify_close_action(before, reference, x, y)
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

        metrics = compute_metrics(canvas, reference, start, goal, initial_open)

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

    summary.update(
        {
            "condition_id": condition_id,
            "group": str(condition["group"]),
            "seed": int(seed),
            "path_class": str(condition["path_class"]),
            "path_type": str(condition["path_type"]),
            "perturbation_type": str(condition["perturbation_type"]),
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
            "proposal_total": int(sum(int(row[idx["proposal_total"]]) for row in rows)),
            "applied_total": int(sum(int(row[idx["applied_actions"]]) for row in rows)),
            "repair_pass_total": int(
                sum(int(row[idx["repair_pass_total"]]) for row in rows)
            ),
            "repair_reject_total": int(
                sum(int(row[idx["repair_reject_total"]]) for row in rows)
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
    return int(summary["final_connectivity"]) == 1 and float(
        summary["final_open_cost_factor"]
    ) <= float(cfg["metrics"]["strict_cost_factor"])


def failure_mode(summary: dict, cfg: dict) -> str:
    if strong_success(summary, cfg):
        return "functional_success"

    if int(summary["final_connectivity"]) == 0:
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
    stability = [float(x["mean_post_damage_connectivity_stability"]) for x in rows]

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(labels, strong)
    ax.set_title("Experiment 11 strong functional success rate")
    ax.set_xlabel("condition")
    ax.set_ylabel("strong success rate")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(str(figures_dir / "strong_functional_success_rate.png"), dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(labels, cost)
    ax.set_title("Experiment 11 mean final open cost factor")
    ax.set_xlabel("condition")
    ax.set_ylabel("open cost factor")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(str(figures_dir / "mean_open_cost_factor.png"), dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(labels, stability)
    ax.set_title("Experiment 11 post-damage connectivity stability")
    ax.set_xlabel("condition")
    ax.set_ylabel("connectivity stability")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(str(figures_dir / "connectivity_stability.png"), dpi=160)
    plt.close(fig)


def main() -> int:
    root = repo_root()
    config_path = root / "config" / "tests" / "exp_11.yaml"
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
        "seed",
        "path_class",
        "path_type",
        "perturbation_type",
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
        "metrics_path",
    ]

    condition_header = [
        "condition_id",
        "group",
        "path_class",
        "path_type",
        "perturbation_type",
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
        "mean_recovery_time",
    ]

    write_csv(
        str(run_dir / "runs_summary.csv"),
        [[s.get(k) for k in run_header] for s in summaries],
        header=run_header,
    )
    write_json(str(run_dir / "runs_summary.json"), summaries)
    write_csv(
        str(run_dir / "functional_summary.csv"),
        [[s.get(k) for k in condition_header] for s in condition_rows],
        header=condition_header,
    )
    write_json(str(run_dir / "functional_summary.json"), condition_rows)

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
        "functional_summary_path": "functional_summary.csv",
        "functional_summary_json_path": "functional_summary.json",
        "figures": [
            "figures/strong_functional_success_rate.png",
            "figures/mean_open_cost_factor.png",
            "figures/connectivity_stability.png",
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
