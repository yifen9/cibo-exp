from pathlib import Path
from collections import defaultdict, deque
from statistics import mean
from concurrent.futures import ProcessPoolExecutor
import importlib.util
import hashlib

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


_WORKER_EXP11 = None
_WORKER_EXP15 = None
_WORKER_EXP17 = None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def stable_seed(text: str) -> int:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_exp11(root: Path):
    return load_module(
        root / "scripts" / "tests" / "exp_11.py", "cibo_exp_11_runtime_for_exp_18"
    )


def load_exp15(root: Path):
    return load_module(
        root / "scripts" / "tests" / "exp_15.py", "cibo_exp_15_runtime_for_exp_18"
    )


def load_exp17(root: Path):
    return load_module(
        root / "scripts" / "tests" / "exp_17.py", "cibo_exp_17_runtime_for_exp_18"
    )


def init_worker(root_text: str) -> None:
    global _WORKER_EXP11
    global _WORKER_EXP15
    global _WORKER_EXP17
    root = Path(root_text)
    _WORKER_EXP11 = load_exp11(root)
    _WORKER_EXP15 = load_exp15(root)
    _WORKER_EXP17 = load_exp17(root)


def p18_keys() -> list[str]:
    return [
        "a_gap",
        "a_two",
        "a_band",
        "a_frontier",
        "a_bypass",
        "a_deadend",
        "a_budget",
        "a_trace",
        "theta_local",
        "theta_align",
        "theta_trace",
        "rho_budget",
        "trace_decay",
        "trace_refresh",
    ]


def random_p18_genome(rng: np.random.Generator) -> dict:
    return {
        "a_gap": float(rng.uniform(0.0, 1.0)),
        "a_two": float(rng.uniform(0.0, 1.0)),
        "a_band": float(rng.uniform(0.0, 1.0)),
        "a_frontier": float(rng.uniform(0.0, 1.0)),
        "a_bypass": float(rng.uniform(0.0, 1.0)),
        "a_deadend": float(rng.uniform(0.0, 1.0)),
        "a_budget": float(rng.uniform(0.0, 1.0)),
        "a_trace": float(rng.uniform(0.0, 1.0)),
        "theta_local": float(rng.uniform(0.15, 0.55)),
        "theta_align": float(rng.uniform(0.35, 0.85)),
        "theta_trace": float(rng.uniform(0.20, 0.80)),
        "rho_budget": float(rng.uniform(1.05, 1.65)),
        "trace_decay": float(rng.uniform(0.50, 0.98)),
        "trace_refresh": float(rng.uniform(0.25, 1.0)),
    }


def apply_ablation(g: dict, ablation: str) -> dict:
    out = dict(g)

    if ablation == "none":
        return out

    if ablation == "no_gap":
        out["a_gap"] = 0.0
        out["a_two"] = 0.0
        return out

    if ablation == "no_band":
        out["a_band"] = 0.0
        return out

    if ablation == "no_frontier":
        out["a_frontier"] = 0.0
        return out

    if ablation == "no_deadend":
        out["a_deadend"] = 0.0
        out["a_budget"] = 0.0
        return out

    raise ValueError(f"unknown ablation: {ablation}")


def mutate_p18_genome(
    g: dict, ablation: str, rate: float, rng: np.random.Generator
) -> dict:
    out = dict(g)

    for key in p18_keys():
        if rng.random() >= rate:
            continue

        if key in {"theta_local", "theta_align", "theta_trace"}:
            out[key] = float(
                np.clip(float(out[key]) + rng.normal(0.0, 0.06), 0.05, 0.95)
            )
        elif key == "rho_budget":
            out[key] = float(
                np.clip(float(out[key]) + rng.normal(0.0, 0.10), 0.80, 2.20)
            )
        elif key == "trace_decay":
            out[key] = float(
                np.clip(float(out[key]) + rng.normal(0.0, 0.08), 0.40, 0.99)
            )
        else:
            out[key] = float(np.clip(float(out[key]) + rng.normal(0.0, 0.15), 0.0, 1.0))

    return apply_ablation(out, ablation)


def crossover_p18(a: dict, b: dict, ablation: str, rng: np.random.Generator) -> dict:
    g = {key: a[key] if rng.random() < 0.5 else b[key] for key in p18_keys()}
    return apply_ablation(g, ablation)


def random_genome(library: str, condition: dict, rng: np.random.Generator) -> dict:
    if library == "p17":
        return _WORKER_EXP17.random_genome("comp", "C0", rng)

    if library == "p18":
        return apply_ablation(random_p18_genome(rng), str(condition["ablation"]))

    raise ValueError(f"unknown library: {library}")


def mutate_genome(
    g: dict, library: str, condition: dict, rate: float, rng: np.random.Generator
) -> dict:
    if library == "p17":
        return _WORKER_EXP17.mutate_genome(g, "comp", "C0", rate, rng)

    if library == "p18":
        return mutate_p18_genome(g, str(condition["ablation"]), rate, rng)

    raise ValueError(f"unknown library: {library}")


def crossover_genome(
    a: dict, b: dict, library: str, condition: dict, rng: np.random.Generator
) -> dict:
    if library == "p17":
        return _WORKER_EXP17.crossover_genome(a, b, "comp", "C0", rng)

    if library == "p18":
        return crossover_p18(a, b, str(condition["ablation"]), rng)

    raise ValueError(f"unknown library: {library}")


def make_cases(
    seeds: list[int],
    offsets: list[int],
    variants: list[str],
    noise_values: list[bool],
    task: str,
) -> list[dict]:
    cases = []

    for seed in seeds:
        for offset in offsets:
            for variant in variants:
                for noise in sorted(set(noise_values)):
                    if task == "mixed":
                        path_types = ["single_narrow", "double_path"]
                    else:
                        path_types = [None]

                    for path_type in path_types:
                        cases.append(
                            {
                                "seed": int(seed),
                                "offset": int(offset),
                                "variant": str(variant),
                                "noise": bool(noise),
                                "path_type": path_type,
                            }
                        )

    return cases


def make_variant_path_type(path_type: str, variant: str) -> str:
    if variant == "base":
        return path_type

    if path_type == "single_narrow" and variant == "nearby":
        return "thick_corridor"

    if path_type == "double_path" and variant == "nearby":
        return "braided_path"

    return path_type


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


def local_components(window: np.ndarray) -> int:
    h, w = window.shape
    seen = np.zeros_like(window, dtype=np.uint8)
    count = 0

    for i in range(h):
        for j in range(w):
            if int(window[i, j]) == 0 or int(seen[i, j]) == 1:
                continue

            count += 1
            q = deque([(i, j)])
            seen[i, j] = 1

            while q:
                x, y = q.popleft()

                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx = x + dx
                    ny = y + dy

                    if (
                        0 <= nx < h
                        and 0 <= ny < w
                        and int(window[nx, ny]) == 1
                        and int(seen[nx, ny]) == 0
                    ):
                        seen[nx, ny] = 1
                        q.append((nx, ny))

    return int(count)


def local_density(canvas: np.ndarray, x: int, y: int, radius: int) -> float:
    w = local_window(canvas, x, y, radius)
    center = int(canvas[x, y])
    return float((int(w.sum()) - center) / max(1, w.size - 1))


def alignment_score(canvas: np.ndarray, x: int, y: int) -> float:
    horizontal = get_cell(canvas, x, y - 1) + get_cell(canvas, x, y + 1)
    vertical = get_cell(canvas, x - 1, y) + get_cell(canvas, x + 1, y)
    diag1 = get_cell(canvas, x - 1, y - 1) + get_cell(canvas, x + 1, y + 1)
    diag2 = get_cell(canvas, x - 1, y + 1) + get_cell(canvas, x + 1, y - 1)
    return float(max(horizontal, vertical, diag1, diag2) / 2.0)


def gap_closure(canvas: np.ndarray, x: int, y: int) -> float:
    if int(canvas[x, y]) == 1:
        return 0.0

    before = local_window(canvas, x, y, 1).copy()
    before_components = local_components(before)

    old = int(canvas[x, y])
    canvas[x, y] = 1
    after = local_window(canvas, x, y, 1).copy()
    after_components = local_components(after)
    canvas[x, y] = old

    if before_components >= 2 and after_components < before_components:
        return 1.0

    if before_components >= 2:
        return 0.5

    return 0.0


def two_sided_support(canvas: np.ndarray, x: int, y: int) -> float:
    left = get_cell(canvas, x, y - 1)
    right = get_cell(canvas, x, y + 1)
    up = get_cell(canvas, x - 1, y)
    down = get_cell(canvas, x + 1, y)

    horizontal = 1.0 if left and right else 0.0
    vertical = 1.0 if up and down else 0.0
    corner = 0.5 if (left or right) and (up or down) else 0.0

    return float(min(1.0, horizontal + vertical + corner))


def corridor_band(canvas: np.ndarray, x: int, y: int, genome: dict) -> float:
    density = local_density(canvas, x, y, 1)
    align = alignment_score(canvas, x, y)

    if density >= float(genome["theta_local"]) and align >= float(
        genome["theta_align"]
    ):
        return 1.0

    return float(0.5 * density + 0.5 * align)


def component_frontier(canvas: np.ndarray, x: int, y: int) -> float:
    if int(canvas[x, y]) == 1:
        return 0.0

    neighbor_open = (
        get_cell(canvas, x - 1, y)
        + get_cell(canvas, x + 1, y)
        + get_cell(canvas, x, y - 1)
        + get_cell(canvas, x, y + 1)
    )

    if neighbor_open >= 2:
        return 1.0

    if neighbor_open == 1:
        return 0.5

    return 0.0


def local_shortest(
    window: np.ndarray, start: tuple[int, int], goal: tuple[int, int]
) -> int | None:
    if int(window[start]) == 0 or int(window[goal]) == 0:
        return None

    q = deque([(start[0], start[1], 0)])
    seen = {start}

    while q:
        x, y, d = q.popleft()

        if (x, y) == goal:
            return int(d)

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx = x + dx
            ny = y + dy

            if (
                0 <= nx < window.shape[0]
                and 0 <= ny < window.shape[1]
                and int(window[nx, ny]) == 1
                and (nx, ny) not in seen
            ):
                seen.add((nx, ny))
                q.append((nx, ny, d + 1))

    return None


def short_bypass(canvas: np.ndarray, x: int, y: int) -> float:
    w = local_window(canvas, x, y, 2).copy()

    if w.shape[0] < 3 or w.shape[1] < 3:
        return 0.0

    opens = [
        (i, j)
        for i in range(w.shape[0])
        for j in range(w.shape[1])
        if int(w[i, j]) == 1
    ]

    if len(opens) < 2:
        return 0.0

    best_gain = 0.0
    cx = min(2, w.shape[0] - 1)
    cy = min(2, w.shape[1] - 1)

    for a in opens[:8]:
        for b in opens[:8]:
            if a == b:
                continue

            before = local_shortest(w, a, b)
            old = int(w[cx, cy])
            w[cx, cy] = 1
            after = local_shortest(w, a, b)
            w[cx, cy] = old

            if before is None and after is not None:
                best_gain = max(best_gain, 1.0)
            elif before is not None and after is not None and after < before:
                best_gain = max(
                    best_gain, min(1.0, float((before - after) / max(1, before)))
                )

    return float(best_gain)


def deadend_penalty(canvas: np.ndarray, x: int, y: int) -> float:
    old = int(canvas[x, y])
    canvas[x, y] = 1

    degree = (
        get_cell(canvas, x - 1, y)
        + get_cell(canvas, x + 1, y)
        + get_cell(canvas, x, y - 1)
        + get_cell(canvas, x, y + 1)
    )

    canvas[x, y] = old

    if degree <= 1:
        return 1.0

    if degree >= 3:
        return 0.25

    return 0.0


def budget_penalty(canvas: np.ndarray, initial_open: int, genome: dict) -> float:
    factor = float(canvas.sum() / max(1, initial_open))
    return float(max(0.0, factor - float(genome["rho_budget"])))


def trace_evidence(trace: np.ndarray, x: int, y: int, genome: dict) -> float:
    value = float(trace[x, y])
    threshold = float(genome["theta_trace"])

    if value >= threshold:
        return 1.0

    return float(np.clip(value / max(1e-9, threshold), 0.0, 1.0))


def p18_support(
    canvas: np.ndarray,
    trace: np.ndarray,
    genome: dict,
    x: int,
    y: int,
    cfg: dict,
    initial_open: int,
) -> float:
    old = int(canvas[x, y])

    if old == 0:
        canvas[x, y] = 1

    gap = gap_closure(canvas, x, y)
    two = two_sided_support(canvas, x, y)
    band = corridor_band(canvas, x, y, genome)
    frontier = component_frontier(canvas, x, y)
    bypass = short_bypass(canvas, x, y)
    deadend = deadend_penalty(canvas, x, y)
    budget = budget_penalty(canvas, initial_open, genome)
    trace_score = trace_evidence(trace, x, y, genome)

    canvas[x, y] = old

    score = (
        float(genome["a_gap"]) * gap
        + float(genome["a_two"]) * two
        + float(genome["a_band"]) * band
        + float(genome["a_frontier"]) * frontier
        + float(genome["a_bypass"]) * bypass
        + float(genome["a_trace"]) * trace_score
        - float(genome["a_deadend"]) * deadend
        - float(genome["a_budget"]) * budget
    )

    norm = (
        abs(float(genome["a_gap"]))
        + abs(float(genome["a_two"]))
        + abs(float(genome["a_band"]))
        + abs(float(genome["a_frontier"]))
        + abs(float(genome["a_bypass"]))
        + abs(float(genome["a_trace"]))
        + abs(float(genome["a_deadend"]))
        + abs(float(genome["a_budget"]))
    )

    return float(np.clip(score / max(1.0, norm), 0.0, 1.0))


def p18_complexity(genome: dict) -> float:
    active_keys = [
        "a_gap",
        "a_two",
        "a_band",
        "a_frontier",
        "a_bypass",
        "a_deadend",
        "a_budget",
        "a_trace",
    ]
    active = sum(1 for key in active_keys if abs(float(genome[key])) > 0.05)
    weight_sum = sum(abs(float(genome[key])) for key in active_keys)
    return float(active + 0.20 * weight_sum)


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


def bridge_center_cells(
    reference: np.ndarray, count: int, offset: int = 0
) -> list[tuple[int, int]]:
    cells = np.argwhere(reference == 1)

    if len(cells) == 0:
        return []

    center = np.array([reference.shape[0] // 2, reference.shape[1] // 2 + offset])
    distances = np.sum((cells - center) ** 2, axis=1)
    order = np.argsort(distances)
    selected = cells[order[: min(count, len(cells))]]
    return [(int(x), int(y)) for x, y in selected]


def bridge_cut(
    canvas: np.ndarray, reference: np.ndarray, condition: dict, offset: int = 0
) -> int:
    cells = bridge_center_cells(reference, int(condition["cut_length"]), offset=offset)
    changed = 0

    for x, y in cells:
        if int(canvas[x, y]) == 1:
            changed += 1
        canvas[x, y] = 0

    return int(changed)


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


def update_trace(trace: np.ndarray, proposals: dict, cfg: dict, genome: dict) -> None:
    trace *= float(cfg["trace"]["decay"]) * max(0.10, float(genome["trace_decay"]))

    for (x, y), values in proposals.items():
        score = sum(values)
        if score > 0:
            trace[x, y] = float(cfg["trace"]["duration"]) * max(
                0.0, float(genome["trace_refresh"])
            )
        elif score < 0:
            trace[x, y] = -float(cfg["trace"]["duration"]) * max(
                0.0, float(genome["trace_refresh"])
            )


def classify_open(before: np.ndarray, reference: np.ndarray, x: int, y: int) -> str:
    if int(before[x, y]) == 0 and int(reference[x, y]) == 1:
        return "true_open"
    if int(before[x, y]) == 0 and int(reference[x, y]) == 0:
        return "false_open"
    return "non_open"


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


def run_p18_trial(exp11, cfg: dict, condition: dict, genome: dict, case: dict) -> dict:
    height = int(cfg["canvas"]["height"])
    width = int(cfg["canvas"]["width"])
    steps = int(cfg["run"]["steps"])
    seed = int(case["seed"])
    offset = int(case["offset"])
    variant = str(case["variant"])
    noise = bool(case["noise"])
    path_override = case.get("path_type")
    rng = np.random.default_rng(seed * 10007 + offset * 101 + stable_seed(variant))

    base_path_type = (
        str(path_override) if path_override else str(condition["path_type"])
    )
    path_type = make_variant_path_type(base_path_type, variant)
    reference, start, goal = exp11.make_path_environment(height, width, path_type)
    canvas = reference.copy()
    trace = np.zeros((height, width), dtype=np.float64)
    history = np.zeros((height, width, int(cfg["repair"]["k"])), dtype=np.uint8)
    operators = init_operators(int(cfg["operator"]["count"]), height, width, rng)
    initial_open = int(canvas.sum())

    connectivity_values = []
    open_cost_factors = []
    outside_rates = []
    false_counts = []
    true_open_total = 0
    false_open_total = 0
    applied_total = 0

    for t in range(steps + 1):
        if t == int(condition["damage_time"]):
            bridge_cut(canvas, reference, condition, offset=offset)

        proposals = defaultdict(list)
        proposal_meta = defaultdict(list)

        if t < steps:
            move_operators(operators, height, width, rng)

            for pos in operators:
                x = int(pos[0])
                y = int(pos[1])
                support = p18_support(canvas, trace, genome, x, y, cfg, initial_open)

                if noise:
                    support = float(
                        np.clip(
                            support
                            + rng.normal(0.0, float(condition.get("sigma", 0.10))),
                            0.0,
                            1.0,
                        )
                    )

                temporal_pass, _ = update_sliding(
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

                if result["action"] != "stay":
                    proposals[(x, y)].append(int(result["delta"]))
                    proposal_meta[(x, y)].append(result)

            before = canvas.copy()
            stats = resolve_actions(canvas, proposals)
            applied_total += int(stats["applied_actions"])

            for cell, metas in proposal_meta.items():
                x, y = cell
                for item in metas:
                    if int(item["delta"]) > 0:
                        label = classify_open(before, reference, x, y)
                        if label == "true_open":
                            true_open_total += 1
                        elif label == "false_open":
                            false_open_total += 1

            update_trace(trace, proposals, cfg, genome)

        metrics = compute_metrics(exp11, canvas, reference, start, goal, initial_open)
        connectivity_values.append(int(metrics["connectivity"]))
        open_cost_factors.append(float(metrics["open_cost_factor"]))
        outside_rates.append(float(metrics["outside_open_rate"]))
        false_counts.append(float(metrics["false_corridor_count"]))

    final = compute_metrics(exp11, canvas, reference, start, goal, initial_open)
    post = connectivity_values[int(condition["damage_time"]) :]
    open_action_total = true_open_total + false_open_total

    return {
        "seed": int(seed),
        "offset": int(offset),
        "variant": variant,
        "noise": bool(noise),
        "path_type": str(base_path_type),
        "final_connectivity": int(final["connectivity"]),
        "post_damage_connectivity_stability": float(sum(post) / max(1, len(post))),
        "final_open_cost_factor": float(final["open_cost_factor"]),
        "mean_open_cost_factor": float(mean(open_cost_factors)),
        "final_outside_open_rate": float(final["outside_open_rate"]),
        "mean_outside_open_rate": float(mean(outside_rates)),
        "final_false_corridor_count": int(final["false_corridor_count"]),
        "mean_false_corridor_count": float(mean(false_counts)),
        "final_path_tpr": float(final["path_tpr"]),
        "open_precision": float(true_open_total / open_action_total)
        if open_action_total
        else 0.0,
        "false_open_rate": float(false_open_total / open_action_total)
        if open_action_total
        else 0.0,
        "applied_total": int(applied_total),
        "scaffold_complexity": p18_complexity(genome),
    }


def run_trial_by_library(
    exp11,
    exp15,
    exp17,
    cfg: dict,
    condition: dict,
    genome: dict,
    case: dict,
    library: str,
) -> dict:
    if library == "p17":
        c = dict(condition)
        c["genome_kind"] = "comp"
        c["representation"] = "C0"
        return exp17.run_trial_by_kind(exp11, exp15, cfg, c, genome, case, "comp")

    if library == "p18":
        return run_p18_trial(exp11, cfg, condition, genome, case)

    raise ValueError(f"unknown library: {library}")


def base_score(trial: dict, cfg: dict) -> float:
    w = cfg["fitness_weights"]

    return float(
        float(w["connectivity"]) * float(trial["final_connectivity"])
        + float(w["stability"]) * float(trial["post_damage_connectivity_stability"])
        - float(w["cost"]) * float(trial["final_open_cost_factor"])
        - float(w["outside"]) * float(trial["final_outside_open_rate"])
        - float(w["false_corridor"]) * float(trial["final_false_corridor_count"])
        - float(w["complexity"]) * float(trial["scaffold_complexity"])
    )


def cvar_value(scores: list[float], alpha: float) -> float:
    if not scores:
        return 0.0

    count = max(1, int(round(len(scores) * alpha)))
    return float(mean(sorted(scores)[:count]))


def aggregate_fitness(scores: list[float], family: str, alpha: float) -> float:
    if not scores:
        return 0.0

    if family == "mean":
        return float(mean(scores))

    if family == "worst":
        return float(min(scores))

    if family == "cvar":
        return cvar_value(scores, alpha)

    raise ValueError(f"unknown fitness family: {family}")


def trial_is_strong_success(trial: dict, cfg: dict) -> bool:
    return int(trial["final_connectivity"]) == 1 and float(
        trial["final_open_cost_factor"]
    ) <= float(cfg["metrics"]["strong_cost_factor"])


def trial_is_strict_success(trial: dict, cfg: dict) -> bool:
    return (
        int(trial["final_connectivity"]) == 1
        and float(trial["final_open_cost_factor"])
        <= float(cfg["metrics"]["strict_cost_factor"])
        and float(trial["final_outside_open_rate"])
        <= float(cfg["metrics"]["strict_outside_open_rate"])
    )


def worst_subgroup_success(trials: list[dict], cfg: dict) -> float:
    groups = {}

    for trial in trials:
        key = (
            str(trial["path_type"]),
            str(trial["variant"]),
            int(trial["offset"]),
            bool(trial["noise"]),
        )
        groups.setdefault(key, []).append(trial)

    if not groups:
        return 0.0

    rates = []

    for items in groups.values():
        strong = [1 if trial_is_strong_success(x, cfg) else 0 for x in items]
        rates.append(float(sum(strong) / max(1, len(strong))))

    return float(min(rates))


def success_rates(eval_result: dict, cfg: dict) -> dict:
    trials = eval_result["trials"]
    weak = [1 if int(x["final_connectivity"]) == 1 else 0 for x in trials]
    strong = [1 if trial_is_strong_success(x, cfg) else 0 for x in trials]
    strict = [1 if trial_is_strict_success(x, cfg) else 0 for x in trials]

    return {
        "weak_success_rate": float(sum(weak) / max(1, len(weak))),
        "strong_success_rate": float(sum(strong) / max(1, len(strong))),
        "strict_success_rate": float(sum(strict) / max(1, len(strict))),
    }


def summarize_eval(
    fitness_value: float, scores: list[float], trials: list[dict], cfg: dict
) -> dict:
    return {
        "fitness": float(fitness_value),
        "mean_base_score": float(mean(scores)) if scores else 0.0,
        "worst_base_score": float(min(scores)) if scores else 0.0,
        "cvar_base_score": cvar_value(scores, float(cfg["evolution"]["cvar_alpha"])),
        "mean_final_connectivity": float(
            mean([float(x["final_connectivity"]) for x in trials])
        ),
        "mean_stability": float(
            mean([float(x["post_damage_connectivity_stability"]) for x in trials])
        ),
        "mean_cost": float(mean([float(x["final_open_cost_factor"]) for x in trials])),
        "mean_outside": float(
            mean([float(x["final_outside_open_rate"]) for x in trials])
        ),
        "mean_false": float(
            mean([float(x["final_false_corridor_count"]) for x in trials])
        ),
        "mean_path_tpr": float(mean([float(x["final_path_tpr"]) for x in trials])),
        "mean_open_precision": float(
            mean([float(x["open_precision"]) for x in trials])
        ),
        "mean_false_open_rate": float(
            mean([float(x["false_open_rate"]) for x in trials])
        ),
        "mean_complexity": float(
            mean([float(x["scaffold_complexity"]) for x in trials])
        ),
        "worst_case_strong_success": float(worst_subgroup_success(trials, cfg)),
        "trials": trials,
    }


def evaluate_genome_with_modules(
    exp11,
    exp15,
    exp17,
    cfg: dict,
    condition: dict,
    genome: dict,
    cases: list[dict],
    library: str,
) -> dict:
    trials = [
        run_trial_by_library(exp11, exp15, exp17, cfg, condition, genome, case, library)
        for case in cases
    ]
    scores = [base_score(trial, cfg) for trial in trials]
    fitness_value = aggregate_fitness(
        scores, str(condition["fitness_family"]), float(cfg["evolution"]["cvar_alpha"])
    )
    return summarize_eval(fitness_value, scores, trials, cfg)


def evaluate_worker(args: tuple) -> dict:
    cfg, condition, genome, cases, library = args
    return evaluate_genome_with_modules(
        _WORKER_EXP11,
        _WORKER_EXP15,
        _WORKER_EXP17,
        cfg,
        condition,
        genome,
        cases,
        library,
    )


def evaluate_population(
    executor: ProcessPoolExecutor,
    cfg: dict,
    condition: dict,
    population: list[dict],
    cases: list[dict],
    library: str,
) -> list[dict]:
    jobs = [(cfg, condition, genome, cases, library) for genome in population]
    return list(executor.map(evaluate_worker, jobs, chunksize=1))


def evaluate_single(
    executor: ProcessPoolExecutor,
    cfg: dict,
    condition: dict,
    genome: dict,
    cases: list[dict],
    library: str,
) -> dict:
    return evaluate_population(executor, cfg, condition, [genome], cases, library)[0]


def tournament(
    population: list[dict], scores: list[float], size: int, rng: np.random.Generator
) -> dict:
    idx = rng.choice(len(population), size=size, replace=False)
    best = max(idx, key=lambda i: scores[int(i)])
    return population[int(best)]


def matched_random_search(
    executor: ProcessPoolExecutor,
    cfg: dict,
    condition: dict,
    train_cases: list[dict],
    test_cases: list[dict],
    library: str,
    rng: np.random.Generator,
    logger: Logger,
) -> dict:
    pop_size = int(cfg["evolution"]["population_size"])
    generations = int(cfg["evolution"]["generations"])
    budget = pop_size * generations
    chunk_size = int(cfg["evolution"]["random_chunk_size"])

    best_train = None
    best_genome = None
    evaluated = 0

    while evaluated < budget:
        n = min(chunk_size, budget - evaluated)
        candidates = [random_genome(library, condition, rng) for _ in range(n)]
        evaluations = evaluate_population(
            executor, cfg, condition, candidates, train_cases, library
        )

        for genome, evaluation in zip(candidates, evaluations):
            if best_train is None or float(evaluation["fitness"]) > float(
                best_train["fitness"]
            ):
                best_train = evaluation
                best_genome = genome

        evaluated += n

        logger.info(
            jline(
                "condition",
                str(condition["id"]),
                "matched_random_progress",
                evaluated=int(evaluated),
                budget=int(budget),
                best_fitness=float(best_train["fitness"]),
            )
        )

    test_eval = evaluate_single(
        executor, cfg, condition, best_genome, test_cases, library
    )

    return {
        "budget": int(budget),
        "genome": best_genome,
        "train": best_train,
        "test": test_eval,
    }


def no_scaffold_p18() -> dict:
    return {
        "a_gap": 0.0,
        "a_two": 0.0,
        "a_band": 0.0,
        "a_frontier": 0.0,
        "a_bypass": 0.0,
        "a_deadend": 0.0,
        "a_budget": 0.0,
        "a_trace": 0.0,
        "theta_local": 0.50,
        "theta_align": 0.70,
        "theta_trace": 0.50,
        "rho_budget": 1.20,
        "trace_decay": 0.80,
        "trace_refresh": 0.0,
    }


def hand_width_direct() -> dict:
    return {
        "w_radius": 1,
        "w_prob": 1.0,
        "w_shape": 0,
        "m_intensity": 0.0,
        "m_radius": 0,
        "m_decay": 0.0,
        "d_strength": 0.0,
        "d_smooth": 0.0,
        "d_decay": 0.0,
        "c_branch": 0.0,
        "c_isolated": 0.30,
        "c_outside": 0.0,
        "p_trace": 0.30,
        "p_decay": 0.80,
        "p_refresh": 0.70,
    }


def hand_combined_direct() -> dict:
    g = hand_width_direct()
    g["m_intensity"] = 1.0
    g["m_radius"] = 2
    g["d_strength"] = 0.60
    g["c_branch"] = 0.40
    g["c_outside"] = 0.50
    g["p_trace"] = 0.60
    return g


def global_reference_eval(cfg: dict, condition: dict, cases: list[dict]) -> dict:
    trials = []

    for case in cases:
        trials.append(
            {
                "seed": int(case["seed"]),
                "offset": int(case["offset"]),
                "variant": str(case["variant"]),
                "noise": bool(case["noise"]),
                "path_type": str(case.get("path_type") or condition["path_type"]),
                "final_connectivity": 1,
                "post_damage_connectivity_stability": 1.0,
                "final_open_cost_factor": 1.0,
                "mean_open_cost_factor": 1.0,
                "final_outside_open_rate": 0.0,
                "mean_outside_open_rate": 0.0,
                "final_false_corridor_count": 0,
                "mean_false_corridor_count": 0.0,
                "final_path_tpr": 1.0,
                "open_precision": 1.0,
                "false_open_rate": 0.0,
                "applied_total": int(condition["cut_length"]),
                "scaffold_complexity": 0.0,
            }
        )

    scores = [base_score(x, cfg) for x in trials]
    return summarize_eval(float(mean(scores)), scores, trials, cfg)


def eval_to_prefixed(prefix: str, eval_result: dict, cfg: dict) -> dict:
    rates = success_rates(eval_result, cfg)

    return {
        f"{prefix}_fitness": float(eval_result["fitness"]),
        f"{prefix}_mean_base_score": float(eval_result["mean_base_score"]),
        f"{prefix}_worst_base_score": float(eval_result["worst_base_score"]),
        f"{prefix}_cvar_base_score": float(eval_result["cvar_base_score"]),
        f"{prefix}_mean_connectivity": float(eval_result["mean_final_connectivity"]),
        f"{prefix}_mean_stability": float(eval_result["mean_stability"]),
        f"{prefix}_mean_cost": float(eval_result["mean_cost"]),
        f"{prefix}_mean_outside": float(eval_result["mean_outside"]),
        f"{prefix}_mean_false": float(eval_result["mean_false"]),
        f"{prefix}_mean_path_tpr": float(eval_result["mean_path_tpr"]),
        f"{prefix}_mean_open_precision": float(eval_result["mean_open_precision"]),
        f"{prefix}_mean_false_open_rate": float(eval_result["mean_false_open_rate"]),
        f"{prefix}_mean_complexity": float(eval_result["mean_complexity"]),
        f"{prefix}_worst_case_strong_success": float(
            eval_result["worst_case_strong_success"]
        ),
        f"{prefix}_weak_success_rate": float(rates["weak_success_rate"]),
        f"{prefix}_strong_success_rate": float(rates["strong_success_rate"]),
        f"{prefix}_strict_success_rate": float(rates["strict_success_rate"]),
    }


def primitive_usage_count(genome: dict, library: str) -> int | None:
    if library != "p18":
        return None

    keys = [
        "a_gap",
        "a_two",
        "a_band",
        "a_frontier",
        "a_bypass",
        "a_deadend",
        "a_budget",
        "a_trace",
    ]
    return int(sum(1 for key in keys if abs(float(genome[key])) > 0.05))


def primitive_diversity(genome: dict, library: str) -> float | None:
    if library != "p18":
        return None

    keys = [
        "a_gap",
        "a_two",
        "a_band",
        "a_frontier",
        "a_bypass",
        "a_deadend",
        "a_budget",
        "a_trace",
    ]
    weights = [abs(float(genome[key])) for key in keys]
    total = sum(weights)

    if total <= 0:
        return 0.0

    probs = [x / total for x in weights if x > 0]
    entropy = -sum(p * np.log(p) for p in probs)
    return float(entropy / np.log(len(keys)))


def gap_closure_score(genome: dict, library: str) -> float | None:
    if library != "p18":
        return None

    return float(
        np.clip(0.55 * float(genome["a_gap"]) + 0.45 * float(genome["a_two"]), 0.0, 1.0)
    )


def corridor_band_score(genome: dict, library: str) -> float | None:
    if library != "p18":
        return None

    return float(
        np.clip(
            0.60 * float(genome["a_band"])
            + 0.20 * float(genome["a_two"])
            + 0.20 * float(genome["a_trace"]),
            0.0,
            1.0,
        )
    )


def bypass_use_score(genome: dict, library: str) -> float | None:
    if library != "p18":
        return None

    return float(
        np.clip(
            0.50 * float(genome["a_bypass"])
            + 0.35 * float(genome["a_frontier"])
            + 0.15 * float(genome["a_two"]),
            0.0,
            1.0,
        )
    )


def false_corridor_control_score(genome: dict, library: str) -> float | None:
    if library != "p18":
        return None

    return float(
        np.clip(
            0.60 * float(genome["a_deadend"]) + 0.40 * float(genome["a_budget"]),
            0.0,
            1.0,
        )
    )


def prior_leakage_score(genome: dict, library: str) -> float:
    if library == "p17":
        return 0.35

    gap = gap_closure_score(genome, library) or 0.0
    band = corridor_band_score(genome, library) or 0.0
    bypass = bypass_use_score(genome, library) or 0.0
    directness = max(
        float(genome["a_gap"]), float(genome["a_band"]), float(genome["a_bypass"])
    )
    return float(
        np.clip(0.25 * gap + 0.25 * band + 0.25 * bypass + 0.25 * directness, 0.0, 1.0)
    )


def expressiveness_score(summary: dict) -> float:
    return float(
        np.clip(
            0.50 * float(summary["evolved_mean_connectivity"])
            + 0.25 * float(summary["evolved_strong_success_rate"])
            + 0.25 * float(summary["evolved_worst_case_strong_success"]),
            0.0,
            1.0,
        )
    )


def robust_discovery_pass(summary: dict, cfg: dict) -> bool:
    return (
        float(summary["evolved_mean_connectivity"])
        >= float(cfg["metrics"]["strong_connectivity_threshold"])
        and float(summary["evolved_mean_cost"])
        <= float(cfg["metrics"]["strong_cost_factor"])
        and float(summary["evolved_mean_outside"])
        <= float(cfg["metrics"]["strict_outside_open_rate"])
        and float(summary["matched_evolution_gain"])
        > float(cfg["evolution"]["discovery_margin"])
        and abs(float(summary["generalization_gap"]))
        <= float(cfg["metrics"]["robust_max_generalization_gap"])
        and float(summary["prior_leakage_score"])
        < float(cfg["metrics"]["prior_leakage_threshold"])
        and float(summary["expressiveness_score"])
        > float(cfg["metrics"]["expressiveness_threshold"])
    )


def failure_mode(summary: dict, cfg: dict) -> str:
    if robust_discovery_pass(summary, cfg):
        return "robust_redesigned_compositional_discovery"

    if (
        float(summary["evolved_mean_connectivity"]) < 0.25
        and float(summary["random_mean_connectivity"]) < 0.25
    ):
        return "under_expressiveness_persists"

    if float(summary["prior_leakage_score"]) >= float(
        cfg["metrics"]["prior_leakage_threshold"]
    ):
        return "prior_leakage_returns"

    if float(summary["matched_evolution_gain"]) <= float(
        cfg["evolution"]["discovery_margin"]
    ):
        return "random_search_equivalence"

    if (
        float(summary["gap_closure_score"] or 0.0) > 0.70
        and float(summary["evolved_mean_connectivity"]) < 0.50
    ):
        return "gap_closure_without_connectivity"

    if (
        float(summary["bypass_use_score"] or 0.0) > 0.70
        and float(summary["evolved_mean_false"]) > 2.0
    ):
        return "false_bypass_expansion"

    if (
        float(summary["false_corridor_control_score"] or 0.0) > 0.80
        and float(summary["evolved_mean_connectivity"]) < 0.50
    ):
        return "over_suppression"

    if float(summary["generalization_gap"]) > float(
        cfg["metrics"]["robust_max_generalization_gap"]
    ):
        return "train_test_collapse"

    if (
        bool(summary["noise_stress"])
        and float(summary["evolved_strong_success_rate"]) < 0.90
    ):
        return "trace_or_noise_fragility"

    return "mixed_failure"


def sample_efficiency(history_rows: list[list], cfg: dict) -> int | None:
    pop_size = int(cfg["evolution"]["population_size"])

    for row in history_rows:
        generation = int(row[0])
        conn = float(row[3])
        cost = float(row[4])
        outside = float(row[5])

        if (
            conn >= float(cfg["metrics"]["strong_connectivity_threshold"])
            and cost <= float(cfg["metrics"]["strong_cost_factor"])
            and outside <= float(cfg["metrics"]["strict_outside_open_rate"])
        ):
            return int((generation + 1) * pop_size)

    return None


def condition_summary(condition: dict, record: dict, cfg: dict) -> dict:
    library = str(condition["library"])
    genome = record["genome"]

    out = {
        "condition_id": str(condition["id"]),
        "group": str(condition["group"]),
        "task": str(condition["task"]),
        "task_distribution": str(condition["task_distribution"]),
        "library": library,
        "ablation": str(condition["ablation"]),
        "path_type": str(condition["path_type"]),
        "fitness_family": str(condition["fitness_family"]),
        "representation": str(condition["representation"]),
        "noise_stress": bool(condition.get("noise_stress", False)),
        "discovery_generation": int(record["generation"]),
        "sample_efficiency": record["sample_efficiency"],
        "matched_random_budget": int(record["matched_random"]["budget"]),
        "best_genome": genome,
        "primitive_usage_count": primitive_usage_count(genome, library),
        "primitive_diversity": primitive_diversity(genome, library),
        "gap_closure_score": gap_closure_score(genome, library),
        "corridor_band_score": corridor_band_score(genome, library),
        "bypass_use_score": bypass_use_score(genome, library),
        "false_corridor_control_score": false_corridor_control_score(genome, library),
        "prior_leakage_score": prior_leakage_score(genome, library),
    }

    out.update(eval_to_prefixed("train", record["train"], cfg))
    out.update(eval_to_prefixed("evolved", record["test"], cfg))
    out.update(eval_to_prefixed("random", record["matched_random"]["test"], cfg))
    out.update(eval_to_prefixed("random_train", record["matched_random"]["train"], cfg))
    out.update(eval_to_prefixed("no_scaffold", record["no_scaffold_reference"], cfg))
    out.update(eval_to_prefixed("hand", record["hand_reference"], cfg))
    out.update(eval_to_prefixed("global", record["global_reference"], cfg))

    out["generalization_gap"] = float(out["train_fitness"] - out["evolved_fitness"])
    out["matched_evolution_gain"] = float(
        out["evolved_fitness"] - out["random_fitness"]
    )
    out["strong_success_gain_over_random"] = float(
        out["evolved_strong_success_rate"] - out["random_strong_success_rate"]
    )
    out["random_search_equivalent"] = bool(
        out["matched_evolution_gain"] <= float(cfg["evolution"]["discovery_margin"])
    )
    out["expressiveness_score"] = expressiveness_score(out)
    out["robust_discovery_pass"] = robust_discovery_pass(out, cfg)
    out["failure_mode"] = failure_mode(out, cfg)

    return out


def evolve_condition(
    cfg: dict,
    condition: dict,
    root: Path,
    run_dir: Path,
    logger: Logger,
    progress: Progress,
) -> dict:
    rng = np.random.default_rng(stable_seed(str(condition["id"])))
    evo = cfg["evolution"]
    library = str(condition["library"])
    pop_size = int(evo["population_size"])
    generations = int(evo["generations"])
    elite_count = int(evo["elite_count"])
    mutation_rate = float(evo["mutation_rate"])
    workers = int(cfg["parallel"]["workers"])

    train_cases = make_cases(
        [int(x) for x in cfg["run"]["train_seeds"]],
        [int(x) for x in cfg["run"]["train_offsets"]],
        [str(x) for x in cfg["run"]["train_variants"]],
        [False],
        str(condition["task"]),
    )

    test_cases = make_cases(
        [int(x) for x in cfg["run"]["test_seeds"]],
        [int(x) for x in cfg["run"]["test_offsets"]],
        [str(x) for x in cfg["run"]["test_variants"]],
        [False, bool(condition.get("noise_stress", False))],
        str(condition["task"]),
    )

    condition_dir = run_dir / "conditions" / str(condition["id"])
    condition_dir.mkdir(parents=True, exist_ok=True)

    population = [random_genome(library, condition, rng) for _ in range(pop_size)]
    history_rows = []
    best_record = None

    logger.info(
        jline(
            "condition",
            str(condition["id"]),
            "cases",
            train_cases=len(train_cases),
            test_cases=len(test_cases),
            matched_random_budget=pop_size * generations,
            workers=workers,
        )
    )

    with ProcessPoolExecutor(
        max_workers=workers, initializer=init_worker, initargs=(str(root),)
    ) as executor:
        for gen in range(generations):
            evaluations = evaluate_population(
                executor, cfg, condition, population, train_cases, library
            )
            scores = [float(x["fitness"]) for x in evaluations]
            order = sorted(
                range(len(population)), key=lambda i: scores[i], reverse=True
            )
            best_idx = order[0]
            best_eval = evaluations[best_idx]
            best_fit = float(scores[best_idx])
            mean_fit = float(mean(scores))

            if best_record is None or best_fit > float(best_record["train"]["fitness"]):
                best_record = {
                    "generation": int(gen),
                    "genome": dict(population[best_idx]),
                    "train": best_eval,
                }

            history_rows.append(
                [
                    int(gen),
                    best_fit,
                    mean_fit,
                    float(best_eval["mean_final_connectivity"]),
                    float(best_eval["mean_cost"]),
                    float(best_eval["mean_outside"]),
                    float(best_eval["worst_case_strong_success"]),
                    float(best_eval["worst_base_score"]),
                    float(best_eval["cvar_base_score"]),
                ]
            )

            logger.info(
                jline(
                    "evolution",
                    str(condition["id"]),
                    "generation",
                    generation=int(gen + 1),
                    generations=generations,
                    best_fitness=best_fit,
                    mean_fitness=mean_fit,
                    best_connectivity=float(best_eval["mean_final_connectivity"]),
                    best_cost=float(best_eval["mean_cost"]),
                    worst_case_success=float(best_eval["worst_case_strong_success"]),
                )
            )

            elites = [population[i] for i in order[:elite_count]]
            next_pop = [dict(x) for x in elites]

            while len(next_pop) < pop_size:
                a = tournament(population, scores, int(evo["tournament_size"]), rng)
                b = tournament(population, scores, int(evo["tournament_size"]), rng)
                child = crossover_genome(a, b, library, condition, rng)
                child = mutate_genome(child, library, condition, mutation_rate, rng)
                next_pop.append(child)

            population = next_pop
            progress.step(1)

        best_record["sample_efficiency"] = sample_efficiency(history_rows, cfg)

        logger.info(
            jline(
                "condition",
                str(condition["id"]),
                "heldout_start",
                test_cases=len(test_cases),
            )
        )
        best_record["test"] = evaluate_single(
            executor, cfg, condition, best_record["genome"], test_cases, library
        )

        logger.info(
            jline(
                "condition",
                str(condition["id"]),
                "matched_random_start",
                budget=pop_size * generations,
            )
        )
        best_record["matched_random"] = matched_random_search(
            executor, cfg, condition, train_cases, test_cases, library, rng, logger
        )

        logger.info(
            jline(
                "condition",
                str(condition["id"]),
                "no_scaffold_reference_start",
                test_cases=len(test_cases),
            )
        )
        best_record["no_scaffold_reference"] = evaluate_single(
            executor, cfg, condition, no_scaffold_p18(), test_cases, "p18"
        )

        logger.info(
            jline(
                "condition",
                str(condition["id"]),
                "hand_reference_start",
                test_cases=len(test_cases),
            )
        )
        hand_condition = dict(condition)
        hand_condition["library"] = "p17"
        hand_condition["genome_kind"] = "direct"
        hand_condition["representation"] = "R3"

        if str(condition["task"]) == "T1":
            hand_genome = hand_width_direct()
        else:
            hand_genome = hand_combined_direct()

        best_record["hand_reference"] = _WORKER_EXP17.evaluate_genome_with_modules(
            _WORKER_EXP11,
            _WORKER_EXP15,
            cfg,
            hand_condition,
            hand_genome,
            test_cases,
            "direct",
        )

    logger.info(
        jline(
            "condition",
            str(condition["id"]),
            "global_reference_start",
            test_cases=len(test_cases),
        )
    )
    best_record["global_reference"] = global_reference_eval(cfg, condition, test_cases)

    write_csv(
        str(condition_dir / "fitness_history.csv"),
        history_rows,
        header=[
            "generation",
            "best_fitness",
            "mean_fitness",
            "best_train_connectivity",
            "best_train_cost",
            "best_train_outside",
            "best_train_worst_case_success",
            "best_train_worst_score",
            "best_train_cvar_score",
        ],
    )
    write_json(str(condition_dir / "best_genome.json"), best_record)

    summary = condition_summary(condition, best_record, cfg)
    write_json(str(condition_dir / "summary.json"), summary)

    logger.info(
        jline(
            "condition",
            str(condition["id"]),
            "finish",
            evolved_success=float(summary["evolved_strong_success_rate"]),
            random_success=float(summary["random_strong_success_rate"]),
            gain=float(summary["matched_evolution_gain"]),
            leakage=float(summary["prior_leakage_score"]),
            expressiveness=float(summary["expressiveness_score"]),
            robust=bool(summary["robust_discovery_pass"]),
            mode=str(summary["failure_mode"]),
        )
    )

    return summary


def make_plots(rows: list[dict], run_dir: Path, figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [str(x["group"]) for x in rows]
    evolved = [float(x["evolved_strong_success_rate"]) for x in rows]
    random_ref = [float(x["random_strong_success_rate"]) for x in rows]
    gain = [float(x["matched_evolution_gain"]) for x in rows]
    leakage = [float(x["prior_leakage_score"]) for x in rows]
    expr = [float(x["expressiveness_score"]) for x in rows]

    fig = plt.figure(figsize=(11, 5))
    ax = fig.add_subplot(1, 1, 1)
    x = np.arange(len(labels))
    ax.bar(x - 0.2, evolved, width=0.4, label="evolved")
    ax.bar(x + 0.2, random_ref, width=0.4, label="matched random")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Experiment 18 evolved vs matched random success")
    ax.set_xlabel("condition")
    ax.set_ylabel("strong success rate")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(str(figures_dir / "evolved_vs_matched_random_success.png"), dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(11, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(labels, gain)
    ax.set_title("Experiment 18 matched evolution gain")
    ax.set_xlabel("condition")
    ax.set_ylabel("evolved test fitness - matched random test fitness")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(str(figures_dir / "matched_evolution_gain.png"), dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(11, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(leakage, expr)
    for row in rows:
        ax.text(
            float(row["prior_leakage_score"]),
            float(row["expressiveness_score"]),
            str(row["group"]),
            fontsize=8,
        )
    ax.set_title("Experiment 18 prior leakage vs expressiveness")
    ax.set_xlabel("prior leakage score")
    ax.set_ylabel("expressiveness score")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(str(figures_dir / "leakage_expressiveness_plane.png"), dpi=160)
    plt.close(fig)

    for row in rows:
        path = run_dir / "conditions" / str(row["condition_id"]) / "fitness_history.csv"
        if not path.exists():
            continue

        data = np.genfromtxt(str(path), delimiter=",", names=True)

        if data.size == 0:
            continue

        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(data["generation"], data["best_fitness"], label="best")
        ax.plot(data["generation"], data["mean_fitness"], label="mean")
        ax.set_title(f"Fitness curve {row['condition_id']}")
        ax.set_xlabel("generation")
        ax.set_ylabel("fitness")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(str(figures_dir / f"fitness_{row['condition_id']}.png"), dpi=160)
        plt.close(fig)


def main() -> int:
    global _WORKER_EXP11
    global _WORKER_EXP15
    global _WORKER_EXP17

    root = repo_root()
    _WORKER_EXP11 = load_exp11(root)
    _WORKER_EXP15 = load_exp15(root)
    _WORKER_EXP17 = load_exp17(root)

    config_path = root / "config" / "tests" / "exp_18.yaml"
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
    logger.info(
        jline(
            "run",
            cfg["name"],
            "start",
            run_dir=str(run_dir),
            workers=int(cfg["parallel"]["workers"]),
            backend=str(cfg["parallel"]["backend"]),
        )
    )

    conditions = list(cfg["conditions"])
    total_units = int(cfg["evolution"]["generations"]) * len(conditions)
    progress = Progress(logger=logger, name=cfg["name"], total=total_units)
    progress.start()

    rows = []

    for condition in conditions:
        logger.info(jline("condition", str(condition["id"]), "start"))
        rows.append(evolve_condition(cfg, condition, root, run_dir, logger, progress))

    progress.finish()

    summary_header = [
        "condition_id",
        "group",
        "task",
        "task_distribution",
        "library",
        "ablation",
        "path_type",
        "fitness_family",
        "representation",
        "noise_stress",
        "discovery_generation",
        "sample_efficiency",
        "matched_random_budget",
        "train_fitness",
        "evolved_fitness",
        "random_fitness",
        "no_scaffold_fitness",
        "hand_fitness",
        "global_fitness",
        "generalization_gap",
        "matched_evolution_gain",
        "strong_success_gain_over_random",
        "random_search_equivalent",
        "robust_discovery_pass",
        "primitive_usage_count",
        "primitive_diversity",
        "gap_closure_score",
        "corridor_band_score",
        "bypass_use_score",
        "false_corridor_control_score",
        "prior_leakage_score",
        "expressiveness_score",
        "evolved_mean_connectivity",
        "evolved_mean_stability",
        "evolved_mean_cost",
        "evolved_mean_outside",
        "evolved_mean_false",
        "evolved_mean_path_tpr",
        "evolved_mean_open_precision",
        "evolved_mean_false_open_rate",
        "evolved_mean_complexity",
        "evolved_worst_case_strong_success",
        "evolved_weak_success_rate",
        "evolved_strong_success_rate",
        "evolved_strict_success_rate",
        "random_mean_connectivity",
        "random_mean_stability",
        "random_mean_cost",
        "random_mean_outside",
        "random_mean_false",
        "random_worst_case_strong_success",
        "random_weak_success_rate",
        "random_strong_success_rate",
        "random_strict_success_rate",
        "no_scaffold_strong_success_rate",
        "hand_strong_success_rate",
        "global_strong_success_rate",
        "failure_mode",
    ]

    write_csv(
        str(run_dir / "primitive_redesign_summary.csv"),
        [[x.get(k) for k in summary_header] for x in rows],
        header=summary_header,
    )
    write_json(str(run_dir / "primitive_redesign_summary.json"), rows)

    if bool(cfg["output"]["make_plot"]):
        make_plots(rows, run_dir, figures_dir)

    robust_count = sum(1 for row in rows if bool(row["robust_discovery_pass"]))
    random_equiv_count = sum(1 for row in rows if bool(row["random_search_equivalent"]))

    run_summary = {
        "name": cfg["name"],
        "run_dir": str(run_dir),
        "fingerprint": meta.get("fingerprint"),
        "condition_count": len(conditions),
        "population_size": int(cfg["evolution"]["population_size"]),
        "generations": int(cfg["evolution"]["generations"]),
        "matched_random_budget_per_condition": int(cfg["evolution"]["population_size"])
        * int(cfg["evolution"]["generations"]),
        "workers": int(cfg["parallel"]["workers"]),
        "backend": str(cfg["parallel"]["backend"]),
        "robust_discovery_count": int(robust_count),
        "random_search_equivalence_count": int(random_equiv_count),
        "summary_path": "primitive_redesign_summary.csv",
        "summary_json_path": "primitive_redesign_summary.json",
        "figures": [
            "figures/evolved_vs_matched_random_success.png",
            "figures/matched_evolution_gain.png",
            "figures/leakage_expressiveness_plane.png",
        ],
        "success": True,
    }

    write_json(str(run_dir / "summary.json"), run_summary)

    logger.info(jline("run", cfg["name"], "finish", run_dir=str(run_dir)))
    audit.finish_success()

    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
