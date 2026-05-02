from pathlib import Path
from collections import defaultdict
from statistics import mean, variance
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
        root / "scripts" / "tests" / "exp_11.py", "cibo_exp_11_runtime_for_exp_15"
    )


def init_worker(root_text: str) -> None:
    global _WORKER_EXP11
    _WORKER_EXP11 = load_exp11(Path(root_text))


def genome_keys() -> list[str]:
    return [
        "w_radius",
        "w_prob",
        "w_shape",
        "m_intensity",
        "m_radius",
        "m_decay",
        "d_strength",
        "d_smooth",
        "d_decay",
        "c_branch",
        "c_isolated",
        "c_outside",
        "p_trace",
        "p_decay",
        "p_refresh",
    ]


def apply_representation(g: dict, representation: str) -> dict:
    out = dict(g)

    if representation == "R0":
        return out

    if representation == "R1":
        out["w_radius"] = min(int(out["w_radius"]), 1)
        out["w_prob"] = min(float(out["w_prob"]), 0.35)
        return out

    if representation == "R2":
        out["w_radius"] = 0
        out["w_prob"] = 0.0
        out["w_shape"] = 0
        return out

    if representation == "R3":
        return out

    raise ValueError(f"unknown representation: {representation}")


def random_genome(representation: str, rng: np.random.Generator) -> dict:
    g = {
        "w_radius": int(rng.integers(0, 3)),
        "w_prob": float(rng.random()),
        "w_shape": int(rng.integers(0, 3)),
        "m_intensity": float(rng.random()),
        "m_radius": int(rng.integers(0, 4)),
        "m_decay": float(rng.uniform(0.50, 0.98)),
        "d_strength": float(rng.random()),
        "d_smooth": float(rng.random()),
        "d_decay": float(rng.uniform(0.50, 0.98)),
        "c_branch": float(rng.random()),
        "c_isolated": float(rng.random()),
        "c_outside": float(rng.random()),
        "p_trace": float(rng.random()),
        "p_decay": float(rng.uniform(0.50, 0.98)),
        "p_refresh": float(rng.random()),
    }
    return apply_representation(g, representation)


def mutate_genome(
    g: dict, representation: str, rate: float, rng: np.random.Generator
) -> dict:
    out = dict(g)

    for k in genome_keys():
        if rng.random() >= rate:
            continue

        if k in {"w_radius", "m_radius"}:
            out[k] = int(np.clip(int(out[k]) + int(rng.choice([-1, 1])), 0, 4))
        elif k == "w_shape":
            out[k] = int(rng.integers(0, 3))
        elif k in {"m_decay", "d_decay", "p_decay"}:
            out[k] = float(np.clip(float(out[k]) + rng.normal(0.0, 0.08), 0.50, 0.98))
        else:
            out[k] = float(np.clip(float(out[k]) + rng.normal(0.0, 0.15), 0.0, 1.0))

    return apply_representation(out, representation)


def crossover(a: dict, b: dict, representation: str, rng: np.random.Generator) -> dict:
    g = {k: a[k] if rng.random() < 0.5 else b[k] for k in genome_keys()}
    return apply_representation(g, representation)


def scaffold_complexity(genome: dict) -> float:
    return float(
        abs(float(genome["w_radius"]))
        + abs(float(genome["w_prob"]))
        + abs(float(genome["m_intensity"]))
        + abs(float(genome["m_radius"]))
        + abs(float(genome["d_strength"]))
        + abs(float(genome["c_branch"]))
        + abs(float(genome["c_isolated"]))
        + abs(float(genome["c_outside"]))
        + abs(float(genome["p_trace"]))
    )


def thicken(
    mask: np.ndarray, radius: int, prob: float, shape: int, rng: np.random.Generator
) -> np.ndarray:
    out = mask.copy()

    if radius <= 0 or prob <= 0:
        return out

    for x, y in np.argwhere(mask == 1):
        if rng.random() > prob:
            continue

        x = int(x)
        y = int(y)

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if shape == 1 and abs(dx) + abs(dy) > radius:
                    continue
                if shape == 2 and dx != 0 and dy != 0:
                    continue

                nx = x + dx
                ny = y + dy

                if 0 <= nx < mask.shape[0] and 0 <= ny < mask.shape[1]:
                    out[nx, ny] = 1

    return out


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


def make_marker(
    reference: np.ndarray, cells: list[tuple[int, int]], radius: int, intensity: float
) -> np.ndarray:
    marker = np.zeros_like(reference, dtype=np.float64)

    if radius <= 0 or intensity <= 0:
        return marker

    for x, y in cells:
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx = x + dx
                ny = y + dy

                if 0 <= nx < reference.shape[0] and 0 <= ny < reference.shape[1]:
                    d = abs(dx) + abs(dy)
                    marker[nx, ny] = max(marker[nx, ny], intensity / max(1.0, d + 1.0))

    return marker


def get_cell(canvas: np.ndarray, x: int, y: int) -> int:
    if x < 0 or x >= canvas.shape[0] or y < 0 or y >= canvas.shape[1]:
        return 0
    return int(canvas[x, y])


def make_direction(reference: np.ndarray, strength: float) -> np.ndarray:
    direction = np.zeros((*reference.shape, 2), dtype=np.float64)

    if strength <= 0:
        return direction

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
        direction[x, y, 0] = strength * vx / n
        direction[x, y, 1] = strength * vy / n

    return direction


def make_variant_path_type(path_type: str, variant: str) -> str:
    if variant == "base":
        return path_type

    if path_type == "single_narrow" and variant == "nearby":
        return "thick_corridor"

    if path_type == "double_path" and variant == "nearby":
        return "braided_path"

    return path_type


def make_environment(
    exp11, cfg: dict, condition: dict, genome: dict, seed: int, variant: str
):
    height = int(cfg["canvas"]["height"])
    width = int(cfg["canvas"]["width"])
    path_type = make_variant_path_type(str(condition["path_type"]), variant)
    reference, start, goal = exp11.make_path_environment(height, width, path_type)
    rng = np.random.default_rng(seed * 7919 + stable_seed(variant))

    reference = thicken(
        reference,
        int(genome["w_radius"]),
        float(genome["w_prob"]),
        int(genome["w_shape"]),
        rng,
    )

    bridge_cells = bridge_center_cells(reference, int(condition["cut_length"]))
    marker = make_marker(
        reference, bridge_cells, int(genome["m_radius"]), float(genome["m_intensity"])
    )
    direction = make_direction(reference, float(genome["d_strength"]))

    return reference, start, goal, marker, direction


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


def cost_penalty(
    canvas: np.ndarray, x: int, y: int, radius: int, genome: dict
) -> float:
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
    branch = 1.0 if neighbors >= 3 else 0.0
    bulky = max(0.0, density - 0.50)

    return float(
        min(
            1.0,
            float(genome["c_isolated"]) * isolated
            + float(genome["c_branch"]) * branch
            + float(genome["c_outside"]) * bulky,
        )
    )


def scaffolded_support(
    canvas: np.ndarray,
    trace: np.ndarray,
    marker: np.ndarray,
    direction: np.ndarray,
    genome: dict,
    x: int,
    y: int,
    cfg: dict,
) -> float:
    radius = int(cfg["operator"]["radius"])

    old = int(canvas[x, y])
    if old == 0:
        canvas[x, y] = 1

    local = local_density(canvas, x, y, radius)
    bridge = bridge_evidence(canvas, x, y)
    mark = float(marker[x, y])
    direct = directional_consistency(canvas, direction, x, y)
    cost = cost_penalty(canvas, x, y, radius, genome)

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
        float(cfg["repair"]["w_local"]) * local
        + float(cfg["repair"]["w_bridge"]) * bridge
        + float(cfg["repair"]["w_marker"]) * mark
        + float(cfg["repair"]["w_direction"]) * direct
        + 0.10 * float(genome["p_trace"]) * trace_score
        - float(cfg["repair"]["w_cost"]) * cost
    )

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
    trace *= float(cfg["trace"]["decay"]) * max(0.10, float(genome["p_decay"]))

    for (x, y), values in proposals.items():
        score = sum(values)
        if score > 0:
            trace[x, y] = float(cfg["trace"]["duration"]) * max(
                0.0, float(genome["p_refresh"])
            )
        elif score < 0:
            trace[x, y] = -float(cfg["trace"]["duration"]) * max(
                0.0, float(genome["p_refresh"])
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


def run_trial(
    exp11,
    cfg: dict,
    condition: dict,
    genome: dict,
    seed: int,
    offset: int,
    variant: str,
    noise: bool,
) -> dict:
    height = int(cfg["canvas"]["height"])
    width = int(cfg["canvas"]["width"])
    steps = int(cfg["run"]["steps"])
    rng = np.random.default_rng(seed * 10007 + offset * 101 + stable_seed(variant))

    reference, start, goal, marker, direction = make_environment(
        exp11, cfg, condition, genome, seed, variant
    )
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
                support = scaffolded_support(
                    canvas, trace, marker, direction, genome, x, y, cfg
                )

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
        "variant": str(variant),
        "noise": bool(noise),
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
        "scaffold_complexity": scaffold_complexity(genome),
    }


def make_cases(
    seeds: list[int], offsets: list[int], variants: list[str], noise_values: list[bool]
) -> list[dict]:
    cases = []

    for seed in seeds:
        for offset in offsets:
            for variant in variants:
                for noise in sorted(set(noise_values)):
                    cases.append(
                        {
                            "seed": int(seed),
                            "offset": int(offset),
                            "variant": str(variant),
                            "noise": bool(noise),
                        }
                    )

    return cases


def trial_score(trial: dict, cfg: dict, condition: dict) -> float:
    fam = str(condition["fitness_family"])
    w = cfg["fitness_weights"]

    conn = float(trial["final_connectivity"])
    stability = float(trial["post_damage_connectivity_stability"])
    cost = float(trial["final_open_cost_factor"])
    outside = float(trial["final_outside_open_rate"])
    false = float(trial["final_false_corridor_count"])
    complexity = float(trial["scaffold_complexity"])

    if fam == "F0":
        return float(float(w["connectivity"]) * conn)

    if fam == "F1":
        return float(
            float(w["connectivity"]) * conn
            + float(w["stability"]) * stability
            - float(w["cost"]) * cost
            - float(w["outside"]) * outside
            - float(w["complexity"]) * complexity
        )

    if fam == "F2":
        return float(
            float(w["connectivity"]) * conn
            + float(w["stability"]) * stability
            - float(w["cost"]) * cost
            - float(w["outside"]) * outside
            - float(w["false_corridor"]) * false
            - float(w["complexity"]) * complexity
        )

    if fam == "F3":
        return float(
            float(w["connectivity"]) * conn
            + float(w["stability"]) * stability
            - float(w["cost"]) * cost
            - float(w["outside"]) * outside
            - float(w["false_corridor"]) * false
            - float(w["complexity"]) * complexity
        )

    raise ValueError(f"unknown fitness family: {fam}")


def safe_var(xs: list[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    return float(variance(xs))


def summarize_eval(fitness_value: float, trials: list[dict]) -> dict:
    return {
        "fitness": float(fitness_value),
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
        "trials": trials,
    }


def evaluate_genome_with_exp11(
    exp11, cfg: dict, condition: dict, genome: dict, cases: list[dict]
) -> dict:
    trials = [
        run_trial(
            exp11,
            cfg,
            condition,
            genome,
            int(case["seed"]),
            int(case["offset"]),
            str(case["variant"]),
            bool(case["noise"]),
        )
        for case in cases
    ]
    scores = [trial_score(x, cfg, condition) for x in trials]

    if str(condition["fitness_family"]) == "F3":
        fit = float(
            mean(scores) - float(cfg["fitness_weights"]["variance"]) * safe_var(scores)
        )
    else:
        fit = float(mean(scores))

    return summarize_eval(fit, trials)


def evaluate_worker(args: tuple) -> dict:
    cfg, condition, genome, cases = args
    return evaluate_genome_with_exp11(_WORKER_EXP11, cfg, condition, genome, cases)


def evaluate_population(
    executor: ProcessPoolExecutor,
    cfg: dict,
    condition: dict,
    population: list[dict],
    cases: list[dict],
) -> list[dict]:
    jobs = [(cfg, condition, genome, cases) for genome in population]
    return list(executor.map(evaluate_worker, jobs, chunksize=1))


def evaluate_single(
    executor: ProcessPoolExecutor,
    cfg: dict,
    condition: dict,
    genome: dict,
    cases: list[dict],
) -> dict:
    return evaluate_population(executor, cfg, condition, [genome], cases)[0]


def tournament(
    population: list[dict], scores: list[float], size: int, rng: np.random.Generator
) -> dict:
    idx = rng.choice(len(population), size=size, replace=False)
    best = max(idx, key=lambda i: scores[int(i)])
    return population[int(best)]


def random_search_reference(
    executor: ProcessPoolExecutor,
    cfg: dict,
    condition: dict,
    cases: list[dict],
    rng: np.random.Generator,
) -> dict:
    samples = int(cfg["evolution"]["random_baseline_samples"])
    representation = str(condition["representation"])
    candidates = [random_genome(representation, rng) for _ in range(samples)]
    evaluations = evaluate_population(executor, cfg, condition, candidates, cases)
    best_idx = max(
        range(len(evaluations)), key=lambda i: float(evaluations[i]["fitness"])
    )
    out = evaluations[best_idx]
    out["genome"] = candidates[best_idx]
    return out


def no_scaffold_genome() -> dict:
    return {
        "w_radius": 0,
        "w_prob": 0.0,
        "w_shape": 0,
        "m_intensity": 0.0,
        "m_radius": 0,
        "m_decay": 0.0,
        "d_strength": 0.0,
        "d_smooth": 0.0,
        "d_decay": 0.0,
        "c_branch": 0.0,
        "c_isolated": 0.0,
        "c_outside": 0.0,
        "p_trace": 0.0,
        "p_decay": 0.50,
        "p_refresh": 0.0,
    }


def hand_width_genome() -> dict:
    g = no_scaffold_genome()
    g["w_radius"] = 1
    g["w_prob"] = 1.0
    g["w_shape"] = 0
    g["c_isolated"] = 0.30
    g["p_trace"] = 0.30
    g["p_refresh"] = 0.70
    g["p_decay"] = 0.80
    return g


def hand_combined_genome() -> dict:
    g = hand_width_genome()
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

    scores = [trial_score(x, cfg, condition) for x in trials]
    return summarize_eval(float(mean(scores)), trials)


def success_rates(eval_result: dict, cfg: dict) -> dict:
    trials = eval_result["trials"]
    weak = [1 if int(x["final_connectivity"]) == 1 else 0 for x in trials]
    strong = [
        1
        if int(x["final_connectivity"]) == 1
        and float(x["final_open_cost_factor"])
        <= float(cfg["metrics"]["strong_cost_factor"])
        else 0
        for x in trials
    ]
    strict = [
        1
        if int(x["final_connectivity"]) == 1
        and float(x["final_open_cost_factor"])
        <= float(cfg["metrics"]["strict_cost_factor"])
        and float(x["final_outside_open_rate"])
        <= float(cfg["metrics"]["strict_outside_open_rate"])
        else 0
        for x in trials
    ]

    return {
        "weak_success_rate": float(sum(weak) / max(1, len(weak))),
        "strong_success_rate": float(sum(strong) / max(1, len(strong))),
        "strict_success_rate": float(sum(strict) / max(1, len(strict))),
    }


def eval_to_prefixed(prefix: str, eval_result: dict, cfg: dict) -> dict:
    rates = success_rates(eval_result, cfg)
    return {
        f"{prefix}_fitness": float(eval_result["fitness"]),
        f"{prefix}_mean_connectivity": float(eval_result["mean_final_connectivity"]),
        f"{prefix}_mean_stability": float(eval_result["mean_stability"]),
        f"{prefix}_mean_cost": float(eval_result["mean_cost"]),
        f"{prefix}_mean_outside": float(eval_result["mean_outside"]),
        f"{prefix}_mean_false": float(eval_result["mean_false"]),
        f"{prefix}_mean_path_tpr": float(eval_result["mean_path_tpr"]),
        f"{prefix}_mean_open_precision": float(eval_result["mean_open_precision"]),
        f"{prefix}_mean_false_open_rate": float(eval_result["mean_false_open_rate"]),
        f"{prefix}_mean_complexity": float(eval_result["mean_complexity"]),
        f"{prefix}_weak_success_rate": float(rates["weak_success_rate"]),
        f"{prefix}_strong_success_rate": float(rates["strong_success_rate"]),
        f"{prefix}_strict_success_rate": float(rates["strict_success_rate"]),
    }


def condition_summary(condition: dict, record: dict, cfg: dict) -> dict:
    out = {
        "condition_id": str(condition["id"]),
        "group": str(condition["group"]),
        "task": str(condition["task"]),
        "path_type": str(condition["path_type"]),
        "fitness_family": str(condition["fitness_family"]),
        "representation": str(condition["representation"]),
        "noise_stress": bool(condition.get("noise_stress", False)),
        "discovery_generation": int(record["generation"]),
        "best_genome": record["genome"],
        "scaffold_complexity": float(scaffold_complexity(record["genome"])),
    }

    out.update(eval_to_prefixed("train", record["train"], cfg))
    out.update(eval_to_prefixed("evolved", record["test"], cfg))
    out.update(eval_to_prefixed("random", record["random_reference"], cfg))
    out.update(eval_to_prefixed("no_scaffold", record["no_scaffold_reference"], cfg))
    out.update(eval_to_prefixed("hand", record["hand_reference"], cfg))
    out.update(eval_to_prefixed("global", record["global_reference"], cfg))

    out["generalization_gap"] = float(out["train_fitness"] - out["evolved_fitness"])
    out["evolution_gain_over_random"] = float(
        out["evolved_fitness"] - out["random_fitness"]
    )
    out["evolution_gain_over_no_scaffold"] = float(
        out["evolved_fitness"] - out["no_scaffold_fitness"]
    )
    out["evolution_gain_over_hand"] = float(
        out["evolved_fitness"] - out["hand_fitness"]
    )
    out["discovery_margin_pass"] = bool(
        out["evolution_gain_over_random"] > float(cfg["evolution"]["discovery_margin"])
    )
    out["over_opening_index"] = float(
        max(0.0, out["evolved_mean_cost"] - float(cfg["metrics"]["strong_cost_factor"]))
    )
    out["heldout_robustness"] = float(out["evolved_strong_success_rate"])

    return out


def failure_mode(row: dict, cfg: dict) -> str:
    if float(row["evolution_gain_over_random"]) <= float(
        cfg["evolution"]["discovery_margin"]
    ):
        return "random_search_equivalence"

    if float(row["evolved_mean_connectivity"]) < float(
        cfg["metrics"]["strong_connectivity_threshold"]
    ):
        return "conservative_no_repair"

    if float(row["evolved_mean_cost"]) > float(cfg["metrics"]["strong_cost_factor"]):
        return "trivial_over_opening"

    if float(row["generalization_gap"]) > 0.50:
        return "training_overfit"

    if float(row["scaffold_complexity"]) > 5.0:
        return "scaffold_bloat"

    if float(row["evolved_mean_false"]) > 2.0:
        return "false_corridor_exploit"

    if (
        str(row["task"]) == "T1"
        and str(row["representation"]) == "R1"
        and float(row["evolved_strong_success_rate"]) < 0.90
    ):
        return "representation_driven_discovery"

    if bool(row["noise_stress"]) and float(row["evolved_strong_success_rate"]) < 0.90:
        return "noise_fragility"

    return "robust_nontrivial_discovery"


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
    representation = str(condition["representation"])
    pop_size = int(evo["population_size"])
    generations = int(evo["generations"])
    elite_count = int(evo["elite_count"])
    mutation_rate = float(evo["mutation_rate"])
    workers = int(cfg["parallel"]["workers"])

    train_cases = make_cases(
        [int(x) for x in cfg["run"]["train_seeds"]],
        [int(x) for x in cfg["run"]["train_offsets"]],
        ["base"],
        [False],
    )

    test_cases = make_cases(
        [int(x) for x in cfg["run"]["test_seeds"]],
        [int(x) for x in cfg["run"]["test_offsets"]],
        ["base", "nearby"],
        [False, bool(condition.get("noise_stress", False))],
    )

    condition_dir = run_dir / "conditions" / str(condition["id"])
    condition_dir.mkdir(parents=True, exist_ok=True)

    population = [random_genome(representation, rng) for _ in range(pop_size)]
    history_rows = []
    best_record = None

    logger.info(
        jline(
            "condition",
            str(condition["id"]),
            "cases",
            train_cases=len(train_cases),
            test_cases=len(test_cases),
            workers=workers,
        )
    )

    with ProcessPoolExecutor(
        max_workers=workers, initializer=init_worker, initargs=(str(root),)
    ) as executor:
        for gen in range(generations):
            evaluations = evaluate_population(
                executor, cfg, condition, population, train_cases
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
                    float(best_eval["mean_stability"]),
                    float(best_eval["mean_cost"]),
                    float(best_eval["mean_outside"]),
                    float(best_eval["mean_false"]),
                    float(best_eval["mean_complexity"]),
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
                )
            )

            elites = [population[i] for i in order[:elite_count]]
            next_pop = [dict(x) for x in elites]

            while len(next_pop) < pop_size:
                a = tournament(population, scores, int(evo["tournament_size"]), rng)
                b = tournament(population, scores, int(evo["tournament_size"]), rng)
                child = crossover(a, b, representation, rng)
                child = mutate_genome(child, representation, mutation_rate, rng)
                next_pop.append(child)

            population = next_pop
            progress.step(1)

        logger.info(
            jline(
                "condition",
                str(condition["id"]),
                "heldout_start",
                test_cases=len(test_cases),
            )
        )
        evolved_test = evaluate_single(
            executor, cfg, condition, best_record["genome"], test_cases
        )

        logger.info(
            jline(
                "condition",
                str(condition["id"]),
                "random_reference_start",
                samples=int(cfg["evolution"]["random_baseline_samples"]),
                test_cases=len(test_cases),
            )
        )
        random_ref = random_search_reference(executor, cfg, condition, test_cases, rng)

        logger.info(
            jline(
                "condition",
                str(condition["id"]),
                "no_scaffold_reference_start",
                test_cases=len(test_cases),
            )
        )
        no_ref = evaluate_single(
            executor, cfg, condition, no_scaffold_genome(), test_cases
        )

        logger.info(
            jline(
                "condition",
                str(condition["id"]),
                "hand_reference_start",
                test_cases=len(test_cases),
            )
        )
        if str(condition["task"]) == "T1":
            hand_ref = evaluate_single(
                executor, cfg, condition, hand_width_genome(), test_cases
            )
        else:
            hand_ref = evaluate_single(
                executor, cfg, condition, hand_combined_genome(), test_cases
            )

    logger.info(
        jline(
            "condition",
            str(condition["id"]),
            "global_reference_start",
            test_cases=len(test_cases),
        )
    )
    global_ref = global_reference_eval(cfg, condition, test_cases)

    best_record["test"] = evolved_test
    best_record["random_reference"] = random_ref
    best_record["no_scaffold_reference"] = no_ref
    best_record["hand_reference"] = hand_ref
    best_record["global_reference"] = global_ref

    write_csv(
        str(condition_dir / "fitness_history.csv"),
        history_rows,
        header=[
            "generation",
            "best_fitness",
            "mean_fitness",
            "best_train_connectivity",
            "best_train_stability",
            "best_train_cost",
            "best_train_outside",
            "best_train_false",
            "best_complexity",
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
            gain=float(summary["evolution_gain_over_random"]),
            mode=failure_mode(summary, cfg),
        )
    )

    return summary


def make_plots(rows: list[dict], run_dir: Path, figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [str(x["group"]) for x in rows]
    evolved = [float(x["evolved_strong_success_rate"]) for x in rows]
    random_ref = [float(x["random_strong_success_rate"]) for x in rows]
    gain = [float(x["evolution_gain_over_random"]) for x in rows]
    gap = [float(x["generalization_gap"]) for x in rows]

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(labels, evolved)
    ax.set_title("Experiment 15 Lite evolved strong success rate")
    ax.set_xlabel("condition")
    ax.set_ylabel("strong success rate")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(str(figures_dir / "evolved_strong_success_rate.png"), dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(1, 1, 1)
    x = np.arange(len(labels))
    ax.bar(x - 0.2, evolved, width=0.4, label="evolved")
    ax.bar(x + 0.2, random_ref, width=0.4, label="random")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Experiment 15 Lite evolved vs random")
    ax.set_xlabel("condition")
    ax.set_ylabel("strong success rate")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(str(figures_dir / "evolved_vs_random_success.png"), dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(labels, gain)
    ax.set_title("Experiment 15 Lite evolution gain over random")
    ax.set_xlabel("condition")
    ax.set_ylabel("fitness gain")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(str(figures_dir / "evolution_gain_over_random.png"), dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(labels, gap)
    ax.set_title("Experiment 15 Lite generalization gap")
    ax.set_xlabel("condition")
    ax.set_ylabel("train fitness - test fitness")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(str(figures_dir / "generalization_gap.png"), dpi=160)
    plt.close(fig)

    for row in rows:
        path = run_dir / "conditions" / str(row["condition_id"]) / "fitness_history.csv"
        if not path.exists():
            continue

        data = np.genfromtxt(str(path), delimiter=",", names=True)
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
    root = repo_root()
    config_path = root / "config" / "tests" / "exp_15.yaml"
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

    for row in rows:
        row["failure_mode"] = failure_mode(row, cfg)

    summary_header = [
        "condition_id",
        "group",
        "task",
        "path_type",
        "fitness_family",
        "representation",
        "noise_stress",
        "discovery_generation",
        "train_fitness",
        "evolved_fitness",
        "random_fitness",
        "no_scaffold_fitness",
        "hand_fitness",
        "global_fitness",
        "generalization_gap",
        "evolution_gain_over_random",
        "evolution_gain_over_no_scaffold",
        "evolution_gain_over_hand",
        "discovery_margin_pass",
        "over_opening_index",
        "heldout_robustness",
        "evolved_mean_connectivity",
        "evolved_mean_stability",
        "evolved_mean_cost",
        "evolved_mean_outside",
        "evolved_mean_false",
        "evolved_mean_path_tpr",
        "evolved_mean_open_precision",
        "evolved_mean_false_open_rate",
        "evolved_mean_complexity",
        "evolved_weak_success_rate",
        "evolved_strong_success_rate",
        "evolved_strict_success_rate",
        "random_weak_success_rate",
        "random_strong_success_rate",
        "random_strict_success_rate",
        "no_scaffold_strong_success_rate",
        "hand_strong_success_rate",
        "global_strong_success_rate",
        "scaffold_complexity",
        "failure_mode",
    ]

    write_csv(
        str(run_dir / "fitness_generalization_summary.csv"),
        [[x.get(k) for k in summary_header] for x in rows],
        header=summary_header,
    )
    write_json(str(run_dir / "fitness_generalization_summary.json"), rows)

    if bool(cfg["output"]["make_plot"]):
        make_plots(rows, run_dir, figures_dir)

    run_summary = {
        "name": cfg["name"],
        "run_dir": str(run_dir),
        "fingerprint": meta.get("fingerprint"),
        "condition_count": len(conditions),
        "population_size": int(cfg["evolution"]["population_size"]),
        "generations": int(cfg["evolution"]["generations"]),
        "workers": int(cfg["parallel"]["workers"]),
        "backend": str(cfg["parallel"]["backend"]),
        "train_seed_count": len(cfg["run"]["train_seeds"]),
        "test_seed_count": len(cfg["run"]["test_seeds"]),
        "train_offset_count": len(cfg["run"]["train_offsets"]),
        "test_offset_count": len(cfg["run"]["test_offsets"]),
        "summary_path": "fitness_generalization_summary.csv",
        "summary_json_path": "fitness_generalization_summary.json",
        "figures": [
            "figures/evolved_strong_success_rate.png",
            "figures/evolved_vs_random_success.png",
            "figures/evolution_gain_over_random.png",
            "figures/generalization_gap.png",
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
