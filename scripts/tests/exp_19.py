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
_WORKER_EXP18 = None


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
        root / "scripts" / "tests" / "exp_11.py",
        "cibo_exp_11_runtime_for_exp_19",
    )


def load_exp15(root: Path):
    return load_module(
        root / "scripts" / "tests" / "exp_15.py",
        "cibo_exp_15_runtime_for_exp_19",
    )


def load_exp18(root: Path):
    return load_module(
        root / "scripts" / "tests" / "exp_18.py",
        "cibo_exp_18_runtime_for_exp_19",
    )


def init_worker(root_text: str) -> None:
    global _WORKER_EXP11
    global _WORKER_EXP15
    global _WORKER_EXP18
    root = Path(root_text)
    _WORKER_EXP11 = load_exp11(root)
    _WORKER_EXP15 = load_exp15(root)
    _WORKER_EXP18 = load_exp18(root)


def grammar_ops() -> list[str]:
    return [
        "detect_gap",
        "detect_two",
        "detect_frontier",
        "detect_deadend",
        "grow_mask",
        "grow_toward_support",
        "grow_local_band",
        "grow_short_bypass",
        "add_trace",
        "decay_trace",
        "refresh_supported",
        "prune_deadend",
        "prune_unsupported",
        "apply_budget",
    ]


def grammar_conditions() -> list[str]:
    return [
        "always",
        "has_gap",
        "has_two",
        "near_frontier",
        "over_budget",
        "has_mask",
    ]


def random_instruction(cfg: dict, rng: np.random.Generator) -> dict:
    return {
        "op": str(rng.choice(grammar_ops())),
        "condition": str(rng.choice(grammar_conditions())),
        "strength": float(rng.uniform(0.20, 1.00)),
        "threshold": float(rng.uniform(0.20, 0.80)),
        "radius": int(rng.integers(1, int(cfg["grammar"]["max_radius"]) + 1)),
    }


def random_grammar_genome(cfg: dict, rng: np.random.Generator) -> dict:
    n = int(
        rng.integers(
            int(cfg["grammar"]["min_program_length"]),
            int(cfg["grammar"]["max_program_length"]) + 1,
        )
    )
    return {
        "program": [random_instruction(cfg, rng) for _ in range(n)],
        "mask_decay": float(
            rng.uniform(
                float(cfg["grammar"]["mask_decay_min"]),
                float(cfg["grammar"]["mask_decay_max"]),
            )
        ),
        "trace_decay": float(
            rng.uniform(
                float(cfg["grammar"]["trace_decay_min"]),
                float(cfg["grammar"]["trace_decay_max"]),
            )
        ),
        "priority_decay": float(
            rng.uniform(
                float(cfg["grammar"]["priority_decay_min"]),
                float(cfg["grammar"]["priority_decay_max"]),
            )
        ),
        "trace_refresh": float(
            rng.uniform(
                float(cfg["grammar"]["trace_refresh_min"]),
                float(cfg["grammar"]["trace_refresh_max"]),
            )
        ),
        "budget_limit": float(
            rng.uniform(
                float(cfg["grammar"]["budget_limit_min"]),
                float(cfg["grammar"]["budget_limit_max"]),
            )
        ),
        "evidence_gain": float(
            rng.uniform(
                float(cfg["grammar"]["evidence_gain_min"]),
                float(cfg["grammar"]["evidence_gain_max"]),
            )
        ),
    }


def random_genome(cfg: dict, condition: dict, rng: np.random.Generator) -> dict:
    library = str(condition["library"])

    if library == "grammar":
        return random_grammar_genome(cfg, rng)

    if library == "p18":
        return _WORKER_EXP18.random_genome("p18", condition, rng)

    raise ValueError(f"unknown library: {library}")


def mutate_instruction(item: dict, cfg: dict, rng: np.random.Generator) -> dict:
    out = dict(item)
    field = str(rng.choice(["op", "condition", "strength", "threshold", "radius"]))

    if field == "op":
        out["op"] = str(rng.choice(grammar_ops()))
    elif field == "condition":
        out["condition"] = str(rng.choice(grammar_conditions()))
    elif field == "strength":
        out["strength"] = float(
            np.clip(float(out["strength"]) + rng.normal(0.0, 0.15), 0.0, 1.5)
        )
    elif field == "threshold":
        out["threshold"] = float(
            np.clip(float(out["threshold"]) + rng.normal(0.0, 0.08), 0.05, 0.95)
        )
    elif field == "radius":
        out["radius"] = int(
            np.clip(
                int(out["radius"]) + int(rng.choice([-1, 1])),
                1,
                int(cfg["grammar"]["max_radius"]),
            )
        )
    else:
        raise ValueError(f"unknown instruction field: {field}")

    return out


def mutate_grammar_genome(
    g: dict,
    cfg: dict,
    rate: float,
    rng: np.random.Generator,
) -> dict:
    out = {
        "program": [dict(x) for x in g["program"]],
        "mask_decay": float(g["mask_decay"]),
        "trace_decay": float(g["trace_decay"]),
        "priority_decay": float(g["priority_decay"]),
        "trace_refresh": float(g["trace_refresh"]),
        "budget_limit": float(g["budget_limit"]),
        "evidence_gain": float(g["evidence_gain"]),
    }

    for i in range(len(out["program"])):
        if rng.random() < rate:
            out["program"][i] = mutate_instruction(out["program"][i], cfg, rng)

    if rng.random() < rate:
        if len(out["program"]) < int(cfg["grammar"]["max_program_length"]):
            pos = int(rng.integers(0, len(out["program"]) + 1))
            out["program"].insert(pos, random_instruction(cfg, rng))

    if rng.random() < rate:
        if len(out["program"]) > int(cfg["grammar"]["min_program_length"]):
            pos = int(rng.integers(0, len(out["program"])))
            out["program"].pop(pos)

    for key in ["mask_decay", "trace_decay", "priority_decay"]:
        if rng.random() < rate:
            out[key] = float(
                np.clip(float(out[key]) + rng.normal(0.0, 0.08), 0.40, 0.99)
            )

    for key in ["trace_refresh", "evidence_gain"]:
        if rng.random() < rate:
            out[key] = float(
                np.clip(float(out[key]) + rng.normal(0.0, 0.15), 0.0, 1.80)
            )

    if rng.random() < rate:
        out["budget_limit"] = float(
            np.clip(float(out["budget_limit"]) + rng.normal(0.0, 0.12), 0.80, 2.40)
        )

    return out


def mutate_genome(
    g: dict,
    cfg: dict,
    condition: dict,
    rate: float,
    rng: np.random.Generator,
) -> dict:
    library = str(condition["library"])

    if library == "grammar":
        return mutate_grammar_genome(g, cfg, rate, rng)

    if library == "p18":
        return _WORKER_EXP18.mutate_genome(g, "p18", condition, rate, rng)

    raise ValueError(f"unknown library: {library}")


def crossover_grammar(
    a: dict,
    b: dict,
    cfg: dict,
    rng: np.random.Generator,
) -> dict:
    pa = [dict(x) for x in a["program"]]
    pb = [dict(x) for x in b["program"]]
    ca = int(rng.integers(0, len(pa) + 1))
    cb = int(rng.integers(0, len(pb) + 1))
    program = pa[:ca] + pb[cb:]

    while len(program) < int(cfg["grammar"]["min_program_length"]):
        program.append(random_instruction(cfg, rng))

    if len(program) > int(cfg["grammar"]["max_program_length"]):
        program = program[: int(cfg["grammar"]["max_program_length"])]

    return {
        "program": program,
        "mask_decay": float(a["mask_decay"] if rng.random() < 0.5 else b["mask_decay"]),
        "trace_decay": float(
            a["trace_decay"] if rng.random() < 0.5 else b["trace_decay"]
        ),
        "priority_decay": float(
            a["priority_decay"] if rng.random() < 0.5 else b["priority_decay"]
        ),
        "trace_refresh": float(
            a["trace_refresh"] if rng.random() < 0.5 else b["trace_refresh"]
        ),
        "budget_limit": float(
            a["budget_limit"] if rng.random() < 0.5 else b["budget_limit"]
        ),
        "evidence_gain": float(
            a["evidence_gain"] if rng.random() < 0.5 else b["evidence_gain"]
        ),
    }


def crossover_genome(
    a: dict,
    b: dict,
    cfg: dict,
    condition: dict,
    rng: np.random.Generator,
) -> dict:
    library = str(condition["library"])

    if library == "grammar":
        return crossover_grammar(a, b, cfg, rng)

    if library == "p18":
        return _WORKER_EXP18.crossover_genome(a, b, "p18", condition, rng)

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
                    if task == "T1":
                        path_types = ["single_narrow"]
                    elif task == "T2":
                        path_types = ["double_path"]
                    elif task == "mixed":
                        path_types = ["single_narrow", "double_path"]
                    else:
                        raise ValueError(f"unknown task: {task}")

                    for path_type in path_types:
                        cases.append(
                            {
                                "seed": int(seed),
                                "offset": int(offset),
                                "variant": str(variant),
                                "noise": bool(noise),
                                "path_type": str(path_type),
                            }
                        )

    return cases


def make_variant_path_type(path_type: str, variant: str) -> str:
    if path_type == "single_narrow" and variant == "base":
        return "single_narrow"

    if path_type == "single_narrow" and variant == "nearby":
        return "thick_corridor"

    if path_type == "double_path" and variant == "base":
        return "double_path"

    if path_type == "double_path" and variant == "nearby":
        return "braided_path"

    raise ValueError(f"unknown path_type/variant pair: {path_type}/{variant}")


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


def gap_map(canvas: np.ndarray) -> np.ndarray:
    out = np.zeros_like(canvas, dtype=np.float64)

    for x in range(canvas.shape[0]):
        for y in range(canvas.shape[1]):
            if int(canvas[x, y]) == 1:
                continue

            before = local_components(local_window(canvas, x, y, 1))
            canvas[x, y] = 1
            after = local_components(local_window(canvas, x, y, 1))
            canvas[x, y] = 0

            if before >= 2 and after < before:
                out[x, y] = 1.0
            elif before >= 2:
                out[x, y] = 0.5

    return out


def two_sided_map(canvas: np.ndarray) -> np.ndarray:
    out = np.zeros_like(canvas, dtype=np.float64)

    for x in range(canvas.shape[0]):
        for y in range(canvas.shape[1]):
            left = get_cell(canvas, x, y - 1)
            right = get_cell(canvas, x, y + 1)
            up = get_cell(canvas, x - 1, y)
            down = get_cell(canvas, x + 1, y)
            horizontal = 1.0 if left and right else 0.0
            vertical = 1.0 if up and down else 0.0
            corner = 0.5 if (left or right) and (up or down) else 0.0
            out[x, y] = float(min(1.0, horizontal + vertical + corner))

    return out


def frontier_map(canvas: np.ndarray) -> np.ndarray:
    out = np.zeros_like(canvas, dtype=np.float64)

    for x in range(canvas.shape[0]):
        for y in range(canvas.shape[1]):
            if int(canvas[x, y]) == 1:
                continue

            n = (
                get_cell(canvas, x - 1, y)
                + get_cell(canvas, x + 1, y)
                + get_cell(canvas, x, y - 1)
                + get_cell(canvas, x, y + 1)
            )

            if n >= 2:
                out[x, y] = 1.0
            elif n == 1:
                out[x, y] = 0.5

    return out


def deadend_map(canvas: np.ndarray) -> np.ndarray:
    out = np.zeros_like(canvas, dtype=np.float64)

    for x in range(canvas.shape[0]):
        for y in range(canvas.shape[1]):
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
                out[x, y] = 1.0
            elif degree >= 3:
                out[x, y] = 0.25

    return out


def band_map(canvas: np.ndarray) -> np.ndarray:
    out = np.zeros_like(canvas, dtype=np.float64)

    for x in range(canvas.shape[0]):
        for y in range(canvas.shape[1]):
            density = local_density(canvas, x, y, 1)
            align = alignment_score(canvas, x, y)
            out[x, y] = float(0.5 * density + 0.5 * align)

    return out


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


def bypass_map(canvas: np.ndarray) -> np.ndarray:
    out = np.zeros_like(canvas, dtype=np.float64)

    for x in range(canvas.shape[0]):
        for y in range(canvas.shape[1]):
            w = local_window(canvas, x, y, 2).copy()

            if w.shape[0] < 3 or w.shape[1] < 3:
                continue

            opens = [
                (i, j)
                for i in range(w.shape[0])
                for j in range(w.shape[1])
                if int(w[i, j]) == 1
            ]

            if len(opens) < 2:
                continue

            cx = min(2, w.shape[0] - 1)
            cy = min(2, w.shape[1] - 1)
            best = 0.0

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
                        best = max(best, 1.0)
                    elif before is not None and after is not None and after < before:
                        best = max(
                            best, min(1.0, float((before - after) / max(1, before)))
                        )

            out[x, y] = float(best)

    return out


def dilate_field(field: np.ndarray, radius: int) -> np.ndarray:
    out = field.copy()

    for x in range(field.shape[0]):
        for y in range(field.shape[1]):
            value = float(field[x, y])

            if value <= 0:
                continue

            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) + abs(dy) > radius:
                        continue

                    nx = x + dx
                    ny = y + dy

                    if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]:
                        out[nx, ny] = max(float(out[nx, ny]), value)

    return out


def build_maps(canvas: np.ndarray) -> dict:
    return {
        "gap": gap_map(canvas),
        "two": two_sided_map(canvas),
        "frontier": frontier_map(canvas),
        "deadend": deadend_map(canvas),
        "band": band_map(canvas),
        "bypass": bypass_map(canvas),
    }


def condition_pass(
    condition_name: str,
    maps: dict,
    mask: np.ndarray,
    canvas: np.ndarray,
    initial_open: int,
    genome: dict,
) -> bool:
    if condition_name == "always":
        return True

    if condition_name == "has_gap":
        return float(maps["gap"].max()) > 0.0

    if condition_name == "has_two":
        return float(maps["two"].max()) > 0.0

    if condition_name == "near_frontier":
        return float(maps["frontier"].max()) > 0.0

    if condition_name == "over_budget":
        factor = float((canvas.sum() + mask.sum()) / max(1, initial_open))
        return factor > float(genome["budget_limit"])

    if condition_name == "has_mask":
        return float(mask.max()) > 0.0

    raise ValueError(f"unknown grammar condition: {condition_name}")


def execute_program(
    canvas: np.ndarray,
    mask: np.ndarray,
    program_trace: np.ndarray,
    priority: np.ndarray,
    budget: np.ndarray,
    genome: dict,
    initial_open: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    maps = build_maps(canvas)
    mask = mask * float(genome["mask_decay"])
    program_trace = program_trace * float(genome["trace_decay"])
    priority = priority * float(genome["priority_decay"])
    budget = budget * 0.90
    growth_total = 0.0
    trace_total = 0.0
    prune_total = 0.0

    for item in genome["program"]:
        op = str(item["op"])
        condition_name = str(item["condition"])
        strength = float(item["strength"])
        threshold = float(item["threshold"])
        radius = int(item["radius"])

        if not condition_pass(condition_name, maps, mask, canvas, initial_open, genome):
            continue

        if op == "detect_gap":
            priority = np.maximum(priority, strength * maps["gap"])
        elif op == "detect_two":
            priority = np.maximum(priority, strength * maps["two"])
        elif op == "detect_frontier":
            priority = np.maximum(priority, strength * maps["frontier"])
        elif op == "detect_deadend":
            budget = np.maximum(budget, strength * maps["deadend"])
        elif op == "grow_mask":
            seed = np.maximum.reduce(
                [maps["gap"], maps["two"], maps["frontier"], priority]
            )
            grown = dilate_field(seed, radius)
            mask = np.maximum(mask, strength * grown)
            growth_total += float(grown.sum())
        elif op == "grow_toward_support":
            seed = np.maximum(maps["gap"], maps["frontier"])
            grown = dilate_field(seed, radius)
            mask = np.maximum(mask, strength * grown * maps["two"])
            priority = np.maximum(priority, strength * grown)
            growth_total += float(grown.sum())
        elif op == "grow_local_band":
            seed = (maps["band"] >= threshold).astype(np.float64)
            grown = dilate_field(seed, radius)
            mask = np.maximum(mask, strength * grown)
            priority = np.maximum(priority, strength * maps["band"])
            growth_total += float(grown.sum())
        elif op == "grow_short_bypass":
            grown = dilate_field(maps["bypass"], radius)
            mask = np.maximum(mask, strength * grown)
            priority = np.maximum(priority, strength * maps["bypass"])
            growth_total += float(grown.sum())
        elif op == "add_trace":
            add = np.maximum(mask, priority)
            program_trace = np.maximum(
                program_trace, float(genome["trace_refresh"]) * strength * add
            )
            trace_total += float(add.sum())
        elif op == "decay_trace":
            program_trace *= max(0.0, 1.0 - 0.25 * strength)
        elif op == "refresh_supported":
            supported = np.maximum.reduce(
                [maps["gap"], maps["two"], maps["frontier"], maps["band"]]
            )
            program_trace = np.maximum(
                program_trace, float(genome["trace_refresh"]) * strength * supported
            )
            trace_total += float(supported.sum())
        elif op == "prune_deadend":
            penalty = np.clip(strength * maps["deadend"], 0.0, 1.0)
            mask *= 1.0 - penalty
            priority *= 1.0 - 0.5 * penalty
            budget = np.maximum(budget, penalty)
            prune_total += float(penalty.sum())
        elif op == "prune_unsupported":
            support = np.maximum.reduce(
                [maps["gap"], maps["two"], maps["frontier"], maps["band"]]
            )
            penalty = np.clip(
                strength * (support < threshold).astype(np.float64), 0.0, 1.0
            )
            mask *= 1.0 - penalty
            priority *= 1.0 - 0.5 * penalty
            budget = np.maximum(budget, penalty)
            prune_total += float(penalty.sum())
        elif op == "apply_budget":
            factor = float((canvas.sum() + mask.sum()) / max(1, initial_open))
            if factor > float(genome["budget_limit"]):
                budget = np.maximum(
                    budget, strength * min(1.0, factor - float(genome["budget_limit"]))
                )
        else:
            raise ValueError(f"unknown grammar op: {op}")

    info = {
        "growth_total": float(growth_total),
        "trace_total": float(trace_total),
        "prune_total": float(prune_total),
    }

    return (
        np.clip(mask, 0.0, 1.0),
        np.clip(program_trace, -1.0, 1.0),
        np.clip(priority, 0.0, 1.0),
        np.clip(budget, 0.0, 1.0),
        info,
    )


def grammar_complexity(genome: dict) -> float:
    program = list(genome["program"])
    length = len(program)
    diversity = len(set(str(x["op"]) for x in program))
    conditional = sum(1 for x in program if str(x["condition"]) != "always")
    growth = sum(1 for x in program if str(x["op"]).startswith("grow"))
    stabilize = sum(
        1
        for x in program
        if str(x["op"]) in {"add_trace", "decay_trace", "refresh_supported"}
    )
    return float(
        length
        + 0.25 * diversity
        + 0.15 * conditional
        + 0.10 * growth
        + 0.10 * stabilize
    )


def program_stats(genome: dict, library: str) -> dict:
    if library == "p18":
        return {
            "program_length": None,
            "instruction_diversity": None,
            "detect_count": None,
            "grow_count": None,
            "stabilize_count": None,
            "prune_count": None,
            "program_interpretability": None,
            "prior_leakage_score": 0.65,
        }

    if library != "grammar":
        raise ValueError(f"unknown library for program_stats: {library}")

    program = list(genome["program"])
    ops = [str(x["op"]) for x in program]
    detect = sum(1 for x in ops if x.startswith("detect"))
    grow = sum(1 for x in ops if x.startswith("grow"))
    stabilize = sum(
        1 for x in ops if x in {"add_trace", "decay_trace", "refresh_supported"}
    )
    prune = sum(1 for x in ops if x.startswith("prune") or x == "apply_budget")
    diversity = len(set(ops))
    length = len(ops)
    balance = min(detect, grow, stabilize + 1, prune + 1) / max(
        1, max(detect, grow, stabilize + 1, prune + 1)
    )
    bloat_penalty = max(0, length - 6) * 0.08
    interpretability = float(
        np.clip(
            0.35 + 0.45 * balance + 0.20 * min(1.0, diversity / 6.0) - bloat_penalty,
            0.0,
            1.0,
        )
    )
    leakage_ops = {
        "detect_gap",
        "grow_toward_support",
        "grow_local_band",
        "grow_short_bypass",
    }
    leakage = sum(1 for x in ops if x in leakage_ops) / max(1, length)
    prior_leakage = float(
        np.clip(0.20 + 0.65 * leakage + 0.15 * max(0, length - 6) / 4.0, 0.0, 1.0)
    )

    return {
        "program_length": int(length),
        "instruction_diversity": int(diversity),
        "detect_count": int(detect),
        "grow_count": int(grow),
        "stabilize_count": int(stabilize),
        "prune_count": int(prune),
        "program_interpretability": interpretability,
        "prior_leakage_score": prior_leakage,
    }


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


def run_grammar_trial(
    exp11, cfg: dict, condition: dict, genome: dict, case: dict
) -> dict:
    height = int(cfg["canvas"]["height"])
    width = int(cfg["canvas"]["width"])
    steps = int(cfg["run"]["steps"])
    seed = int(case["seed"])
    offset = int(case["offset"])
    variant = str(case["variant"])
    noise = bool(case["noise"])
    path_type = make_variant_path_type(str(case["path_type"]), variant)
    rng = np.random.default_rng(seed * 10007 + offset * 101 + stable_seed(variant))

    reference, start, goal = exp11.make_path_environment(height, width, path_type)
    canvas = reference.copy()
    mask = np.zeros((height, width), dtype=np.float64)
    program_trace = np.zeros((height, width), dtype=np.float64)
    priority = np.zeros((height, width), dtype=np.float64)
    budget = np.zeros((height, width), dtype=np.float64)
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
    growth_total = 0.0
    trace_total = 0.0
    prune_total = 0.0

    for t in range(steps + 1):
        if t == int(condition["damage_time"]):
            bridge_cut(canvas, reference, condition, offset=offset)

        mask, program_trace, priority, budget, info = execute_program(
            canvas,
            mask,
            program_trace,
            priority,
            budget,
            genome,
            initial_open,
        )

        growth_total += float(info["growth_total"])
        trace_total += float(info["trace_total"])
        prune_total += float(info["prune_total"])

        proposals = defaultdict(list)
        proposal_meta = defaultdict(list)

        if t < steps:
            move_operators(operators, height, width, rng)

            for pos in operators:
                x = int(pos[0])
                y = int(pos[1])
                local = local_density(canvas, x, y, int(cfg["operator"]["radius"]))
                support = float(
                    float(genome["evidence_gain"])
                    * (
                        float(cfg["repair"]["w_mask"]) * float(mask[x, y])
                        + float(cfg["repair"]["w_trace"])
                        * float(max(0.0, program_trace[x, y]))
                        + float(cfg["repair"]["w_priority"]) * float(priority[x, y])
                        + float(cfg["repair"]["w_local"]) * local
                        - float(cfg["repair"]["w_budget"]) * float(budget[x, y])
                    )
                )
                support = float(np.clip(support, 0.0, 1.0))

                if noise:
                    support = float(
                        np.clip(
                            support + rng.normal(0.0, float(condition["sigma"])),
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
        "path_type": str(case["path_type"]),
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
        "scaffold_complexity": grammar_complexity(genome),
        "scaffold_growth_amount": float(growth_total),
        "trace_stabilization_amount": float(trace_total),
        "prune_amount": float(prune_total),
    }


def run_p18_trial(
    exp11, exp18, cfg: dict, condition: dict, genome: dict, case: dict
) -> dict:
    c = dict(condition)
    c["ablation"] = "none"
    trial = exp18.run_p18_trial(exp11, cfg, c, genome, case)
    trial["scaffold_growth_amount"] = 0.0
    trial["trace_stabilization_amount"] = 0.0
    trial["prune_amount"] = 0.0
    return trial


def run_trial_by_library(
    exp11, exp18, cfg: dict, condition: dict, genome: dict, case: dict
) -> dict:
    library = str(condition["library"])

    if library == "grammar":
        return run_grammar_trial(exp11, cfg, condition, genome, case)

    if library == "p18":
        return run_p18_trial(exp11, exp18, cfg, condition, genome, case)

    raise ValueError(f"unknown library: {library}")


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


def run_direct_hand_trial(exp11, exp15, cfg: dict, condition: dict, case: dict) -> dict:
    path_type = str(case["path_type"])

    if path_type == "single_narrow":
        genome = hand_width_direct()
    elif path_type == "double_path":
        genome = hand_combined_direct()
    else:
        raise ValueError(f"unknown hand reference path_type: {path_type}")

    c = dict(condition)
    c["path_type"] = path_type
    trial = exp15.run_trial(
        exp11,
        cfg,
        c,
        genome,
        int(case["seed"]),
        int(case["offset"]),
        str(case["variant"]),
        bool(case["noise"]),
    )
    trial["path_type"] = path_type
    trial["scaffold_growth_amount"] = 0.0
    trial["trace_stabilization_amount"] = 0.0
    trial["prune_amount"] = 0.0
    return trial


def no_scaffold_genome() -> dict:
    return {
        "program": [
            {
                "op": "decay_trace",
                "condition": "always",
                "strength": 1.0,
                "threshold": 0.5,
                "radius": 1,
            },
            {
                "op": "apply_budget",
                "condition": "always",
                "strength": 0.0,
                "threshold": 0.5,
                "radius": 1,
            },
        ],
        "mask_decay": 0.50,
        "trace_decay": 0.50,
        "priority_decay": 0.50,
        "trace_refresh": 0.0,
        "budget_limit": 1.05,
        "evidence_gain": 0.0,
    }


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
        "mean_scaffold_growth": float(
            mean([float(x["scaffold_growth_amount"]) for x in trials])
        ),
        "mean_trace_stabilization": float(
            mean([float(x["trace_stabilization_amount"]) for x in trials])
        ),
        "mean_prune_amount": float(mean([float(x["prune_amount"]) for x in trials])),
        "worst_case_strong_success": float(worst_subgroup_success(trials, cfg)),
        "trials": trials,
    }


def evaluate_genome_with_modules(
    exp11, exp15, exp18, cfg: dict, condition: dict, genome: dict, cases: list[dict]
) -> dict:
    trials = [
        run_trial_by_library(exp11, exp18, cfg, condition, genome, case)
        for case in cases
    ]
    scores = [base_score(trial, cfg) for trial in trials]
    fitness_value = aggregate_fitness(
        scores,
        str(condition["fitness_family"]),
        float(cfg["evolution"]["cvar_alpha"]),
    )
    return summarize_eval(fitness_value, scores, trials, cfg)


def evaluate_worker(args: tuple) -> dict:
    cfg, condition, genome, cases = args
    return evaluate_genome_with_modules(
        _WORKER_EXP11,
        _WORKER_EXP15,
        _WORKER_EXP18,
        cfg,
        condition,
        genome,
        cases,
    )


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


def evaluate_no_scaffold(
    executor: ProcessPoolExecutor, cfg: dict, condition: dict, cases: list[dict]
) -> dict:
    c = dict(condition)
    c["library"] = "grammar"
    c["representation"] = "G0"
    return evaluate_single(executor, cfg, c, no_scaffold_genome(), cases)


def evaluate_hand_reference(
    exp11, exp15, cfg: dict, condition: dict, cases: list[dict]
) -> dict:
    trials = [
        run_direct_hand_trial(exp11, exp15, cfg, condition, case) for case in cases
    ]
    scores = [base_score(trial, cfg) for trial in trials]
    return summarize_eval(float(mean(scores)), scores, trials, cfg)


def global_reference_eval(cfg: dict, condition: dict, cases: list[dict]) -> dict:
    trials = []

    for case in cases:
        trials.append(
            {
                "seed": int(case["seed"]),
                "offset": int(case["offset"]),
                "variant": str(case["variant"]),
                "noise": bool(case["noise"]),
                "path_type": str(case["path_type"]),
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
                "scaffold_growth_amount": 0.0,
                "trace_stabilization_amount": 0.0,
                "prune_amount": 0.0,
            }
        )

    scores = [base_score(x, cfg) for x in trials]
    return summarize_eval(float(mean(scores)), scores, trials, cfg)


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
        candidates = [random_genome(cfg, condition, rng) for _ in range(n)]
        evaluations = evaluate_population(
            executor, cfg, condition, candidates, train_cases
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

    test_eval = evaluate_single(executor, cfg, condition, best_genome, test_cases)

    return {
        "budget": int(budget),
        "genome": best_genome,
        "train": best_train,
        "test": test_eval,
    }


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
        f"{prefix}_mean_scaffold_growth": float(eval_result["mean_scaffold_growth"]),
        f"{prefix}_mean_trace_stabilization": float(
            eval_result["mean_trace_stabilization"]
        ),
        f"{prefix}_mean_prune_amount": float(eval_result["mean_prune_amount"]),
        f"{prefix}_worst_case_strong_success": float(
            eval_result["worst_case_strong_success"]
        ),
        f"{prefix}_weak_success_rate": float(rates["weak_success_rate"]),
        f"{prefix}_strong_success_rate": float(rates["strong_success_rate"]),
        f"{prefix}_strict_success_rate": float(rates["strict_success_rate"]),
    }


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
        and float(summary["program_interpretability"] or 0.0)
        > float(cfg["metrics"]["interpretability_threshold"])
    )


def failure_mode(summary: dict, cfg: dict) -> str:
    if robust_discovery_pass(summary, cfg):
        return "robust_grammar_discovery"

    if str(summary["search_mode"]) == "grammar_random":
        return "random_grammar_reference"

    if (
        float(summary["evolved_mean_connectivity"]) < 0.25
        and float(summary["random_mean_connectivity"]) < 0.25
    ):
        return "grammar_under_expressiveness"

    if float(summary["prior_leakage_score"]) >= float(
        cfg["metrics"]["prior_leakage_threshold"]
    ):
        return "grammar_over_leakage"

    if float(summary["matched_evolution_gain"]) <= float(
        cfg["evolution"]["discovery_margin"]
    ):
        return "random_grammar_equivalence"

    if (
        float(summary["program_length"] or 0)
        >= int(cfg["grammar"]["max_program_length"])
        and float(summary["evolved_mean_connectivity"]) < 0.90
    ):
        return "program_bloat"

    if float(summary["evolved_mean_scaffold_growth"]) > 500.0 and float(
        summary["evolved_mean_cost"]
    ) > float(cfg["metrics"]["strong_cost_factor"]):
        return "false_growth_expansion"

    if (
        float(summary["evolved_mean_prune_amount"]) > 500.0
        and float(summary["evolved_mean_connectivity"]) < 0.50
    ):
        return "over_pruning"

    if float(summary["generalization_gap"]) > float(
        cfg["metrics"]["robust_max_generalization_gap"]
    ):
        return "train_test_program_overfit"

    if (
        bool(summary["noise_stress"])
        and float(summary["evolved_strong_success_rate"]) < 0.90
    ):
        return "trace_lock_in_or_noise_fragility"

    return "mixed_failure"


def condition_summary(condition: dict, record: dict, cfg: dict) -> dict:
    library = str(condition["library"])
    genome = record["genome"]
    stats = program_stats(genome, library)

    out = {
        "condition_id": str(condition["id"]),
        "group": str(condition["group"]),
        "task": str(condition["task"]),
        "task_distribution": str(condition["task_distribution"]),
        "library": library,
        "ablation": str(condition["ablation"]),
        "search_mode": str(condition["search_mode"]),
        "path_type": str(condition["path_type"]),
        "perturbation_type": str(condition["perturbation_type"]),
        "fitness_family": str(condition["fitness_family"]),
        "representation": str(condition["representation"]),
        "noise_stress": bool(condition["noise_stress"]),
        "discovery_generation": int(record["generation"]),
        "sample_efficiency": record["sample_efficiency"],
        "matched_random_budget": int(record["matched_random"]["budget"]),
        "best_genome": genome,
        **stats,
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
    pop_size = int(evo["population_size"])
    generations = int(evo["generations"])
    elite_count = int(evo["elite_count"])
    mutation_rate = float(evo["mutation_rate"])
    workers = int(cfg["parallel"]["workers"])
    search_mode = str(condition["search_mode"])

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
        [False, bool(condition["noise_stress"])],
        str(condition["task"]),
    )

    condition_dir = run_dir / "conditions" / str(condition["id"])
    condition_dir.mkdir(parents=True, exist_ok=True)
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
        max_workers=workers,
        initializer=init_worker,
        initargs=(str(root),),
    ) as executor:
        if search_mode == "grammar_random":
            matched_random = matched_random_search(
                executor, cfg, condition, train_cases, test_cases, rng, logger
            )
            best_record = {
                "generation": 0,
                "genome": matched_random["genome"],
                "train": matched_random["train"],
                "test": matched_random["test"],
                "matched_random": matched_random,
                "sample_efficiency": None,
            }
            progress.step(1)
        elif search_mode in {"grammar_evolved", "p18_evolved"}:
            population = [random_genome(cfg, condition, rng) for _ in range(pop_size)]

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

                if best_record is None or best_fit > float(
                    best_record["train"]["fitness"]
                ):
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
                        float(best_eval["mean_scaffold_growth"]),
                        float(best_eval["mean_trace_stabilization"]),
                        float(best_eval["mean_prune_amount"]),
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
                    child = crossover_genome(a, b, cfg, condition, rng)
                    child = mutate_genome(child, cfg, condition, mutation_rate, rng)
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
                executor, cfg, condition, best_record["genome"], test_cases
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
                executor, cfg, condition, train_cases, test_cases, rng, logger
            )
        else:
            raise ValueError(f"unknown search_mode: {search_mode}")

        logger.info(
            jline(
                "condition",
                str(condition["id"]),
                "no_scaffold_reference_start",
                test_cases=len(test_cases),
            )
        )
        best_record["no_scaffold_reference"] = evaluate_no_scaffold(
            executor, cfg, condition, test_cases
        )

    logger.info(
        jline(
            "condition",
            str(condition["id"]),
            "hand_reference_start",
            test_cases=len(test_cases),
        )
    )
    best_record["hand_reference"] = evaluate_hand_reference(
        _WORKER_EXP11, _WORKER_EXP15, cfg, condition, test_cases
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
            "best_train_growth",
            "best_train_trace",
            "best_train_prune",
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
    interpretability = [float(x["program_interpretability"] or 0.0) for x in rows]

    fig = plt.figure(figsize=(11, 5))
    ax = fig.add_subplot(1, 1, 1)
    x = np.arange(len(labels))
    ax.bar(x - 0.2, evolved, width=0.4, label="evolved")
    ax.bar(x + 0.2, random_ref, width=0.4, label="matched random")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Experiment 19 evolved vs matched random success")
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
    ax.set_title("Experiment 19 matched evolution gain")
    ax.set_xlabel("condition")
    ax.set_ylabel("evolved test fitness - matched random test fitness")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(str(figures_dir / "matched_evolution_gain.png"), dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(11, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(leakage, interpretability)
    for row in rows:
        ax.text(
            float(row["prior_leakage_score"]),
            float(row["program_interpretability"] or 0.0),
            str(row["group"]),
            fontsize=8,
        )
    ax.set_title("Experiment 19 prior leakage vs interpretability")
    ax.set_xlabel("prior leakage score")
    ax.set_ylabel("program interpretability")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(str(figures_dir / "leakage_interpretability_plane.png"), dpi=160)
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
    global _WORKER_EXP18

    root = repo_root()
    _WORKER_EXP11 = load_exp11(root)
    _WORKER_EXP15 = load_exp15(root)
    _WORKER_EXP18 = load_exp18(root)

    config_path = root / "config" / "tests" / "exp_19.yaml"
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
    total_units = sum(
        1
        if str(c["search_mode"]) == "grammar_random"
        else int(cfg["evolution"]["generations"])
        for c in conditions
    )
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
        "search_mode",
        "path_type",
        "perturbation_type",
        "fitness_family",
        "representation",
        "noise_stress",
        "discovery_generation",
        "sample_efficiency",
        "matched_random_budget",
        "program_length",
        "instruction_diversity",
        "detect_count",
        "grow_count",
        "stabilize_count",
        "prune_count",
        "program_interpretability",
        "prior_leakage_score",
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
        "evolved_mean_connectivity",
        "evolved_mean_stability",
        "evolved_mean_cost",
        "evolved_mean_outside",
        "evolved_mean_false",
        "evolved_mean_path_tpr",
        "evolved_mean_open_precision",
        "evolved_mean_false_open_rate",
        "evolved_mean_complexity",
        "evolved_mean_scaffold_growth",
        "evolved_mean_trace_stabilization",
        "evolved_mean_prune_amount",
        "evolved_worst_case_strong_success",
        "evolved_weak_success_rate",
        "evolved_strong_success_rate",
        "evolved_strict_success_rate",
        "random_mean_connectivity",
        "random_mean_stability",
        "random_mean_cost",
        "random_mean_outside",
        "random_mean_false",
        "random_mean_scaffold_growth",
        "random_mean_trace_stabilization",
        "random_mean_prune_amount",
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
        str(run_dir / "grammar_scaffold_summary.csv"),
        [[x.get(k) for k in summary_header] for x in rows],
        header=summary_header,
    )
    write_json(str(run_dir / "grammar_scaffold_summary.json"), rows)

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
        "summary_path": "grammar_scaffold_summary.csv",
        "summary_json_path": "grammar_scaffold_summary.json",
        "figures": [
            "figures/evolved_vs_matched_random_success.png",
            "figures/matched_evolution_gain.png",
            "figures/leakage_interpretability_plane.png",
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
