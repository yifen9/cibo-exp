from pathlib import Path
from collections import defaultdict
from statistics import mean
from concurrent.futures import ProcessPoolExecutor
import importlib.util
import hashlib
import argparse

import numpy as np

from cibo_pycore.io.csv import write_csv
from cibo_pycore.io.json import write_json, read_json
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
_WORKER_EXP19 = None


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
        "cibo_exp_11_runtime_for_exp_20",
    )


def load_exp15(root: Path):
    return load_module(
        root / "scripts" / "tests" / "exp_15.py",
        "cibo_exp_15_runtime_for_exp_20",
    )


def load_exp19(root: Path):
    return load_module(
        root / "scripts" / "tests" / "exp_19.py",
        "cibo_exp_19_runtime_for_exp_20",
    )


def init_worker(root_text: str) -> None:
    global _WORKER_EXP11
    global _WORKER_EXP15
    global _WORKER_EXP19
    root = Path(root_text)
    _WORKER_EXP11 = load_exp11(root)
    _WORKER_EXP15 = load_exp15(root)
    _WORKER_EXP19 = load_exp19(root)


def detect_ops() -> list[str]:
    return [
        "detect_gap",
        "detect_two",
        "detect_frontier",
        "detect_deadend",
        "detect_weak_support",
        "detect_local_break",
    ]


def grow_ops() -> list[str]:
    return [
        "grow_mask_if_detected",
        "grow_band_if_two",
        "grow_bypass_if_frontier",
        "grow_priority_if_gap",
    ]


def prune_ops() -> list[str]:
    return [
        "prune_deadend",
        "prune_unsupported",
        "prune_over_budget",
        "prune_weak_priority",
    ]


def stabilize_ops() -> list[str]:
    return [
        "refresh_supported",
        "decay_unsupported",
        "accumulate_priority",
        "stabilize_confirmed",
    ]


def random_block_item(ops: list[str], cfg: dict, rng: np.random.Generator) -> dict:
    return {
        "op": str(rng.choice(ops)),
        "strength": float(rng.uniform(0.20, 1.00)),
        "threshold": float(rng.uniform(0.20, 0.80)),
        "radius": int(rng.integers(1, int(cfg["grammar"]["max_radius"]) + 1)),
    }


def random_block(ops: list[str], cfg: dict, rng: np.random.Generator) -> list[dict]:
    n = int(
        rng.integers(
            int(cfg["grammar"]["min_block_length"]),
            int(cfg["grammar"]["max_block_length"]) + 1,
        )
    )
    return [random_block_item(ops, cfg, rng) for _ in range(n)]


def apply_ablation(genome: dict, ablation: str) -> dict:
    out = {
        "detect_block": [dict(x) for x in genome["detect_block"]],
        "grow_block": [dict(x) for x in genome["grow_block"]],
        "prune_block": [dict(x) for x in genome["prune_block"]],
        "stabilize_block": [dict(x) for x in genome["stabilize_block"]],
        "mask_decay": float(genome["mask_decay"]),
        "trace_decay": float(genome["trace_decay"]),
        "priority_decay": float(genome["priority_decay"]),
        "trace_refresh": float(genome["trace_refresh"]),
        "budget_limit": float(genome["budget_limit"]),
        "evidence_gain": float(genome["evidence_gain"]),
        "detect_threshold": float(genome["detect_threshold"]),
        "prune_threshold": float(genome["prune_threshold"]),
    }

    if ablation == "none":
        return out

    if ablation == "no_prune":
        out["prune_block"] = []
        return out

    if ablation == "no_stabilize":
        out["stabilize_block"] = []
        return out

    raise ValueError(f"unknown ablation: {ablation}")


def random_constrained_genome(
    cfg: dict, condition: dict, rng: np.random.Generator
) -> dict:
    genome = {
        "detect_block": random_block(detect_ops(), cfg, rng),
        "grow_block": random_block(grow_ops(), cfg, rng),
        "prune_block": random_block(prune_ops(), cfg, rng),
        "stabilize_block": random_block(stabilize_ops(), cfg, rng),
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
        "detect_threshold": float(
            rng.uniform(
                float(cfg["grammar"]["detect_threshold_min"]),
                float(cfg["grammar"]["detect_threshold_max"]),
            )
        ),
        "prune_threshold": float(
            rng.uniform(
                float(cfg["grammar"]["prune_threshold_min"]),
                float(cfg["grammar"]["prune_threshold_max"]),
            )
        ),
    }
    return apply_ablation(genome, str(condition["ablation"]))


def free_condition(condition: dict) -> dict:
    out = dict(condition)
    out["library"] = "grammar"
    out["representation"] = "G0"
    return out


def random_genome(cfg: dict, condition: dict, rng: np.random.Generator) -> dict:
    library = str(condition["library"])

    if library == "free_grammar":
        return _WORKER_EXP19.random_genome(cfg, free_condition(condition), rng)

    if library == "constrained_grammar":
        return random_constrained_genome(cfg, condition, rng)

    raise ValueError(f"unknown library: {library}")


def mutate_block_item(
    item: dict, ops: list[str], cfg: dict, rng: np.random.Generator
) -> dict:
    out = dict(item)
    field = str(rng.choice(["op", "strength", "threshold", "radius"]))

    if field == "op":
        out["op"] = str(rng.choice(ops))
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
        raise ValueError(f"unknown mutation field: {field}")

    return out


def mutate_block(
    block: list[dict], ops: list[str], cfg: dict, rate: float, rng: np.random.Generator
) -> list[dict]:
    out = [dict(x) for x in block]

    for i in range(len(out)):
        if rng.random() < rate:
            out[i] = mutate_block_item(out[i], ops, cfg, rng)

    if rng.random() < rate:
        if len(out) < int(cfg["grammar"]["max_block_length"]):
            pos = int(rng.integers(0, len(out) + 1))
            out.insert(pos, random_block_item(ops, cfg, rng))

    if rng.random() < rate:
        if len(out) > int(cfg["grammar"]["min_block_length"]):
            pos = int(rng.integers(0, len(out)))
            out.pop(pos)

    return out


def mutate_constrained_genome(
    g: dict, cfg: dict, condition: dict, rate: float, rng: np.random.Generator
) -> dict:
    out = {
        "detect_block": mutate_block(g["detect_block"], detect_ops(), cfg, rate, rng),
        "grow_block": mutate_block(g["grow_block"], grow_ops(), cfg, rate, rng),
        "prune_block": mutate_block(g["prune_block"], prune_ops(), cfg, rate, rng),
        "stabilize_block": mutate_block(
            g["stabilize_block"], stabilize_ops(), cfg, rate, rng
        ),
        "mask_decay": float(g["mask_decay"]),
        "trace_decay": float(g["trace_decay"]),
        "priority_decay": float(g["priority_decay"]),
        "trace_refresh": float(g["trace_refresh"]),
        "budget_limit": float(g["budget_limit"]),
        "evidence_gain": float(g["evidence_gain"]),
        "detect_threshold": float(g["detect_threshold"]),
        "prune_threshold": float(g["prune_threshold"]),
    }

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

    for key in ["detect_threshold", "prune_threshold"]:
        if rng.random() < rate:
            out[key] = float(
                np.clip(float(out[key]) + rng.normal(0.0, 0.08), 0.05, 0.95)
            )

    return apply_ablation(out, str(condition["ablation"]))


def mutate_genome(
    g: dict, cfg: dict, condition: dict, rate: float, rng: np.random.Generator
) -> dict:
    library = str(condition["library"])

    if library == "free_grammar":
        return _WORKER_EXP19.mutate_genome(g, cfg, free_condition(condition), rate, rng)

    if library == "constrained_grammar":
        return mutate_constrained_genome(g, cfg, condition, rate, rng)

    raise ValueError(f"unknown library: {library}")


def crossover_block(
    a: list[dict], b: list[dict], ops: list[str], cfg: dict, rng: np.random.Generator
) -> list[dict]:
    ca = int(rng.integers(0, len(a) + 1))
    cb = int(rng.integers(0, len(b) + 1))
    out = [dict(x) for x in a[:ca] + b[cb:]]

    while len(out) < int(cfg["grammar"]["min_block_length"]):
        out.append(random_block_item(ops, cfg, rng))

    if len(out) > int(cfg["grammar"]["max_block_length"]):
        out = out[: int(cfg["grammar"]["max_block_length"])]

    return out


def crossover_constrained(
    a: dict, b: dict, cfg: dict, condition: dict, rng: np.random.Generator
) -> dict:
    out = {
        "detect_block": crossover_block(
            a["detect_block"], b["detect_block"], detect_ops(), cfg, rng
        ),
        "grow_block": crossover_block(
            a["grow_block"], b["grow_block"], grow_ops(), cfg, rng
        ),
        "prune_block": crossover_block(
            a["prune_block"], b["prune_block"], prune_ops(), cfg, rng
        ),
        "stabilize_block": crossover_block(
            a["stabilize_block"], b["stabilize_block"], stabilize_ops(), cfg, rng
        ),
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
        "detect_threshold": float(
            a["detect_threshold"] if rng.random() < 0.5 else b["detect_threshold"]
        ),
        "prune_threshold": float(
            a["prune_threshold"] if rng.random() < 0.5 else b["prune_threshold"]
        ),
    }
    return apply_ablation(out, str(condition["ablation"]))


def crossover_genome(
    a: dict, b: dict, cfg: dict, condition: dict, rng: np.random.Generator
) -> dict:
    library = str(condition["library"])

    if library == "free_grammar":
        return _WORKER_EXP19.crossover_genome(a, b, cfg, free_condition(condition), rng)

    if library == "constrained_grammar":
        return crossover_constrained(a, b, cfg, condition, rng)

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


def constrained_complexity(genome: dict) -> float:
    detect = len(genome["detect_block"])
    grow = len(genome["grow_block"])
    prune = len(genome["prune_block"])
    stabilize = len(genome["stabilize_block"])
    return float(
        detect + grow + prune + stabilize + 0.25 * len(constrained_ops(genome))
    )


def constrained_ops(genome: dict) -> list[str]:
    out = []

    for key in ["detect_block", "grow_block", "prune_block", "stabilize_block"]:
        out.extend([str(x["op"]) for x in genome[key]])

    return out


def constrained_program_stats(genome: dict, library: str) -> dict:
    if library == "free_grammar":
        stats = _WORKER_EXP19.program_stats(genome, "grammar")
        return {
            "program_length": stats["program_length"],
            "instruction_diversity": stats["instruction_diversity"],
            "detect_count": stats["detect_count"],
            "grow_count": stats["grow_count"],
            "prune_count": stats["prune_count"],
            "stabilize_count": stats["stabilize_count"],
            "repair_loop_completeness": 0,
            "program_interpretability": stats["program_interpretability"],
            "prior_leakage_score": stats["prior_leakage_score"],
        }

    if library != "constrained_grammar":
        raise ValueError(f"unknown library for program stats: {library}")

    ops = constrained_ops(genome)
    detect = len(genome["detect_block"])
    grow = len(genome["grow_block"])
    prune = len(genome["prune_block"])
    stabilize = len(genome["stabilize_block"])
    length = detect + grow + prune + stabilize
    diversity = len(set(ops))
    completeness = int(detect > 0 and grow > 0 and prune > 0 and stabilize > 0)
    balance = min(detect, grow, prune + 1, stabilize + 1) / max(
        1, max(detect, grow, prune + 1, stabilize + 1)
    )
    interpretability = float(
        np.clip(0.45 + 0.35 * balance + 0.20 * min(1.0, diversity / 8.0), 0.0, 1.0)
    )
    leakage_ops = {
        "detect_gap",
        "detect_local_break",
        "grow_band_if_two",
        "grow_bypass_if_frontier",
        "grow_priority_if_gap",
    }
    leakage = sum(1 for x in ops if x in leakage_ops) / max(1, length)
    prior_leakage = float(np.clip(0.20 + 0.65 * leakage, 0.0, 1.0))

    return {
        "program_length": int(length),
        "instruction_diversity": int(diversity),
        "detect_count": int(detect),
        "grow_count": int(grow),
        "prune_count": int(prune),
        "stabilize_count": int(stabilize),
        "repair_loop_completeness": int(completeness),
        "program_interpretability": interpretability,
        "prior_leakage_score": prior_leakage,
    }


def detect_field_from_item(maps: dict, item: dict) -> np.ndarray:
    op = str(item["op"])
    strength = float(item["strength"])
    threshold = float(item["threshold"])

    if op == "detect_gap":
        return strength * maps["gap"]

    if op == "detect_two":
        return strength * maps["two"]

    if op == "detect_frontier":
        return strength * maps["frontier"]

    if op == "detect_deadend":
        return strength * maps["deadend"]

    if op == "detect_weak_support":
        support = np.maximum.reduce([maps["two"], maps["frontier"], maps["band"]])
        return strength * (support >= threshold).astype(np.float64)

    if op == "detect_local_break":
        return strength * np.maximum(maps["gap"], maps["bypass"])

    raise ValueError(f"unknown detect op: {op}")


def execute_constrained_program(
    exp19,
    canvas: np.ndarray,
    mask: np.ndarray,
    program_trace: np.ndarray,
    priority: np.ndarray,
    budget: np.ndarray,
    genome: dict,
    initial_open: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    maps = exp19.build_maps(canvas)
    mask = mask * float(genome["mask_decay"])
    program_trace = program_trace * float(genome["trace_decay"])
    priority = priority * float(genome["priority_decay"])
    budget = budget * 0.90
    detection = np.zeros_like(canvas, dtype=np.float64)

    for item in genome["detect_block"]:
        detection = np.maximum(detection, detect_field_from_item(maps, item))

    detection_gate = (detection >= float(genome["detect_threshold"])).astype(np.float64)
    gated_area = exp19.dilate_field(detection_gate, 1)
    growth_before_prune = np.zeros_like(canvas, dtype=np.float64)
    growth_total = 0.0
    gated_growth_total = 0.0
    prune_total = 0.0
    trace_total = 0.0

    for item in genome["grow_block"]:
        op = str(item["op"])
        strength = float(item["strength"])
        radius = int(item["radius"])
        threshold = float(item["threshold"])

        if op == "grow_mask_if_detected":
            grown = exp19.dilate_field(detection, radius) * gated_area
        elif op == "grow_band_if_two":
            grown = (
                exp19.dilate_field(
                    (maps["two"] >= threshold).astype(np.float64), radius
                )
                * gated_area
            )
        elif op == "grow_bypass_if_frontier":
            grown = (
                exp19.dilate_field(maps["bypass"] * maps["frontier"], radius)
                * gated_area
            )
        elif op == "grow_priority_if_gap":
            grown = exp19.dilate_field(maps["gap"], radius) * gated_area
        else:
            raise ValueError(f"unknown grow op: {op}")

        growth_before_prune = np.maximum(growth_before_prune, strength * grown)
        mask = np.maximum(mask, strength * grown)
        priority = np.maximum(priority, strength * np.maximum(grown, detection))
        growth_total += float(grown.sum())
        gated_growth_total += float((grown * gated_area).sum())

    for item in genome["prune_block"]:
        op = str(item["op"])
        strength = float(item["strength"])
        threshold = float(item["threshold"])

        if op == "prune_deadend":
            penalty = maps["deadend"]
        elif op == "prune_unsupported":
            support = np.maximum.reduce(
                [maps["gap"], maps["two"], maps["frontier"], maps["band"], detection]
            )
            penalty = (
                support < max(threshold, float(genome["prune_threshold"]))
            ).astype(np.float64)
        elif op == "prune_over_budget":
            factor = float((canvas.sum() + mask.sum()) / max(1, initial_open))
            penalty = np.ones_like(mask) * max(
                0.0, factor - float(genome["budget_limit"])
            )
        elif op == "prune_weak_priority":
            penalty = (
                priority < max(threshold, float(genome["prune_threshold"]))
            ).astype(np.float64)
        else:
            raise ValueError(f"unknown prune op: {op}")

        clipped = np.clip(strength * penalty, 0.0, 1.0)
        mask *= 1.0 - clipped
        priority *= 1.0 - 0.5 * clipped
        budget = np.maximum(budget, clipped)
        prune_total += float(clipped.sum())

    survived_growth = np.minimum(mask, growth_before_prune)

    for item in genome["stabilize_block"]:
        op = str(item["op"])
        strength = float(item["strength"])
        threshold = float(item["threshold"])

        if op == "refresh_supported":
            support = np.maximum.reduce([survived_growth, priority, detection])
            add = strength * support
            program_trace = np.maximum(
                program_trace, float(genome["trace_refresh"]) * add
            )
            trace_total += float(add.sum())
        elif op == "decay_unsupported":
            support = np.maximum.reduce([survived_growth, priority, detection])
            decay_mask = (support < threshold).astype(np.float64)
            program_trace *= 1.0 - np.clip(0.25 * strength * decay_mask, 0.0, 1.0)
        elif op == "accumulate_priority":
            add = strength * priority
            program_trace = np.maximum(
                program_trace, float(genome["trace_refresh"]) * add
            )
            trace_total += float(add.sum())
        elif op == "stabilize_confirmed":
            confirmed = (survived_growth * priority >= threshold).astype(np.float64)
            add = strength * confirmed
            program_trace = np.maximum(
                program_trace, float(genome["trace_refresh"]) * add
            )
            trace_total += float(add.sum())
        else:
            raise ValueError(f"unknown stabilize op: {op}")

    info = {
        "growth_total": float(growth_total),
        "gated_growth_total": float(gated_growth_total),
        "prune_total": float(prune_total),
        "trace_total": float(trace_total),
        "survived_growth_total": float(survived_growth.sum()),
    }

    return (
        np.clip(mask, 0.0, 1.0),
        np.clip(program_trace, -1.0, 1.0),
        np.clip(priority, 0.0, 1.0),
        np.clip(budget, 0.0, 1.0),
        info,
    )


def run_constrained_trial(
    exp11, exp19, cfg: dict, condition: dict, genome: dict, case: dict
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
    operators = exp19.init_operators(int(cfg["operator"]["count"]), height, width, rng)
    initial_open = int(canvas.sum())

    connectivity_values = []
    open_cost_factors = []
    outside_rates = []
    false_counts = []
    true_open_total = 0
    false_open_total = 0
    applied_total = 0
    growth_total = 0.0
    gated_growth_total = 0.0
    prune_total = 0.0
    trace_total = 0.0
    survived_growth_total = 0.0

    for t in range(steps + 1):
        if t == int(condition["damage_time"]):
            exp19.bridge_cut(canvas, reference, condition, offset=offset)

        mask, program_trace, priority, budget, info = execute_constrained_program(
            exp19,
            canvas,
            mask,
            program_trace,
            priority,
            budget,
            genome,
            initial_open,
        )
        growth_total += float(info["growth_total"])
        gated_growth_total += float(info["gated_growth_total"])
        prune_total += float(info["prune_total"])
        trace_total += float(info["trace_total"])
        survived_growth_total += float(info["survived_growth_total"])

        proposals = defaultdict(list)
        proposal_meta = defaultdict(list)

        if t < steps:
            exp19.move_operators(operators, height, width, rng)

            for pos in operators:
                x = int(pos[0])
                y = int(pos[1])
                local = exp19.local_density(
                    canvas, x, y, int(cfg["operator"]["radius"])
                )
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

                temporal_pass, _ = exp19.update_sliding(
                    history,
                    support,
                    float(cfg["repair"]["theta_repair"]),
                    x,
                    y,
                    t,
                    int(cfg["repair"]["k"]),
                    int(cfg["repair"]["m"]),
                )
                result = exp19.choose_action(
                    canvas, support, temporal_pass, x, y, cfg, rng
                )

                if result["action"] != "stay":
                    proposals[(x, y)].append(int(result["delta"]))
                    proposal_meta[(x, y)].append(result)

            before = canvas.copy()
            stats = exp19.resolve_actions(canvas, proposals)
            applied_total += int(stats["applied_actions"])

            for cell, metas in proposal_meta.items():
                x, y = cell

                for item in metas:
                    if int(item["delta"]) > 0:
                        label = exp19.classify_open(before, reference, x, y)
                        if label == "true_open":
                            true_open_total += 1
                        elif label == "false_open":
                            false_open_total += 1

        metrics = exp19.compute_metrics(
            exp11, canvas, reference, start, goal, initial_open
        )
        connectivity_values.append(int(metrics["connectivity"]))
        open_cost_factors.append(float(metrics["open_cost_factor"]))
        outside_rates.append(float(metrics["outside_open_rate"]))
        false_counts.append(float(metrics["false_corridor_count"]))

    final = exp19.compute_metrics(exp11, canvas, reference, start, goal, initial_open)
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
        "scaffold_complexity": constrained_complexity(genome),
        "scaffold_growth_amount": float(growth_total),
        "gated_growth_amount": float(gated_growth_total),
        "trace_stabilization_amount": float(trace_total),
        "prune_amount": float(prune_total),
        "survived_growth_amount": float(survived_growth_total),
    }


def run_trial_by_library(
    exp11, exp19, cfg: dict, condition: dict, genome: dict, case: dict
) -> dict:
    library = str(condition["library"])

    if library == "free_grammar":
        return exp19.run_grammar_trial(
            exp11, cfg, free_condition(condition), genome, case
        )

    if library == "constrained_grammar":
        return run_constrained_trial(exp11, exp19, cfg, condition, genome, case)

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
        "mean_scaffold_growth": float(
            mean([float(x["scaffold_growth_amount"]) for x in trials])
        ),
        "mean_gated_growth": float(
            mean([float(x.get("gated_growth_amount", 0.0)) for x in trials])
        ),
        "mean_trace_stabilization": float(
            mean([float(x["trace_stabilization_amount"]) for x in trials])
        ),
        "mean_prune_amount": float(mean([float(x["prune_amount"]) for x in trials])),
        "mean_survived_growth": float(
            mean([float(x.get("survived_growth_amount", 0.0)) for x in trials])
        ),
        "worst_case_strong_success": float(worst_subgroup_success(trials, cfg)),
        "trials": trials,
    }


def evaluate_genome_once(
    exp11, exp19, cfg: dict, condition: dict, genome: dict, cases: list[dict]
) -> dict:
    trials = [
        run_trial_by_library(exp11, exp19, cfg, condition, genome, case)
        for case in cases
    ]
    scores = [base_score(trial, cfg) for trial in trials]
    fitness_value = aggregate_fitness(
        scores,
        str(condition["fitness_family"]),
        float(cfg["evolution"]["cvar_alpha"]),
    )
    return summarize_eval(fitness_value, scores, trials, cfg)


def evaluate_genome_with_modules(
    exp11,
    exp19,
    cfg: dict,
    condition: dict,
    genome: dict,
    train_cases: list[dict],
    val_cases: list[dict],
) -> dict:
    train = evaluate_genome_once(exp11, exp19, cfg, condition, genome, train_cases)

    if str(condition["selection_mode"]) == "train_cvar":
        return {
            "fitness": float(train["fitness"]),
            "train": train,
            "val": None,
        }

    if str(condition["selection_mode"]) == "validation_cvar":
        val = evaluate_genome_once(exp11, exp19, cfg, condition, genome, val_cases)
        beta = float(cfg["selection"]["beta_val"])
        penalty = float(cfg["selection"]["gap_penalty"])
        fit = (
            (1.0 - beta) * float(train["fitness"])
            + beta * float(val["fitness"])
            - penalty * abs(float(train["fitness"]) - float(val["fitness"]))
        )
        return {
            "fitness": float(fit),
            "train": train,
            "val": val,
        }

    raise ValueError(f"unknown selection_mode: {condition['selection_mode']}")


def evaluate_worker(args: tuple) -> dict:
    cfg, condition, genome, train_cases, val_cases = args
    return evaluate_genome_with_modules(
        _WORKER_EXP11,
        _WORKER_EXP19,
        cfg,
        condition,
        genome,
        train_cases,
        val_cases,
    )


def evaluate_population(
    executor: ProcessPoolExecutor,
    cfg: dict,
    condition: dict,
    population: list[dict],
    train_cases: list[dict],
    val_cases: list[dict],
) -> list[dict]:
    jobs = [(cfg, condition, genome, train_cases, val_cases) for genome in population]
    return list(executor.map(evaluate_worker, jobs, chunksize=1))


def evaluate_test_worker(args: tuple) -> dict:
    cfg, condition, genome, test_cases = args
    return evaluate_genome_once(
        _WORKER_EXP11,
        _WORKER_EXP19,
        cfg,
        condition,
        genome,
        test_cases,
    )


def evaluate_single_test(
    executor: ProcessPoolExecutor,
    cfg: dict,
    condition: dict,
    genome: dict,
    test_cases: list[dict],
) -> dict:
    jobs = [(cfg, condition, genome, test_cases)]
    return list(executor.map(evaluate_test_worker, jobs, chunksize=1))[0]


def evaluate_no_scaffold(
    executor: ProcessPoolExecutor, cfg: dict, condition: dict, test_cases: list[dict]
) -> dict:
    c = dict(condition)
    c["library"] = "free_grammar"
    c["representation"] = "G0"
    genome = _WORKER_EXP19.no_scaffold_genome()
    return evaluate_single_test(executor, cfg, c, genome, test_cases)


def adapt_legacy_reference_eval(eval_result: dict) -> dict:
    out = dict(eval_result)
    out["mean_gated_growth"] = 0.0
    out["mean_survived_growth"] = 0.0
    return out


def evaluate_hand_reference(
    exp11, exp15, exp19, cfg: dict, condition: dict, test_cases: list[dict]
) -> dict:
    return adapt_legacy_reference_eval(
        exp19.evaluate_hand_reference(
            exp11, exp15, cfg, free_condition(condition), test_cases
        )
    )


def global_reference_eval(
    exp19, cfg: dict, condition: dict, test_cases: list[dict]
) -> dict:
    return adapt_legacy_reference_eval(
        exp19.global_reference_eval(cfg, free_condition(condition), test_cases)
    )


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
    val_cases: list[dict],
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
            executor, cfg, condition, candidates, train_cases, val_cases
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

    test_eval = evaluate_single_test(executor, cfg, condition, best_genome, test_cases)

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
        f"{prefix}_mean_gated_growth": float(eval_result["mean_gated_growth"]),
        f"{prefix}_mean_trace_stabilization": float(
            eval_result["mean_trace_stabilization"]
        ),
        f"{prefix}_mean_prune_amount": float(eval_result["mean_prune_amount"]),
        f"{prefix}_mean_survived_growth": float(eval_result["mean_survived_growth"]),
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
        and int(summary["repair_loop_completeness"]) == 1
    )


def failure_mode(summary: dict, cfg: dict) -> str:
    if robust_discovery_pass(summary, cfg):
        return "robust_constrained_grammar_discovery"

    if (
        str(summary["library"]) == "constrained_grammar"
        and float(summary["evolved_mean_connectivity"]) < 0.25
        and float(summary["random_mean_connectivity"]) < 0.25
    ):
        return "constrained_grammar_under_expressiveness"

    if float(summary["prior_leakage_score"]) >= float(
        cfg["metrics"]["prior_leakage_threshold"]
    ):
        return "prior_leakage_return"

    if float(summary["matched_evolution_gain"]) <= float(
        cfg["evolution"]["discovery_margin"]
    ):
        return "random_constrained_grammar_equivalence"

    if float(summary["evolved_mean_scaffold_growth"]) > 500.0 and float(
        summary["evolved_mean_cost"]
    ) > float(cfg["metrics"]["strong_cost_factor"]):
        return "residual_false_growth"

    if (
        float(summary["evolved_mean_prune_amount"]) > 500.0
        and float(summary["evolved_mean_connectivity"]) < 0.50
    ):
        return "over_pruning"

    if (
        bool(summary["noise_stress"])
        and float(summary["evolved_strong_success_rate"]) < 0.90
    ):
        return "trace_lock_in_or_noise_fragility"

    if float(summary["generalization_gap"]) > float(
        cfg["metrics"]["robust_max_generalization_gap"]
    ):
        return "validation_or_train_test_overfit"

    if (
        str(summary["task"]) == "mixed"
        and float(summary["evolved_strong_success_rate"]) < 0.50
    ):
        return "mixed_task_collapse"

    return "mixed_failure"


def condition_summary(condition: dict, record: dict, cfg: dict) -> dict:
    library = str(condition["library"])
    genome = record["genome"]
    stats = constrained_program_stats(genome, library)

    out = {
        "condition_id": str(condition["id"]),
        "group": str(condition["group"]),
        "task": str(condition["task"]),
        "task_distribution": str(condition["task_distribution"]),
        "library": library,
        "ablation": str(condition["ablation"]),
        "search_mode": str(condition["search_mode"]),
        "selection_mode": str(condition["selection_mode"]),
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

    out.update(eval_to_prefixed("train", record["train_eval"], cfg))
    out.update(eval_to_prefixed("evolved", record["test"], cfg))
    out.update(eval_to_prefixed("random", record["matched_random"]["test"], cfg))
    out.update(eval_to_prefixed("no_scaffold", record["no_scaffold_reference"], cfg))
    out.update(eval_to_prefixed("hand", record["hand_reference"], cfg))
    out.update(eval_to_prefixed("global", record["global_reference"], cfg))

    if record["val_eval"] is None:
        out["val_fitness"] = None
        out["val_mean_connectivity"] = None
        out["train_val_gap"] = None
    else:
        out.update(eval_to_prefixed("val", record["val_eval"], cfg))
        out["train_val_gap"] = float(out["train_fitness"] - out["val_fitness"])

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
    out["gated_growth_ratio"] = float(
        out["evolved_mean_gated_growth"]
        / max(1e-9, out["evolved_mean_scaffold_growth"])
    )
    out["prune_survival_ratio"] = float(
        out["evolved_mean_survived_growth"]
        / max(1e-9, out["evolved_mean_scaffold_growth"])
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

    train_cases = make_cases(
        [int(x) for x in cfg["run"]["train_seeds"]],
        [int(x) for x in cfg["run"]["train_offsets"]],
        [str(x) for x in cfg["run"]["train_variants"]],
        [False],
        str(condition["task"]),
    )

    val_cases = make_cases(
        [int(x) for x in cfg["run"]["val_seeds"]],
        [int(x) for x in cfg["run"]["val_offsets"]],
        [str(x) for x in cfg["run"]["val_variants"]],
        [False, bool(condition["noise_stress"])],
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
            val_cases=len(val_cases),
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
        population = [random_genome(cfg, condition, rng) for _ in range(pop_size)]

        for gen in range(generations):
            evaluations = evaluate_population(
                executor, cfg, condition, population, train_cases, val_cases
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
                best_record["selection_fitness"]
            ):
                best_record = {
                    "generation": int(gen),
                    "genome": dict(population[best_idx]),
                    "selection_fitness": best_fit,
                    "train_eval": best_eval["train"],
                    "val_eval": best_eval["val"],
                }

            history_rows.append(
                [
                    int(gen),
                    best_fit,
                    mean_fit,
                    float(best_eval["train"]["mean_final_connectivity"]),
                    float(best_eval["train"]["mean_cost"]),
                    float(best_eval["train"]["mean_outside"]),
                    float(best_eval["train"]["worst_case_strong_success"]),
                    float(best_eval["train"]["worst_base_score"]),
                    float(best_eval["train"]["cvar_base_score"]),
                    float(best_eval["train"]["mean_scaffold_growth"]),
                    float(best_eval["train"]["mean_gated_growth"]),
                    float(best_eval["train"]["mean_trace_stabilization"]),
                    float(best_eval["train"]["mean_prune_amount"]),
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
                    best_connectivity=float(
                        best_eval["train"]["mean_final_connectivity"]
                    ),
                    best_cost=float(best_eval["train"]["mean_cost"]),
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
        best_record["test"] = evaluate_single_test(
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
            executor, cfg, condition, train_cases, val_cases, test_cases, rng, logger
        )

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
        _WORKER_EXP11, _WORKER_EXP15, _WORKER_EXP19, cfg, condition, test_cases
    )

    logger.info(
        jline(
            "condition",
            str(condition["id"]),
            "global_reference_start",
            test_cases=len(test_cases),
        )
    )
    best_record["global_reference"] = global_reference_eval(
        _WORKER_EXP19, cfg, condition, test_cases
    )

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
            "best_train_gated_growth",
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
            gated_growth=float(summary["gated_growth_ratio"]),
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
    gap = [float(x["generalization_gap"]) for x in rows]
    gated = [float(x["gated_growth_ratio"]) for x in rows]

    fig = plt.figure(figsize=(11, 5))
    ax = fig.add_subplot(1, 1, 1)
    x = np.arange(len(labels))
    ax.bar(x - 0.2, evolved, width=0.4, label="evolved")
    ax.bar(x + 0.2, random_ref, width=0.4, label="matched random")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Experiment 20 evolved vs matched random success")
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
    ax.set_title("Experiment 20 matched evolution gain")
    ax.set_xlabel("condition")
    ax.set_ylabel("evolved test fitness - matched random test fitness")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(str(figures_dir / "matched_evolution_gain.png"), dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(11, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(labels, gap)
    ax.set_title("Experiment 20 train-test gap")
    ax.set_xlabel("condition")
    ax.set_ylabel("train fitness - held-out fitness")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(str(figures_dir / "train_test_gap.png"), dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(11, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(labels, gated)
    ax.set_title("Experiment 20 gated growth ratio")
    ax.set_xlabel("condition")
    ax.set_ylabel("gated growth / total growth")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(str(figures_dir / "gated_growth_ratio.png"), dpi=160)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-run-dir", type=str, default="")
    parser.add_argument("--force-new", action="store_true")
    return parser.parse_args()


def resolve_run_dir(
    output_root: Path, meta: dict, args: argparse.Namespace
) -> tuple[Path, bool]:
    if args.force_new:
        return Path(build_version_dir(str(output_root), meta)), False

    if args.resume_run_dir:
        path = Path(args.resume_run_dir)

        if not path.is_absolute():
            path = repo_root() / path

        if not path.exists():
            raise FileNotFoundError(f"resume run dir does not exist: {path}")

        if not path.is_dir():
            raise NotADirectoryError(f"resume run dir is not a directory: {path}")

        return path, True

    return Path(build_version_dir(str(output_root), meta)), False


def condition_summary_path(run_dir: Path, condition: dict) -> Path:
    return run_dir / "conditions" / str(condition["id"]) / "summary.json"


def condition_best_record_path(run_dir: Path, condition: dict) -> Path:
    return run_dir / "conditions" / str(condition["id"]) / "best_genome.json"


def condition_is_complete(run_dir: Path, condition: dict) -> bool:
    path = condition_summary_path(run_dir, condition)

    if not path.exists():
        return False

    try:
        obj = read_json(str(path))
    except Exception:
        return False

    return str(obj.get("condition_id")) == str(condition["id"])


def condition_has_recoverable_best(run_dir: Path, condition: dict) -> bool:
    path = condition_best_record_path(run_dir, condition)

    if not path.exists():
        return False

    try:
        obj = read_json(str(path))
    except Exception:
        return False

    required = ["genome", "train_eval", "sample_efficiency"]

    for key in required:
        if key not in obj:
            return False

    return True


def load_condition_summary(run_dir: Path, condition: dict) -> dict:
    path = condition_summary_path(run_dir, condition)

    if not path.exists():
        raise FileNotFoundError(f"condition summary does not exist: {path}")

    return read_json(str(path))


def load_recoverable_best_record(run_dir: Path, condition: dict) -> dict:
    path = condition_best_record_path(run_dir, condition)

    if not path.exists():
        raise FileNotFoundError(f"best_genome does not exist: {path}")

    obj = read_json(str(path))

    if "genome" not in obj:
        raise KeyError(f"recoverable best_genome missing genome: {path}")

    if "train_eval" not in obj:
        raise KeyError(f"recoverable best_genome missing train_eval: {path}")

    obj["generation"] = int(obj.get("generation", 0))
    obj["sample_efficiency"] = obj.get("sample_efficiency")
    obj["val_eval"] = obj.get("val_eval")

    return obj


def finalize_recoverable_condition(
    cfg: dict,
    condition: dict,
    root: Path,
    run_dir: Path,
    logger: Logger,
) -> dict:
    rng = np.random.default_rng(stable_seed(str(condition["id"])))
    evo = cfg["evolution"]
    pop_size = int(evo["population_size"])
    generations = int(evo["generations"])
    workers = int(cfg["parallel"]["workers"])

    train_cases = make_cases(
        [int(x) for x in cfg["run"]["train_seeds"]],
        [int(x) for x in cfg["run"]["train_offsets"]],
        [str(x) for x in cfg["run"]["train_variants"]],
        [False],
        str(condition["task"]),
    )

    val_cases = make_cases(
        [int(x) for x in cfg["run"]["val_seeds"]],
        [int(x) for x in cfg["run"]["val_offsets"]],
        [str(x) for x in cfg["run"]["val_variants"]],
        [False, bool(condition["noise_stress"])],
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
    best_record = load_recoverable_best_record(run_dir, condition)

    logger.info(
        jline(
            "condition",
            str(condition["id"]),
            "recover_best_record",
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
        if "test" not in best_record:
            logger.info(
                jline(
                    "condition",
                    str(condition["id"]),
                    "recover_heldout_start",
                    test_cases=len(test_cases),
                )
            )
            best_record["test"] = evaluate_single_test(
                executor,
                cfg,
                condition,
                best_record["genome"],
                test_cases,
            )

        if "matched_random" not in best_record:
            logger.info(
                jline(
                    "condition",
                    str(condition["id"]),
                    "recover_matched_random_start",
                    budget=pop_size * generations,
                )
            )
            best_record["matched_random"] = matched_random_search(
                executor,
                cfg,
                condition,
                train_cases,
                val_cases,
                test_cases,
                rng,
                logger,
            )

        if "no_scaffold_reference" not in best_record:
            logger.info(
                jline(
                    "condition",
                    str(condition["id"]),
                    "recover_no_scaffold_reference_start",
                    test_cases=len(test_cases),
                )
            )
            best_record["no_scaffold_reference"] = evaluate_no_scaffold(
                executor,
                cfg,
                condition,
                test_cases,
            )

    if "hand_reference" not in best_record:
        logger.info(
            jline(
                "condition",
                str(condition["id"]),
                "recover_hand_reference_start",
                test_cases=len(test_cases),
            )
        )
        best_record["hand_reference"] = evaluate_hand_reference(
            _WORKER_EXP11,
            _WORKER_EXP15,
            _WORKER_EXP19,
            cfg,
            condition,
            test_cases,
        )

    if "global_reference" not in best_record:
        logger.info(
            jline(
                "condition",
                str(condition["id"]),
                "recover_global_reference_start",
                test_cases=len(test_cases),
            )
        )
        best_record["global_reference"] = global_reference_eval(
            _WORKER_EXP19,
            cfg,
            condition,
            test_cases,
        )

    write_json(str(condition_dir / "best_genome.json"), best_record)
    summary = condition_summary(condition, best_record, cfg)
    write_json(str(condition_dir / "summary.json"), summary)

    logger.info(
        jline(
            "condition",
            str(condition["id"]),
            "recover_finish",
            evolved_success=float(summary["evolved_strong_success_rate"]),
            random_success=float(summary["random_strong_success_rate"]),
            gain=float(summary["matched_evolution_gain"]),
            robust=bool(summary["robust_discovery_pass"]),
            mode=str(summary["failure_mode"]),
        )
    )

    return summary


def make_summary_header() -> list[str]:
    return [
        "condition_id",
        "group",
        "task",
        "task_distribution",
        "library",
        "ablation",
        "search_mode",
        "selection_mode",
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
        "prune_count",
        "stabilize_count",
        "repair_loop_completeness",
        "program_interpretability",
        "prior_leakage_score",
        "train_fitness",
        "val_fitness",
        "evolved_fitness",
        "random_fitness",
        "no_scaffold_fitness",
        "hand_fitness",
        "global_fitness",
        "train_val_gap",
        "generalization_gap",
        "matched_evolution_gain",
        "strong_success_gain_over_random",
        "random_search_equivalent",
        "robust_discovery_pass",
        "gated_growth_ratio",
        "prune_survival_ratio",
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
        "evolved_mean_gated_growth",
        "evolved_mean_trace_stabilization",
        "evolved_mean_prune_amount",
        "evolved_mean_survived_growth",
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
        "random_mean_gated_growth",
        "random_worst_case_strong_success",
        "random_weak_success_rate",
        "random_strong_success_rate",
        "random_strict_success_rate",
        "no_scaffold_strong_success_rate",
        "hand_strong_success_rate",
        "global_strong_success_rate",
        "failure_mode",
    ]


def main() -> int:
    global _WORKER_EXP11
    global _WORKER_EXP15
    global _WORKER_EXP19

    args = parse_args()
    root = repo_root()
    _WORKER_EXP11 = load_exp11(root)
    _WORKER_EXP15 = load_exp15(root)
    _WORKER_EXP19 = load_exp19(root)

    config_path = root / "config" / "tests" / "exp_20.yaml"
    cfg = read_yaml(str(config_path))

    meta = build_meta(
        params=cfg,
        env=str(root / "uv.lock"),
        script=str(Path(__file__).resolve()),
        src=str(root / "src"),
    )

    output_root = root / cfg["output"]["root"]
    run_dir, resumed = resolve_run_dir(output_root, meta, args)
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not resumed:
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
            resumed=bool(resumed),
        )
    )

    conditions = list(cfg["conditions"])
    remaining_conditions = []

    for condition in conditions:
        if condition_is_complete(run_dir, condition):
            continue

        remaining_conditions.append(condition)

    total_units = int(cfg["evolution"]["generations"]) * len(remaining_conditions)

    if total_units == 0:
        total_units = 1

    progress = Progress(logger=logger, name=cfg["name"], total=total_units)
    progress.start()

    rows = []

    for condition in conditions:
        if condition_is_complete(run_dir, condition):
            logger.info(
                jline(
                    "condition",
                    str(condition["id"]),
                    "skip_completed",
                    summary=str(condition_summary_path(run_dir, condition)),
                )
            )
            rows.append(load_condition_summary(run_dir, condition))
            continue

        if resumed and condition_has_recoverable_best(run_dir, condition):
            logger.info(
                jline(
                    "condition",
                    str(condition["id"]),
                    "resume_partial",
                    best=str(condition_best_record_path(run_dir, condition)),
                )
            )
            rows.append(
                finalize_recoverable_condition(cfg, condition, root, run_dir, logger)
            )
            continue

        logger.info(jline("condition", str(condition["id"]), "start"))
        rows.append(evolve_condition(cfg, condition, root, run_dir, logger, progress))

    progress.finish()

    summary_header = make_summary_header()

    write_csv(
        str(run_dir / "constrained_grammar_summary.csv"),
        [[x.get(k) for k in summary_header] for x in rows],
        header=summary_header,
    )
    write_json(str(run_dir / "constrained_grammar_summary.json"), rows)

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
        "resumed": bool(resumed),
        "robust_discovery_count": int(robust_count),
        "random_search_equivalence_count": int(random_equiv_count),
        "summary_path": "constrained_grammar_summary.csv",
        "summary_json_path": "constrained_grammar_summary.json",
        "figures": [
            "figures/evolved_vs_matched_random_success.png",
            "figures/matched_evolution_gain.png",
            "figures/train_test_gap.png",
            "figures/gated_growth_ratio.png",
        ],
        "success": True,
    }

    write_json(str(run_dir / "summary.json"), run_summary)

    logger.info(
        jline("run", cfg["name"], "finish", run_dir=str(run_dir), resumed=bool(resumed))
    )
    audit.finish_success()

    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
