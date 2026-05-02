from pathlib import Path
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
        root / "scripts" / "tests" / "exp_11.py", "cibo_exp_11_runtime_for_exp_16"
    )


def load_exp15(root: Path):
    return load_module(
        root / "scripts" / "tests" / "exp_15.py", "cibo_exp_15_runtime_for_exp_16"
    )


def init_worker(root_text: str) -> None:
    global _WORKER_EXP11
    global _WORKER_EXP15
    root = Path(root_text)
    _WORKER_EXP11 = load_exp11(root)
    _WORKER_EXP15 = load_exp15(root)


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


def aggregate_fitness(scores: list[float], family: str, alpha: float) -> float:
    if not scores:
        return 0.0

    if family == "mean":
        return float(mean(scores))

    if family == "worst":
        return float(min(scores))

    if family == "cvar":
        count = max(1, int(round(len(scores) * alpha)))
        selected = sorted(scores)[:count]
        return float(mean(selected))

    raise ValueError(f"unknown fitness family: {family}")


def evaluate_genome_with_modules(
    exp11, exp15, cfg: dict, condition: dict, genome: dict, cases: list[dict]
) -> dict:
    trials = [
        exp15.run_trial(
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

    scores = [base_score(trial, cfg) for trial in trials]
    fitness_value = aggregate_fitness(
        scores, str(condition["fitness_family"]), float(cfg["evolution"]["cvar_alpha"])
    )

    return summarize_eval(fitness_value, scores, trials)


def evaluate_worker(args: tuple) -> dict:
    cfg, condition, genome, cases = args
    return evaluate_genome_with_modules(
        _WORKER_EXP11, _WORKER_EXP15, cfg, condition, genome, cases
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


def summarize_eval(
    fitness_value: float, scores: list[float], trials: list[dict]
) -> dict:
    return {
        "fitness": float(fitness_value),
        "mean_base_score": float(mean(scores)) if scores else 0.0,
        "worst_base_score": float(min(scores)) if scores else 0.0,
        "cvar_base_score": cvar_value(scores, 0.25),
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
        "worst_case_strong_success": float(worst_subgroup_success(trials)),
        "trials": trials,
    }


def cvar_value(scores: list[float], alpha: float) -> float:
    if not scores:
        return 0.0

    count = max(1, int(round(len(scores) * alpha)))
    return float(mean(sorted(scores)[:count]))


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


def worst_subgroup_success(trials: list[dict]) -> float:
    groups = {}

    for trial in trials:
        key = (str(trial["variant"]), int(trial["offset"]), bool(trial["noise"]))
        groups.setdefault(key, []).append(trial)

    if not groups:
        return 0.0

    rates = []

    for items in groups.values():
        strong = [
            1
            if int(x["final_connectivity"]) == 1
            and float(x["final_open_cost_factor"]) <= 1.50
            else 0
            for x in items
        ]
        rates.append(float(sum(strong) / max(1, len(strong))))

    return float(min(rates))


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


def tournament(
    population: list[dict], scores: list[float], size: int, rng: np.random.Generator
) -> dict:
    idx = rng.choice(len(population), size=size, replace=False)
    best = max(idx, key=lambda i: scores[int(i)])
    return population[int(best)]


def evolved_sample_efficiency(history_rows: list[list], cfg: dict) -> int | None:
    idx = {
        "generation": 0,
        "best_fitness": 1,
        "mean_fitness": 2,
        "best_train_connectivity": 3,
        "best_train_cost": 4,
        "best_train_outside": 5,
        "best_train_worst_case": 6,
    }

    pop_size = int(cfg["evolution"]["population_size"])

    for row in history_rows:
        if (
            float(row[idx["best_train_connectivity"]])
            >= float(cfg["metrics"]["strong_connectivity_threshold"])
            and float(row[idx["best_train_cost"]])
            <= float(cfg["metrics"]["strong_cost_factor"])
            and float(row[idx["best_train_outside"]])
            <= float(cfg["metrics"]["strict_outside_open_rate"])
        ):
            return int((int(row[idx["generation"]]) + 1) * pop_size)

    return None


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
    representation = str(condition["representation"])

    best_train = None
    best_genome = None
    evaluated = 0

    while evaluated < budget:
        n = min(chunk_size, budget - evaluated)
        candidates = [_WORKER_SAFE_RANDOM_GENOME(representation, rng) for _ in range(n)]
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


def _WORKER_SAFE_RANDOM_GENOME(representation: str, rng: np.random.Generator) -> dict:
    return random_genome(representation, rng)


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


def apply_representation(g: dict, representation: str) -> dict:
    out = dict(g)

    if representation == "R1":
        out["w_radius"] = min(int(out["w_radius"]), 1)
        out["w_prob"] = min(float(out["w_prob"]), 0.35)
        return out

    if representation == "R3":
        return out

    raise ValueError(f"unknown representation: {representation}")


def mutate_genome(
    g: dict, representation: str, rate: float, rng: np.random.Generator
) -> dict:
    out = dict(g)

    for key in genome_keys():
        if rng.random() >= rate:
            continue

        if key in {"w_radius", "m_radius"}:
            out[key] = int(np.clip(int(out[key]) + int(rng.choice([-1, 1])), 0, 4))
        elif key == "w_shape":
            out[key] = int(rng.integers(0, 3))
        elif key in {"m_decay", "d_decay", "p_decay"}:
            out[key] = float(
                np.clip(float(out[key]) + rng.normal(0.0, 0.08), 0.50, 0.98)
            )
        else:
            out[key] = float(np.clip(float(out[key]) + rng.normal(0.0, 0.15), 0.0, 1.0))

    return apply_representation(out, representation)


def crossover(a: dict, b: dict, representation: str, rng: np.random.Generator) -> dict:
    g = {key: a[key] if rng.random() < 0.5 else b[key] for key in genome_keys()}
    return apply_representation(g, representation)


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

    scores = [base_score(x, cfg) for x in trials]
    return summarize_eval(float(mean(scores)), scores, trials)


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
    )


def failure_mode(summary: dict, cfg: dict) -> str:
    if robust_discovery_pass(summary, cfg):
        return "robust_evolutionary_discovery"

    if float(summary["matched_evolution_gain"]) <= float(
        cfg["evolution"]["discovery_margin"]
    ):
        return "random_search_equivalence"

    if float(summary["train_fitness"]) > float(summary["evolved_fitness"]) + float(
        cfg["metrics"]["robust_max_generalization_gap"]
    ):
        return "train_test_collapse"

    if (
        str(summary["fitness_family"]) in {"worst", "cvar"}
        and float(summary["evolved_mean_connectivity"]) < 0.50
    ):
        return "conservative_distributional_collapse"

    if float(summary["evolved_mean_cost"]) > float(
        cfg["metrics"]["strong_cost_factor"]
    ):
        return "over_opening_exploit"

    if (
        bool(summary["noise_stress"])
        and float(summary["evolved_strong_success_rate"]) < 0.90
    ):
        return "noise_fragility"

    if float(summary["evolved_mean_connectivity"]) < float(
        cfg["metrics"]["strong_connectivity_threshold"]
    ):
        return "no_discovery"

    return "mixed_failure"


def condition_summary(condition: dict, record: dict, cfg: dict) -> dict:
    out = {
        "condition_id": str(condition["id"]),
        "group": str(condition["group"]),
        "task": str(condition["task"]),
        "task_distribution": str(condition["task_distribution"]),
        "path_type": str(condition["path_type"]),
        "fitness_family": str(condition["fitness_family"]),
        "representation": str(condition["representation"]),
        "noise_stress": bool(condition.get("noise_stress", False)),
        "discovery_generation": int(record["generation"]),
        "sample_efficiency": record["sample_efficiency"],
        "matched_random_budget": int(record["matched_random"]["budget"]),
        "best_genome": record["genome"],
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
    out["robust_discovery_score"] = robust_discovery_score(out, cfg)
    out["robust_discovery_pass"] = robust_discovery_pass(out, cfg)
    out["failure_mode"] = failure_mode(out, cfg)

    return out


def robust_discovery_score(summary: dict, cfg: dict) -> float:
    success = float(summary["evolved_strong_success_rate"])
    gain = max(0.0, float(summary["matched_evolution_gain"]))
    gap_penalty = abs(float(summary["generalization_gap"]))
    cost_penalty = max(
        0.0,
        float(summary["evolved_mean_cost"])
        - float(cfg["metrics"]["strong_cost_factor"]),
    )
    outside_penalty = max(
        0.0,
        float(summary["evolved_mean_outside"])
        - float(cfg["metrics"]["strict_outside_open_rate"]),
    )

    return float(success + gain - 0.50 * gap_penalty - cost_penalty - outside_penalty)


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
        [str(x) for x in cfg["run"]["train_variants"]],
        [False],
    )

    test_cases = make_cases(
        [int(x) for x in cfg["run"]["test_seeds"]],
        [int(x) for x in cfg["run"]["test_offsets"]],
        [str(x) for x in cfg["run"]["test_variants"]],
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
            matched_random_budget=pop_size * generations,
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
                child = crossover(a, b, representation, rng)
                child = mutate_genome(child, representation, mutation_rate, rng)
                next_pop.append(child)

            population = next_pop
            progress.step(1)

        sample_efficiency = evolved_sample_efficiency(history_rows, cfg)

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
                "matched_random_start",
                budget=pop_size * generations,
                train_cases=len(train_cases),
                test_cases=len(test_cases),
            )
        )
        matched_random = matched_random_search(
            executor, cfg, condition, train_cases, test_cases, rng, logger
        )

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
    best_record["matched_random"] = matched_random
    best_record["no_scaffold_reference"] = no_ref
    best_record["hand_reference"] = hand_ref
    best_record["global_reference"] = global_ref
    best_record["sample_efficiency"] = sample_efficiency

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
    worst = [float(x["evolved_worst_case_strong_success"]) for x in rows]
    score = [float(x["robust_discovery_score"]) for x in rows]

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    x = np.arange(len(labels))
    ax.bar(x - 0.2, evolved, width=0.4, label="evolved")
    ax.bar(x + 0.2, random_ref, width=0.4, label="matched random")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Experiment 16 evolved vs matched random success")
    ax.set_xlabel("condition")
    ax.set_ylabel("strong success rate")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(str(figures_dir / "evolved_vs_matched_random_success.png"), dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(labels, gain)
    ax.set_title("Experiment 16 matched evolution gain")
    ax.set_xlabel("condition")
    ax.set_ylabel("evolved test fitness - matched random test fitness")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(str(figures_dir / "matched_evolution_gain.png"), dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(labels, gap)
    ax.set_title("Experiment 16 train-test generalization gap")
    ax.set_xlabel("condition")
    ax.set_ylabel("train fitness - held-out fitness")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(str(figures_dir / "generalization_gap.png"), dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(labels, worst)
    ax.set_title("Experiment 16 worst-case held-out strong success")
    ax.set_xlabel("condition")
    ax.set_ylabel("worst subgroup strong success")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(str(figures_dir / "worst_case_heldout_success.png"), dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(labels, score)
    ax.set_title("Experiment 16 robust discovery score")
    ax.set_xlabel("condition")
    ax.set_ylabel("robust discovery score")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(str(figures_dir / "robust_discovery_score.png"), dpi=160)
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
    config_path = root / "config" / "tests" / "exp_16.yaml"
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
        "robust_discovery_score",
        "robust_discovery_pass",
        "evolved_mean_connectivity",
        "evolved_mean_stability",
        "evolved_mean_cost",
        "evolved_mean_outside",
        "evolved_mean_false",
        "evolved_mean_path_tpr",
        "evolved_mean_open_precision",
        "evolved_mean_false_open_rate",
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
        str(run_dir / "distributional_evolution_summary.csv"),
        [[x.get(k) for k in summary_header] for x in rows],
        header=summary_header,
    )
    write_json(str(run_dir / "distributional_evolution_summary.json"), rows)

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
        "train_case_count": len(
            make_cases(
                cfg["run"]["train_seeds"],
                cfg["run"]["train_offsets"],
                cfg["run"]["train_variants"],
                [False],
            )
        ),
        "test_case_count_clean": len(
            make_cases(
                cfg["run"]["test_seeds"],
                cfg["run"]["test_offsets"],
                cfg["run"]["test_variants"],
                [False],
            )
        ),
        "robust_discovery_count": int(robust_count),
        "random_search_equivalence_count": int(random_equiv_count),
        "summary_path": "distributional_evolution_summary.csv",
        "summary_json_path": "distributional_evolution_summary.json",
        "figures": [
            "figures/evolved_vs_matched_random_success.png",
            "figures/matched_evolution_gain.png",
            "figures/generalization_gap.png",
            "figures/worst_case_heldout_success.png",
            "figures/robust_discovery_score.png",
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
