from pathlib import Path
from collections import defaultdict
from statistics import mean, variance
import importlib.util
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


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_exp11(root: Path):
    return load_module(
        root / "scripts" / "tests" / "exp_11.py", "cibo_exp_11_runtime_for_exp_13"
    )


def load_exp12(root: Path):
    return load_module(
        root / "scripts" / "tests" / "exp_12.py", "cibo_exp_12_runtime_for_exp_13"
    )


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


def make_population(
    condition: dict, operator_count: int, base_theta: float, rng: np.random.Generator
) -> dict:
    population = str(condition["population"])

    theta = np.full(operator_count, base_theta, dtype=np.float64)
    sensitivity = np.ones(operator_count, dtype=np.float64)

    if population in {"threshold_heterogeneous", "mixed_heterogeneous"}:
        jitter = float(condition.get("threshold_jitter", 0.05))
        theta += rng.normal(0.0, jitter, size=operator_count)
        theta = np.clip(theta, 0.50, 0.95)

    if population in {"scaffold_heterogeneous", "mixed_heterogeneous"}:
        jitter = float(condition.get("sensitivity_jitter", 0.10))
        sensitivity += rng.normal(0.0, jitter, size=operator_count)
        sensitivity = np.clip(sensitivity, 0.70, 1.30)

    return {"theta": theta, "sensitivity": sensitivity}


def noisy_signal(
    clean: float,
    shared: float,
    sigma: float,
    rho: float,
    bias: float,
    sensitivity: float,
    rng: np.random.Generator,
) -> float:
    independent = rng.normal(0.0, 1.0)
    noise = sigma * (
        math.sqrt(max(0.0, 1.0 - rho)) * independent + math.sqrt(max(0.0, rho)) * shared
    )
    return float(np.clip(clean * sensitivity + noise + bias, 0.0, 1.0))


def apply_perturbation(
    exp12, canvas: np.ndarray, reference: np.ndarray, condition: dict, seed: int, t: int
) -> int:
    return exp12.apply_perturbation(canvas, reference, condition, seed, t)


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
    damage_time = int(condition["damage_time"])
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
        "mean_signal_accuracy": float(
            mean([float(row[idx["signal_accuracy"]]) for row in rows])
        ),
        "mean_error_rate": float(mean([float(row[idx["error_rate"]]) for row in rows])),
        "mean_false_confirmation_rate": float(
            mean([float(row[idx["shared_false_confirmation_rate"]]) for row in rows])
        ),
        "mean_independent_noise_rejection_rate": float(
            mean([float(row[idx["noise_rejection_rate"]]) for row in rows])
        ),
    }


def run_condition(
    exp11,
    exp12,
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

    reference, start, goal, marker, direction = exp12.apply_scaffold(
        exp11, cfg, condition
    )
    canvas = reference.copy()
    trace = np.zeros((height, width), dtype=np.float64)
    history = np.zeros((height, width, int(cfg["repair"]["k"])), dtype=np.uint8)

    operator_count = int(cfg["operator"]["count"])
    operators = init_operators(operator_count, height, width, rng)
    population = make_population(
        condition, operator_count, float(cfg["repair"]["theta_repair"]), rng
    )

    sigma = float(condition["sigma"])
    rho = float(condition["rho"])
    bias = float(condition["bias"])

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
        "conflict_attempts",
        "clean_signal_mean",
        "noisy_signal_mean",
        "signal_accuracy",
        "error_rate",
        "shared_false_confirmation_rate",
        "noise_rejection_rate",
        "positive_trace_mass",
        "negative_trace_mass",
        "perturbed",
    ]

    for t in range(steps + 1):
        perturbed = apply_perturbation(exp12, canvas, reference, condition, seed, t)

        proposals = defaultdict(list)
        proposal_meta = defaultdict(list)

        clean_values = []
        noisy_values = []
        signal_match = 0
        signal_total = 0
        error_total = 0
        false_signal_total = 0
        false_confirm_total = 0
        noise_reject_total = 0
        repair_pass_total = 0
        repair_reject_total = 0

        if t < steps:
            move_operators(operators, height, width, rng)
            shared_noise = rng.normal(0.0, 1.0, size=(height, width))

            for i, pos in enumerate(operators):
                x = int(pos[0])
                y = int(pos[1])
                clean = exp12.scaffolded_support(
                    canvas, trace, marker, direction, x, y, cfg, condition
                )
                theta_i = float(population["theta"][i])
                perceived = noisy_signal(
                    clean,
                    float(shared_noise[x, y]),
                    sigma,
                    rho,
                    bias,
                    float(population["sensitivity"][i]),
                    rng,
                )

                clean_high = clean >= theta_i
                noisy_high = perceived >= theta_i

                clean_values.append(float(clean))
                noisy_values.append(float(perceived))

                signal_total += 1
                if clean_high == noisy_high:
                    signal_match += 1
                else:
                    error_total += 1

                temporal_pass, temporal_score = update_sliding(
                    history,
                    perceived,
                    theta_i,
                    x,
                    y,
                    t,
                    int(cfg["repair"]["k"]),
                    int(cfg["repair"]["m"]),
                )

                if int(canvas[x, y]) == 0:
                    if noisy_high and not clean_high:
                        false_signal_total += 1

                    if temporal_pass and not clean_high:
                        false_confirm_total += 1

                    if clean_high and not temporal_pass:
                        noise_reject_total += 1

                result = choose_action(canvas, perceived, temporal_pass, x, y, cfg, rng)

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
                int(operator_count),
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
                float(mean(clean_values)) if clean_values else 0.0,
                float(mean(noisy_values)) if noisy_values else 0.0,
                float(signal_match / max(1, signal_total)),
                float(error_total / max(1, signal_total)),
                float(false_confirm_total / max(1, false_signal_total)),
                float(noise_reject_total / max(1, signal_total)),
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
            "anchor": str(condition["anchor"]),
            "path_class": str(condition["path_class"]),
            "path_type": str(condition["path_type"]),
            "perturbation_type": str(condition["perturbation_type"]),
            "scaffold": str(condition["scaffold"]),
            "sigma": float(sigma),
            "rho": float(rho),
            "bias": float(bias),
            "population": str(condition["population"]),
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
            mean_signal_accuracy=summary["mean_signal_accuracy"],
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
        if float(summary["rho"]) >= 0.75 and float(summary["sigma"]) > 0:
            return "correlated_conservative_collapse"
        return "independent_noise_under_repair"

    if float(summary["final_open_cost_factor"]) > float(
        cfg["metrics"]["strong_cost_factor"]
    ):
        if float(summary["rho"]) >= 0.75:
            return "correlated_false_confirmation"
        return "costly_functional_success"

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
        accuracy = [float(x["mean_signal_accuracy"]) for x in items]
        error = [float(x["mean_error_rate"]) for x in items]
        false_confirm = [float(x["mean_false_confirmation_rate"]) for x in items]
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
                "anchor": str(first["anchor"]),
                "path_class": str(first["path_class"]),
                "path_type": str(first["path_type"]),
                "scaffold": str(first["scaffold"]),
                "sigma": float(first["sigma"]),
                "rho": float(first["rho"]),
                "bias": float(first["bias"]),
                "population": str(first["population"]),
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
                "mean_signal_accuracy": float(mean(accuracy)),
                "mean_error_rate": float(mean(error)),
                "mean_false_confirmation_rate": float(mean(false_confirm)),
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


def diversity_rescue_score(rows: list[dict]) -> float | None:
    hom = [x for x in rows if str(x["condition_id"]) == "H_C2_COR_020_075"]
    het = [x for x in rows if str(x["condition_id"]) == "I_C2_COR_020_075_HET"]

    if not hom or not het:
        return None

    return float(het[0]["strong_success_rate"]) - float(hom[0]["strong_success_rate"])


def make_plots(rows: list[dict], figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [str(x["group"]) for x in rows]
    strong = [float(x["strong_success_rate"]) for x in rows]
    acc = [float(x["mean_signal_accuracy"]) for x in rows]
    false_confirm = [float(x["mean_false_confirmation_rate"]) for x in rows]

    fig = plt.figure(figsize=(11, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(labels, strong)
    ax.set_title("Experiment 13 strong functional success rate")
    ax.set_xlabel("condition")
    ax.set_ylabel("strong success rate")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(str(figures_dir / "strong_functional_success_rate.png"), dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(11, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(labels, acc)
    ax.set_title("Experiment 13 mean signal accuracy")
    ax.set_xlabel("condition")
    ax.set_ylabel("signal accuracy")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(str(figures_dir / "mean_signal_accuracy.png"), dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(11, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(labels, false_confirm)
    ax.set_title("Experiment 13 false confirmation rate")
    ax.set_xlabel("condition")
    ax.set_ylabel("false confirmation rate")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(str(figures_dir / "false_confirmation_rate.png"), dpi=160)
    plt.close(fig)


def main() -> int:
    root = repo_root()
    config_path = root / "config" / "tests" / "exp_13.yaml"
    cfg = read_yaml(str(config_path))
    exp11 = load_exp11(root)
    exp12 = load_exp12(root)

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
                run_condition(
                    exp11, exp12, cfg, condition, seed, run_dir, logger, progress
                )
            )

    progress.finish()

    summaries = [item["summary"] for item in all_runs]
    condition_rows = aggregate_by_condition(summaries, cfg)

    run_header = [
        "condition_id",
        "group",
        "seed",
        "anchor",
        "path_class",
        "path_type",
        "scaffold",
        "sigma",
        "rho",
        "bias",
        "population",
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
        "mean_signal_accuracy",
        "mean_error_rate",
        "mean_false_confirmation_rate",
        "mean_independent_noise_rejection_rate",
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
        "anchor",
        "path_class",
        "path_type",
        "scaffold",
        "sigma",
        "rho",
        "bias",
        "population",
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
        "mean_signal_accuracy",
        "mean_error_rate",
        "mean_false_confirmation_rate",
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
        str(run_dir / "error_summary.csv"),
        [[s.get(k) for k in condition_header] for s in condition_rows],
        header=condition_header,
    )
    write_json(str(run_dir / "error_summary.json"), condition_rows)

    if bool(cfg["output"]["make_plot"]):
        make_plots(condition_rows, figures_dir)

    run_summary = {
        "name": cfg["name"],
        "run_dir": str(run_dir),
        "fingerprint": meta.get("fingerprint"),
        "condition_count": len(conditions),
        "seed_count": len(seeds),
        "run_count": len(summaries),
        "diversity_rescue_score": diversity_rescue_score(condition_rows),
        "runs_summary_path": "runs_summary.csv",
        "runs_summary_json_path": "runs_summary.json",
        "error_summary_path": "error_summary.csv",
        "error_summary_json_path": "error_summary.json",
        "figures": [
            "figures/strong_functional_success_rate.png",
            "figures/mean_signal_accuracy.png",
            "figures/false_confirmation_rate.png",
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
