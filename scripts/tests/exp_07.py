from pathlib import Path
from copy import deepcopy
import importlib.util
from statistics import mean, variance

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


def load_exp06_module(root: Path):
    path = root / "scripts" / "tests" / "exp_06.py"
    spec = importlib.util.spec_from_file_location("cibo_exp_06_runtime", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def condition_to_exp06_condition(condition: dict) -> dict:
    return {
        "id": str(condition["id"]),
        "group": str(condition["group"]),
        "operator": str(condition["operator"]),
        "mechanism": str(condition["mechanism"]),
        "theta_repair": float(condition["theta_repair"]),
        "k": int(condition["k"]),
        "m": int(condition["m"]),
        "gamma": float(condition["gamma"]),
        "budget": bool(condition["budget"]),
    }


def condition_to_perturbation(cfg: dict, condition: dict) -> list[dict]:
    base = cfg["base_perturbation"]
    return [
        {
            "type": "random_noise",
            "time": int(base["noise_time"]),
            "strength": float(condition.get("noise_strength", base["noise_strength"])),
            "scope": str(base["noise_scope"]),
        },
        {
            "type": "local_gap_attack",
            "time": int(base["gap_time"]),
            "length": int(condition.get("gap_length", base["gap_length"])),
            "side": str(base["gap_side"]),
        },
    ]


def build_condition_cfg(cfg: dict, condition: dict) -> dict:
    out = deepcopy(cfg)
    out["perturbation"] = {"schedule": condition_to_perturbation(cfg, condition)}
    out["output"]["save_initial_snapshot"] = bool(
        cfg["output"]["save_initial_snapshot"]
    )
    out["output"]["save_perturbation_snapshot"] = bool(
        cfg["output"]["save_perturbation_snapshot"]
    )
    out["output"]["save_final_snapshot"] = bool(cfg["output"]["save_final_snapshot"])
    return out


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


def aggregate_by_condition(summaries: list[dict], cfg: dict) -> list[dict]:
    groups = {}

    for s in summaries:
        groups.setdefault(str(s["condition_id"]), []).append(s)

    rows = []

    for condition_id, items in groups.items():
        balanced = [float(x["final_balanced_integrity"]) for x in items]
        tpr = [float(x["final_target_true_positive_rate"]) for x in items]
        fpr = [float(x["final_outside_false_positive_rate"]) for x in items]
        expansion = [float(x["expansion_total"]) for x in items]
        gap = [float(x["gap_closure_total"]) for x in items]
        repair_precision = [float(x["repair_precision"]) for x in items]
        false_repair = [float(x["false_repair_rate"]) for x in items]
        strong = [1 if is_strong_success(x, cfg) else 0 for x in items]
        strict = [1 if is_strict_success(x, cfg) else 0 for x in items]
        failures = [1 if is_failure(x) else 0 for x in items]
        first = items[0]

        rows.append(
            {
                "condition_id": condition_id,
                "group": str(first["group"]),
                "operator": str(first["operator"]),
                "mechanism": str(first["mechanism"]),
                "theta_repair": float(first["theta_repair"]),
                "k": int(first["k"]),
                "m": int(first["m"]),
                "gamma": float(first["gamma"]),
                "budget": bool(first["budget"]),
                "seed_count": len(items),
                "strong_success_rate": float(sum(strong) / len(strong)),
                "strict_success_rate": float(sum(strict) / len(strict)),
                "failure_rate": float(sum(failures) / len(failures)),
                "mean_final_balanced_integrity": float(mean(balanced)),
                "var_final_balanced_integrity": safe_var(balanced),
                "worst_final_balanced_integrity": float(min(balanced)),
                "best_final_balanced_integrity": float(max(balanced)),
                "mean_final_tpr": float(mean(tpr)),
                "worst_final_tpr": float(min(tpr)),
                "mean_final_fpr": float(mean(fpr)),
                "worst_final_fpr": float(max(fpr)),
                "mean_expansion_total": float(mean(expansion)),
                "mean_gap_closure_total": float(mean(gap)),
                "mean_repair_precision": float(mean(repair_precision)),
                "mean_false_repair_rate": float(mean(false_repair)),
            }
        )

    rows.sort(
        key=lambda x: (
            -x["strong_success_rate"],
            -x["mean_final_balanced_integrity"],
            x["mean_final_fpr"],
        )
    )
    return rows


def make_aggregate_plots(rows: list[dict], figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [str(x["condition_id"]) for x in rows]
    success = [float(x["strong_success_rate"]) for x in rows]
    balanced = [float(x["mean_final_balanced_integrity"]) for x in rows]
    fpr = [float(x["mean_final_fpr"]) for x in rows]
    tpr = [float(x["mean_final_tpr"]) for x in rows]

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(labels, success)
    ax.set_title("Experiment 07 strong success rate")
    ax.set_xlabel("condition")
    ax.set_ylabel("success rate")
    ax.tick_params(axis="x", rotation=75)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(str(figures_dir / "strong_success_rate.png"), dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(labels, balanced)
    ax.set_title("Experiment 07 mean final balanced integrity")
    ax.set_xlabel("condition")
    ax.set_ylabel("mean final balanced integrity")
    ax.tick_params(axis="x", rotation=75)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(str(figures_dir / "mean_final_balanced_integrity.png"), dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(1, 1, 1)

    for item in rows:
        ax.scatter(float(item["mean_final_fpr"]), float(item["mean_final_tpr"]))
        ax.text(
            float(item["mean_final_fpr"]),
            float(item["mean_final_tpr"]),
            str(item["condition_id"]),
            fontsize=7,
        )

    ax.set_title("Experiment 07 mean TPR-FPR plane")
    ax.set_xlabel("mean final outside FPR")
    ax.set_ylabel("mean final TPR")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(str(figures_dir / "mean_tpr_fpr_plane.png"), dpi=160)
    plt.close(fig)


def main() -> int:
    root = repo_root()
    config_path = root / "config" / "tests" / "exp_07.yaml"
    cfg = read_yaml(str(config_path))
    exp06 = load_exp06_module(root)

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
            exp06_condition = condition_to_exp06_condition(condition)
            condition_cfg = build_condition_cfg(cfg, condition)
            logger.info(
                jline("condition", str(condition["id"]), "start", seed=int(seed))
            )
            result = exp06.run_condition(
                condition_cfg, exp06_condition, seed, run_dir, logger, progress
            )
            result["summary"]["purpose"] = str(condition["purpose"])
            result["summary"]["noise_strength"] = float(
                condition.get(
                    "noise_strength", cfg["base_perturbation"]["noise_strength"]
                )
            )
            result["summary"]["gap_length"] = int(
                condition.get("gap_length", cfg["base_perturbation"]["gap_length"])
            )
            all_runs.append(result)

    progress.finish()

    summaries = [item["summary"] for item in all_runs]
    aggregate_rows = aggregate_by_condition(summaries, cfg)

    summary_header = [
        "condition_id",
        "group",
        "purpose",
        "seed",
        "operator",
        "mechanism",
        "theta_preserve",
        "theta_repair",
        "k",
        "m",
        "gamma",
        "budget",
        "noise_strength",
        "gap_length",
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

    aggregate_header = [
        "condition_id",
        "group",
        "operator",
        "mechanism",
        "theta_repair",
        "k",
        "m",
        "gamma",
        "budget",
        "seed_count",
        "strong_success_rate",
        "strict_success_rate",
        "failure_rate",
        "mean_final_balanced_integrity",
        "var_final_balanced_integrity",
        "worst_final_balanced_integrity",
        "best_final_balanced_integrity",
        "mean_final_tpr",
        "worst_final_tpr",
        "mean_final_fpr",
        "worst_final_fpr",
        "mean_expansion_total",
        "mean_gap_closure_total",
        "mean_repair_precision",
        "mean_false_repair_rate",
    ]

    summary_rows = [[s.get(k) for k in summary_header] for s in summaries]
    aggregate_table = [[s.get(k) for k in aggregate_header] for s in aggregate_rows]

    write_csv(str(run_dir / "runs_summary.csv"), summary_rows, header=summary_header)
    write_json(str(run_dir / "runs_summary.json"), summaries)
    write_csv(
        str(run_dir / "robustness_summary.csv"),
        aggregate_table,
        header=aggregate_header,
    )
    write_json(str(run_dir / "robustness_summary.json"), aggregate_rows)

    if bool(cfg["output"]["make_plot"]):
        make_aggregate_plots(aggregate_rows, figures_dir)

    basin_success = [
        x for x in aggregate_rows if float(x["strong_success_rate"]) >= 0.8
    ]
    basin_strict = [x for x in aggregate_rows if float(x["strict_success_rate"]) >= 0.8]

    run_summary = {
        "name": cfg["name"],
        "run_dir": str(run_dir),
        "fingerprint": meta.get("fingerprint"),
        "condition_count": len(conditions),
        "seed_count": len(seeds),
        "run_count": len(summaries),
        "robust_success_condition_count": len(basin_success),
        "strict_success_condition_count": len(basin_strict),
        "best_condition": aggregate_rows[0]["condition_id"] if aggregate_rows else None,
        "runs_summary_path": "runs_summary.csv",
        "runs_summary_json_path": "runs_summary.json",
        "robustness_summary_path": "robustness_summary.csv",
        "robustness_summary_json_path": "robustness_summary.json",
        "figures": [
            "figures/strong_success_rate.png",
            "figures/mean_final_balanced_integrity.png",
            "figures/mean_tpr_fpr_plane.png",
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
