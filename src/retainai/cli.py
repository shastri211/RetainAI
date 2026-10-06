"""Command-line entry point: ``python -m retainai <command>`` or ``retainai <command>``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from retainai import config
from retainai.data.clean import DataContractError, clean_telco
from retainai.data.load import dataset_sha256, load_raw
from retainai.data.validate import render_text, validate_raw
from retainai.features.build import DEFAULT_SPEC, derive_features, model_ready_frame
from retainai.explain.linear import LinearExplainer
from retainai.fairness.audit import run_fairness_audit
from retainai.io import write_csv, write_json
from retainai.models.baseline import run_baseline
from retainai.models.registry import build_metadata, load_artifacts, save_artifacts
from retainai.scoring.bands import risk_band_report
from retainai.scoring.risk import records_to_frame, score_cleaned
from retainai.value.proxy import band_summary, top_revenue_at_risk, value_table


def _cmd_validate(args: argparse.Namespace) -> int:
    path = Path(args.input)
    report = validate_raw(load_raw(path))
    report.dataset_sha256 = dataset_sha256(path)
    print(render_text(report))
    if args.report:
        write_json(report.to_dict(), args.report)
        print(f"Report written: {args.report}")
    return 0 if report.passed else 1


def _cmd_prepare(args: argparse.Namespace) -> int:
    path = Path(args.input)
    raw = load_raw(path)
    report = validate_raw(raw)
    report.dataset_sha256 = dataset_sha256(path)
    print(render_text(report))
    try:
        cleaned = clean_telco(raw, report)
    except DataContractError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    out_dir = Path(args.out_dir)
    target = write_csv(cleaned, out_dir / config.CLEANED_CSV_NAME)
    print(f"Cleaned table written: {target} ({len(cleaned)} rows)")
    features = model_ready_frame(derive_features(cleaned), DEFAULT_SPEC)
    target = write_csv(features, out_dir / config.FEATURES_CSV_NAME)
    print(f"Model-ready table written: {target} ({features.shape[1]} columns, feature spec {DEFAULT_SPEC.fingerprint()})")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    path = Path(args.input)
    raw = load_raw(path)
    report = validate_raw(raw)
    report.dataset_sha256 = dataset_sha256(path)
    if not report.passed:
        print(render_text(report), file=sys.stderr)
        return 1
    cleaned = clean_telco(raw, report)
    run = run_baseline(
        cleaned,
        dataset_sha256=report.dataset_sha256,
        seed=args.seed,
        n_repeats=args.n_repeats,
    )
    bands = risk_band_report(run.oof, run.holdout_pred)
    metadata = build_metadata(
        run, dataset_sha256=report.dataset_sha256, seed=args.seed, extra={"risk_bands": bands}
    )
    explainer = LinearExplainer.from_artifacts(run.final_model, metadata)
    importance = explainer.global_importance(run.frame[list(run.spec.all)])
    global_explanation = {
        "method": (
            "Exact logistic-regression decomposition: contribution of an original feature = sum of "
            "coefficient x (encoded value - training mean) over its encoded columns. Evidence type: MODEL_SIGNAL."
        ),
        "reference": "average training customer profile (final model, in-sample, descriptive only)",
        "reference_log_odds": round(explainer.reference_logit, 6),
        "causal_explanation_available": False,
        "features": [
            {
                "feature": name,
                "mean_abs_contribution_log_odds": round(float(row.mean_abs_contribution), 6),
                "mean_signed_contribution_log_odds": round(float(row.mean_signed_contribution), 6),
            }
            for name, row in importance.iterrows()
        ],
    }
    fairness = run_fairness_audit(
        run.oof,
        run.holdout_pred,
        cleaned,
        run.spec,
        {**run.operating_thresholds, "high_band": config.RISK_BAND_CUTOFFS["HIGH"]},
        n_boot=args.bootstrap,
        seed=args.seed,
    )
    fairness["model_version"] = metadata["model_version"]
    write_json(fairness, args.fairness_report)
    run.report["model_version"] = metadata["model_version"]
    run.report["risk_bands"] = bands
    run.report["global_explanation"] = global_explanation
    write_json(run.report, args.report)
    save_artifacts(run.final_model, metadata, args.model_dir)
    print_training_summary(run, metadata)
    print(f"Report written: {args.report}")
    print(f"Fairness audit written: {args.fairness_report} (descriptive; not a fairness verdict)")
    print(f"Model written:  {args.model_dir}")
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    path = Path(args.input)
    raw = load_raw(path)
    has_target = "Churn" in raw.columns
    report = validate_raw(raw, require_target=has_target)
    if not report.passed:
        print(render_text(report), file=sys.stderr)
        return 1
    cleaned = clean_telco(raw, report, require_target=has_target)
    model, metadata = load_artifacts(args.model_dir)
    in_sample = dataset_sha256(path) == metadata["dataset"]["sha256_lf_normalised"]
    explainer = None if args.no_reasons else LinearExplainer.from_artifacts(model, metadata)
    records = score_cleaned(
        cleaned, model, metadata, in_sample=in_sample, explainer=explainer, top_k=args.top_k
    )

    out = Path(args.output)
    if out.suffix.lower() == ".json":
        write_json([r.model_dump(mode="json") for r in records], out)
    else:
        write_csv(records_to_frame(records), out)
    counts = {band: sum(r.risk_band == band for r in records) for band in ("LOW", "MEDIUM", "HIGH")}
    print(f"Scored {len(records)} customers with {metadata['model_version']}: {counts}")
    if in_sample:
        print(
            "WARNING: this file is the model's own training data, so these scores are in-sample (optimistic). "
            "Use out-of-sample customers, or the cross-validated metrics in reports/, to judge quality."
        )
    print(f"Scores written: {out}")
    return 0


def _cmd_value(args: argparse.Namespace) -> int:
    path = Path(args.input)
    raw = load_raw(path)
    has_target = "Churn" in raw.columns
    report = validate_raw(raw, require_target=has_target)
    if not report.passed:
        print(render_text(report), file=sys.stderr)
        return 1
    cleaned = clean_telco(raw, report, require_target=has_target)
    model, metadata = load_artifacts(args.model_dir)
    in_sample = dataset_sha256(path) == metadata["dataset"]["sha256_lf_normalised"]
    records = score_cleaned(cleaned, model, metadata, in_sample=in_sample)
    table = value_table(cleaned, records, assumed_months=args.assumed_months)
    summary = band_summary(table, cleaned)
    summary.update(
        {
            "model_version": metadata["model_version"],
            "assumed_months": args.assumed_months,
            "scoring_context": "in_sample" if in_sample else "out_of_sample",
            "provenance_label": metadata["dataset"]["provenance_label"],
        }
    )

    write_csv(table, args.output)
    if args.summary:
        write_json(summary, args.summary)
    print(f"Revenue exposure for {len(table)} customers (model {metadata['model_version']}, assumed months={args.assumed_months})")
    print(f"  total monthly revenue (observed):                 {summary['total_monthly_revenue']:>12,.2f}")
    print(f"  risk-weighted monthly revenue at risk (derived):  {summary['total_revenue_at_risk_monthly']:>12,.2f}")
    if "observed_monthly_revenue_of_churned_customers" in summary:
        print(f"  observed monthly revenue of churned (label):      {summary['observed_monthly_revenue_of_churned_customers']:>12,.2f}")
    for row in summary["bands"]:
        print(f"  {row['band']:<7} customers={row['customers']:<5} revenue_at_risk={row['revenue_at_risk_monthly']:>11,.2f} share={row['share_of_revenue_at_risk']}")
    print("Top 5 by risk-weighted monthly revenue:")
    print(top_revenue_at_risk(table, 5)[["customer_id", "risk_score", "risk_band", "monthly_revenue", "revenue_at_risk_monthly"]].to_string(index=False))
    print("NOTE: proxies only. No margin, horizon or true CLV exists in this dataset; clv_proxy uses an ASSUMED number of months.")
    if in_sample:
        print("WARNING: scored on the model's own training data (in-sample).")
    print(f"Value table written: {args.output}")
    return 0


def print_training_summary(run, metadata: dict) -> None:
    print(f"Model version: {metadata['model_version']} ({metadata['model_type']})")
    print(f"{'candidate':<24}{'CV ROC-AUC':>12}{'CV PR-AUC':>12}{'holdout ROC':>13}{'holdout PR':>12}")
    for name, entry in run.report["candidates"].items():
        cv, ho = entry["cv"], entry["holdout"]["ranking"]
        print(f"{name:<24}{cv['roc_auc']['mean']:>12.4f}{cv['pr_auc']['mean']:>12.4f}{ho['roc_auc']:>13.4f}{ho['pr_auc']:>12.4f}")
    sel = run.report["model_selection"]
    print(
        f"Shipped: {sel['shipped_model']}; best benchmark by CV PR-AUC: {sel['best_by_cv_pr_auc']} "
        f"(gap {sel['pr_auc_gap_best_minus_shipped']}); exceeds by >1 SD: {sel['benchmark_exceeds_shipped_by_more_than_1sd']}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retainai", description="RetainAI Phase 1 foundation pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_val = sub.add_parser("validate", help="Validate the raw CSV against the dataset contract")
    p_val.add_argument("--input", default=str(config.RAW_CSV))
    p_val.add_argument("--report", default=str(config.REPORTS_DIR / config.VALIDATION_REPORT_NAME))
    p_val.set_defaults(func=_cmd_validate)

    p_prep = sub.add_parser("prepare", help="Validate, then write the cleaned and model-ready tables")
    p_prep.add_argument("--input", default=str(config.RAW_CSV))
    p_prep.add_argument("--out-dir", default=str(config.PROCESSED_DIR))
    p_prep.set_defaults(func=_cmd_prepare)

    p_train = sub.add_parser("train", help="Train and evaluate baseline models; save the shipped model")
    p_train.add_argument("--input", default=str(config.RAW_CSV))
    p_train.add_argument("--model-dir", default=str(config.MODEL_DIR))
    p_train.add_argument("--report", default=str(config.REPORTS_DIR / config.BASELINE_REPORT_NAME))
    p_train.add_argument("--seed", type=int, default=config.SEED)
    p_train.add_argument("--n-repeats", type=int, default=3)
    p_train.add_argument("--fairness-report", default=str(config.REPORTS_DIR / "fairness_audit.json"))
    p_train.add_argument("--bootstrap", type=int, default=500, help="Bootstrap resamples for the subgroup audit")
    p_train.set_defaults(func=_cmd_train)

    p_score = sub.add_parser("score", help="Generate risk scores for a raw-format customer CSV")
    p_score.add_argument("--input", default=str(config.RAW_CSV))
    p_score.add_argument("--model-dir", default=str(config.MODEL_DIR))
    p_score.add_argument("--output", default=str(config.PROCESSED_DIR / "risk_scores.csv"))
    p_score.add_argument("--top-k", type=int, default=3, help="Max risk-increasing reasons per customer")
    p_score.add_argument("--no-reasons", action="store_true", help="Skip model-grounded reasons")
    p_score.set_defaults(func=_cmd_score)

    p_value = sub.add_parser("value", help="Compute revenue-at-risk proxies from risk scores")
    p_value.add_argument("--input", default=str(config.RAW_CSV))
    p_value.add_argument("--model-dir", default=str(config.MODEL_DIR))
    p_value.add_argument("--output", default=str(config.PROCESSED_DIR / "value_at_risk.csv"))
    p_value.add_argument("--summary", default=str(config.REPORTS_DIR / "value_summary.json"))
    p_value.add_argument(
        "--assumed-months",
        type=float,
        default=config.ASSUMED_VALUE_MONTHS,
        help="ASSUMED months of revenue treated as at stake in the CLV proxy",
    )
    p_value.set_defaults(func=_cmd_value)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
