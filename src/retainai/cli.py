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
from retainai.io import write_csv, write_json
from retainai.models.baseline import run_baseline
from retainai.models.registry import build_metadata, save_artifacts


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
    metadata = build_metadata(run, dataset_sha256=report.dataset_sha256, seed=args.seed)
    run.report["model_version"] = metadata["model_version"]
    write_json(run.report, args.report)
    save_artifacts(run.final_model, metadata, args.model_dir)
    print_training_summary(run, metadata)
    print(f"Report written: {args.report}")
    print(f"Model written:  {args.model_dir}")
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
    p_train.set_defaults(func=_cmd_train)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
