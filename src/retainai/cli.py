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

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
