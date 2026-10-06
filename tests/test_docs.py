"""Documentation must match the implementation: commands exist, links resolve, numbers match the reports."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from retainai.cli import build_parser

ROOT = Path(__file__).resolve().parents[1]
DOCS = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]


def _subcommands() -> set[str]:
    parser = build_parser()
    return set(next(a for a in parser._actions if a.dest == "command").choices)


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_every_documented_command_exists():
    commands = _subcommands()
    assert commands == {"validate", "prepare", "train", "score", "value"}
    pattern = re.compile(r"python -m retainai (\w+)")
    for path in (ROOT / "README.md", ROOT / "docs" / "PHASE_1_PIPELINE.md"):
        used = set(pattern.findall(_text(path)))
        assert used and used <= commands, (path.name, used - commands)
    readme = _text(ROOT / "README.md")
    assert commands <= set(pattern.findall(readme))  # README documents every command


def test_documented_cli_options_exist():
    parser = build_parser()
    sub = next(a for a in parser._actions if a.dest == "command").choices
    for flag, command in [("--input", "score"), ("--output", "score"), ("--assumed-months", "value"), ("--no-reasons", "score"), ("--top-k", "score")]:
        assert any(flag in action.option_strings for action in sub[command]._actions), (command, flag)
    assert "--runslow" in (ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")


@pytest.mark.parametrize("path", DOCS, ids=lambda p: p.name)
def test_relative_links_and_referenced_paths_resolve(path):
    text = _text(path)
    for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", text):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        assert (path.parent / target).resolve().exists(), (path.name, target)
    for ref in re.findall(r"`((?:reports|docs|src/retainai|tests|data/raw|notebooks)/[\w./-]+\.\w+)`", text):
        assert (ROOT / ref).exists(), (path.name, ref)


def test_pipeline_doc_numbers_match_the_committed_reports():
    doc = _text(ROOT / "docs" / "PHASE_1_PIPELINE.md")
    report = json.loads((ROOT / "reports" / "baseline_metrics.json").read_text(encoding="utf-8"))
    value = json.loads((ROOT / "reports" / "value_summary.json").read_text(encoding="utf-8"))

    assert report["model_version"] in doc
    assert report["dataset"]["sha256_lf_normalised"][:16] in doc
    lr = report["candidates"]["logistic_regression"]
    for metric in ("roc_auc", "pr_auc"):
        assert f"{lr['cv'][metric]['mean']:.3f}" in doc
        assert f"{lr['holdout']['ranking'][metric]:.3f}" in doc
    for name in ("random_forest", "hist_gradient_boosting"):
        for metric in ("roc_auc", "pr_auc"):
            assert f"{report['candidates'][name]['cv'][metric]['mean']:.3f}" in doc
    holdout = report["shipped_model_analysis"]["holdout"]
    cm = holdout["at_dev_best_f1_threshold"]["confusion_matrix"]
    assert f"tn {cm['tn']}, fp {cm['fp']}, fn {cm['fn']}, tp {cm['tp']}" in doc
    assert f"{report['shipped_model_analysis']['best_f1_threshold_dev_oof']:.2f}" in doc
    for band in report["risk_bands"]["holdout"]:
        assert f"{band['observed_churn_rate']:.3f}" in doc
    for feature in report["global_explanation"]["features"][:5]:
        assert f"`{feature['feature']}` {feature['mean_abs_contribution_log_odds']:.2f}" in doc
    assert f"{value['total_revenue_at_risk_monthly']:,.0f}" in doc
    assert f"{value['total_monthly_revenue']:,.0f}" in doc
    assert f"{value['observed_monthly_revenue_of_churned_customers']:,.0f}" in doc


def test_pipeline_doc_fairness_figures_match_the_audit():
    doc = _text(ROOT / "docs" / "PHASE_1_PIPELINE.md")
    audit = json.loads((ROOT / "reports" / "fairness_audit.json").read_text(encoding="utf-8"))
    senior = audit["evaluation_sets"]["dev_oof"]["attributes"]["senior_citizen"]["groups"]
    for key in ("observed_churn_rate", "selection_rate@best_f1", "tpr@best_f1", "fpr@best_f1", "roc_auc"):
        for group in ("0", "1"):
            assert f"{senior[group]['metrics'][key]['estimate']:.3f}" in doc, (key, group)
    gender = audit["evaluation_sets"]["dev_oof"]["attributes"]["gender"]["differences"]["Male_minus_Female"]
    assert not [k for k, v in gender.items() if v["ci_excludes_zero"]]  # the doc says no detectable gender gaps


def test_readme_states_the_evidence_boundaries():
    readme = _text(ROOT / "README.md")
    for phrase in ("fictional", "no causal claims", "undefined churn horizon", "proxies", "in_sample"):
        assert phrase.lower() in readme.lower(), phrase
    assert "web-based interface" not in readme  # not implemented; must not be promised as current
