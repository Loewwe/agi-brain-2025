#!/usr/bin/env python3
"""
Alpha Scan Results Aggregator (Stage 9 Phase 2)

Aggregates all experiment results, classifies them, and generates summary report.

Usage:
    python scripts/summarize_alpha_scan.py --results-dir results/alpha_scan
    
    # Generate markdown report
    python scripts/summarize_alpha_scan.py --results-dir results/alpha_scan --output alpha_scan_summary.md
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


# Classification thresholds from ALPHA_SCAN_SPEC.md
CANDIDATE_CRITERIA = {
    'pf_post_cost': 1.15,
    'sharpe_post_cost': 1.5,
    'trades_per_month': 1000,
    'max_drawdown_post_cost': -0.25,
}

BORDERLINE_CRITERIA = {
    'pf_post_cost': 1.05,
    'sharpe_post_cost': 1.0,
    'trades_per_month': 2000,
}


def load_results(results_dir: str) -> list[dict]:
    """Load all result JSON files from directory."""
    results = []
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return []
    
    for json_file in sorted(results_path.glob("*.json")):
        try:
            with open(json_file, "r") as f:
                result = json.load(f)
                results.append(result)
        except Exception as e:
            print(f"⚠️  Error loading {json_file}: {e}")
    
    return results


def classify_result(result: dict) -> str:
    """Classify result as 'candidate', 'borderline', or 'rejected'."""
    
    # Check for errors
    if "_meta" in result and "error" in result.get("_meta", {}):
        return "error"
    
    pf = result.get("profit_factor_post_cost")
    sharpe = result.get("sharpe_post_cost")
    trades = result.get("trades_per_month", result.get("n_trades", 0))
    max_dd = result.get("max_drawdown_post_cost")
    
    # Handle None values
    if pf is None or sharpe is None:
        return "rejected"
    
    # Check candidate criteria (must pass ALL)
    is_candidate = (
        pf >= CANDIDATE_CRITERIA['pf_post_cost'] and
        sharpe >= CANDIDATE_CRITERIA['sharpe_post_cost'] and
        trades <= CANDIDATE_CRITERIA['trades_per_month'] and
        (max_dd is None or max_dd >= CANDIDATE_CRITERIA['max_drawdown_post_cost'])
    )
    
    if is_candidate:
        return "candidate"
    
    # Check borderline criteria
    is_borderline = (
        pf >= BORDERLINE_CRITERIA['pf_post_cost'] and
        sharpe >= BORDERLINE_CRITERIA['sharpe_post_cost'] and
        trades <= BORDERLINE_CRITERIA['trades_per_month']
    )
    
    if is_borderline:
        return "borderline"
    
    return "rejected"


def compute_score(result: dict) -> float:
    """Compute composite score for ranking."""
    pf = result.get("profit_factor_post_cost", 1.0)
    sharpe = result.get("sharpe_post_cost", 0.0)
    n_trades = result.get("n_trades", 1)
    
    if pf is None or sharpe is None:
        return -999.0
    
    # Avoid division by zero
    trades_per_day = max(n_trades / 30, 0.1)
    frequency_penalty = np.log(1 + trades_per_day)
    
    # Score: higher Sharpe and PF, lower frequency
    score = sharpe * max(pf - 1.0, 0) / max(frequency_penalty, 0.1)
    return score


def aggregate_results(results: list[dict]) -> dict:
    """Aggregate and classify all results."""
    
    classified = {
        "candidates": [],
        "borderline": [],
        "rejected": [],
        "errors": [],
    }
    
    for result in results:
        classification = classify_result(result)
        result["_classification"] = classification
        result["_score"] = compute_score(result)
        
        if classification == "candidate":
            classified["candidates"].append(result)
        elif classification == "borderline":
            classified["borderline"].append(result)
        elif classification == "error":
            classified["errors"].append(result)
        else:
            classified["rejected"].append(result)
    
    # Sort by score (descending)
    for category in ["candidates", "borderline", "rejected"]:
        classified[category].sort(key=lambda r: r.get("_score", 0), reverse=True)
    
    return classified


def format_table_row(result: dict) -> str:
    """Format result as markdown table row."""
    meta = result.get("_meta", {})
    
    exp_id = meta.get("id", "unknown")
    symbol = meta.get("symbol", "?")
    tf = meta.get("timeframe", "?")
    target = meta.get("target_type", "?")
    features = meta.get("feature_set", "?")[:3]  # Abbreviate
    
    pf_pre = result.get("profit_factor", 0)
    pf_post = result.get("profit_factor_post_cost", 0)
    sharpe_pre = result.get("sharpe", 0)
    sharpe_post = result.get("sharpe_post_cost", 0)
    n_trades = result.get("n_trades", 0)
    max_dd = result.get("max_drawdown_post_cost", 0)
    auc = result.get("auc", 0)
    score = result.get("_score", 0)
    
    return f"| {exp_id} | {symbol} | {tf} | {target} | {features} | {pf_pre:.3f} | {pf_post:.3f} | {sharpe_pre:.2f} | {sharpe_post:.2f} | {n_trades} | {max_dd:.1%} | {auc:.3f} | {score:.3f} |"


def generate_summary(classified: dict, output_path: str | None = None) -> str:
    """Generate markdown summary report."""
    
    total = sum(len(v) for v in classified.values())
    n_candidates = len(classified["candidates"])
    n_borderline = len(classified["borderline"])
    n_rejected = len(classified["rejected"])
    n_errors = len(classified["errors"])
    
    lines = [
        "# Alpha Scan v1 — Summary Report",
        "",
        f"**Generated:** {__import__('datetime').datetime.now().isoformat()[:19]}",
        "",
        "---",
        "",
        "## Overview",
        "",
        f"**Total Experiments:** {total}",
        "",
        f"| Category | Count | % |",
        f"|----------|-------|---|",
        f"| 🟢 Candidates | {n_candidates} | {n_candidates/total*100:.1f}% |",
        f"| 🟡 Borderline | {n_borderline} | {n_borderline/total*100:.1f}% |",
        f"| ⚪ Rejected | {n_rejected} | {n_rejected/total*100:.1f}% |",
        f"| ❌ Errors | {n_errors} | {n_errors/total*100:.1f}% |",
        "",
        "---",
        "",
    ]
    
    # Candidate criteria reminder
    lines.extend([
        "## Thresholds",
        "",
        "**Candidate (must meet ALL):**",
        f"- PF_post_cost ≥ {CANDIDATE_CRITERIA['pf_post_cost']}",
        f"- Sharpe_post_cost ≥ {CANDIDATE_CRITERIA['sharpe_post_cost']}",
        f"- trades/month ≤ {CANDIDATE_CRITERIA['trades_per_month']}",
        f"- MaxDD ≥ {CANDIDATE_CRITERIA['max_drawdown_post_cost']:.0%}",
        "",
        "**Borderline (some promise):**",
        f"- PF_post_cost ≥ {BORDERLINE_CRITERIA['pf_post_cost']}",
        f"- Sharpe_post_cost ≥ {BORDERLINE_CRITERIA['sharpe_post_cost']}",
        f"- trades/month ≤ {BORDERLINE_CRITERIA['trades_per_month']}",
        "",
        "---",
        "",
    ])
    
    # Results tables
    table_header = "| ID | Symbol | TF | Target | Feat | PF_pre | PF_post | Sharpe_pre | Sharpe_post | Trades | MaxDD | AUC | Score |"
    table_separator = "|-----|--------|-----|--------|------|--------|---------|------------|-------------|--------|-------|------|-------|"
    
    # Candidates
    lines.extend([
        "## 🟢 Candidates",
        "",
    ])
    
    if classified["candidates"]:
        lines.append(table_header)
        lines.append(table_separator)
        for result in classified["candidates"]:
            lines.append(format_table_row(result))
        lines.append("")
    else:
        lines.append("**No candidates found.**")
        lines.append("")
    
    # Borderline
    lines.extend([
        "---",
        "",
        "## 🟡 Borderline",
        "",
    ])
    
    if classified["borderline"]:
        lines.append(table_header)
        lines.append(table_separator)
        for result in classified["borderline"][:10]:  # Top 10 only
            lines.append(format_table_row(result))
        if len(classified["borderline"]) > 10:
            lines.append(f"... and {len(classified['borderline']) - 10} more")
        lines.append("")
    else:
        lines.append("**No borderline experiments.**")
        lines.append("")
    
    # Rejected summary (don't list all)
    lines.extend([
        "---",
        "",
        "## ⚪ Rejected",
        "",
        f"**{n_rejected} experiments rejected** (not meeting borderline criteria)",
        "",
        "Common rejection reasons:",
        "- PF_post_cost ≤ 1.02 (thin or no edge)",
        "- Sharpe_post_cost ≤ 1.0 (poor risk-adjusted returns)",
        "- Too many trades (costs dominate)",
        "",
    ])
    
    # Top rejected (for analysis)
    if classified["rejected"]:
        lines.extend([
            "### Top 5 Rejected (best of the rejected):",
            "",
            table_header,
            table_separator,
        ])
        for result in classified["rejected"][:5]:
            lines.append(format_table_row(result))
        lines.append("")
    
    # Verdict
    lines.extend([
        "---",
        "",
        "## Verdict",
        "",
    ])
    
    if n_candidates >= 2:
        lines.extend([
            f"✅ **SUCCESS:** Found {n_candidates} candidate(s) meeting all criteria.",
            "",
            "**Next Steps:**",
            "1. Proceed to Phase 3 (Multi-Period Robustness Check)",
            "2. Test candidates on 2 additional periods",
            "3. If robust → Stage 10 (Paper Trading)",
        ])
    elif n_candidates == 1:
        lines.extend([
            f"⚠️ **MARGINAL:** Found only 1 candidate.",
            "",
            "Consider:",
            "- Still run robustness check",
            "- Review borderline experiments for near-misses",
        ])
    elif n_borderline >= 3:
        lines.extend([
            f"🟡 **BORDERLINE:** No candidates, but {n_borderline} borderline experiments.",
            "",
            "Options:",
            "- Relax thresholds slightly (PF ≥ 1.10?)",
            "- Test borderline on different periods",
            "- Consider this a weak signal, not definitive null",
        ])
    else:
        lines.extend([
            "🛑 **NULL RESULT:** No candidates or meaningful borderline experiments.",
            "",
            "**Conclusion:**",
            "> No sustainable edge found in current search space:",
            "> - Symbols: BTC, ETH, SOL",
            "> - Timeframes: 15m, 1h",
            "> - Targets: Momentum, Reversal, Vol Expansion",
            "> - Realistic costs: 0.3% per round-trip",
            "",
            "**Decision:** Archive alpha research, **PIVOT TO AGI-BRAIN**",
        ])
    
    lines.append("")
    
    report = "\n".join(lines)
    
    # Save if output path provided
    if output_path:
        with open(output_path, "w") as f:
            f.write(report)
        print(f"📝 Report saved to: {output_path}")
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Aggregate and summarize alpha scan results")
    parser.add_argument("--results-dir", default="results/alpha_scan", help="Results directory")
    parser.add_argument("--output", help="Output markdown file path")
    parser.add_argument("--json", help="Output JSON file path for raw aggregated data")
    
    args = parser.parse_args()
    
    print(f"Loading results from: {args.results_dir}")
    results = load_results(args.results_dir)
    
    if not results:
        print("No results found.")
        return
    
    print(f"Loaded {len(results)} results")
    
    classified = aggregate_results(results)
    
    # Print quick summary to console
    print(f"\n{'='*60}")
    print(f"ALPHA SCAN v1 — SUMMARY")
    print(f"{'='*60}")
    print(f"🟢 Candidates:  {len(classified['candidates'])}")
    print(f"🟡 Borderline:  {len(classified['borderline'])}")
    print(f"⚪ Rejected:    {len(classified['rejected'])}")
    print(f"❌ Errors:      {len(classified['errors'])}")
    print(f"{'='*60}\n")
    
    # Generate report
    report = generate_summary(classified, args.output)
    
    if not args.output:
        print(report)
    
    # Save JSON if requested
    if args.json:
        with open(args.json, "w") as f:
            json.dump(classified, f, indent=2, default=str)
        print(f"📊 JSON data saved to: {args.json}")


if __name__ == "__main__":
    main()
