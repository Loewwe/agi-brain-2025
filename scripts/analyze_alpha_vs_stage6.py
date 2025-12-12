#!/usr/bin/env python3
"""
Analyze Alpha vs Stage6 Attribution

Compares ideal alpha signals with actual Stage6 trades to find
where winrate drops from ~70% to ~20%.

Categories:
  - GOOD_AND_GOOD: Alpha right + Stage6 won
  - GOOD_BUT_NO_TRADE: Alpha right but no trade
  - GOOD_BUT_LOSS_TRADE: Alpha right but Stage6 lost
  - BAD_BUT_TRADE: Alpha wrong but Stage6 traded
  - NO_SIGNAL: No alpha signal that day

Inputs:
  - reports/alpha_signals_daily.csv
  - reports/stage6_trades.csv (from backtest)

Outputs:
  - reports/alpha_vs_stage6_attribution.json
  - reports/alpha_vs_stage6_attribution.md

Usage:
  python scripts/analyze_alpha_vs_stage6.py
"""

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# ATTRIBUTION CATEGORIES
# =============================================================================

class AttributionCategory:
    GOOD_AND_GOOD = "GOOD_AND_GOOD"
    GOOD_BUT_NO_TRADE = "GOOD_BUT_NO_TRADE"
    GOOD_BUT_LOSS_TRADE = "GOOD_BUT_LOSS_TRADE"
    BAD_BUT_TRADE = "BAD_BUT_TRADE"
    NO_SIGNAL = "NO_SIGNAL"


# =============================================================================
# ANALYZER
# =============================================================================

class AlphaStage6Analyzer:
    """Analyzes attribution between alpha signals and Stage6 trades."""
    
    def __init__(self, alpha_path: Path, trades_path: Path):
        self.alpha_path = alpha_path
        self.trades_path = trades_path
        self.alpha_df = None
        self.trades_df = None
    
    def load_data(self):
        """Load alpha signals and trades data."""
        if not self.alpha_path.exists():
            raise FileNotFoundError(f"Alpha signals not found: {self.alpha_path}")
        
        self.alpha_df = pd.read_csv(self.alpha_path)
        logger.info(f"Loaded {len(self.alpha_df)} alpha signals")
        
        if self.trades_path.exists():
            self.trades_df = pd.read_csv(self.trades_path)
            logger.info(f"Loaded {len(self.trades_df)} Stage6 trades")
        else:
            logger.warning(f"Trades file not found: {self.trades_path}")
            logger.info("Creating empty trades DataFrame for analysis")
            self.trades_df = pd.DataFrame(columns=[
                "trade_id", "date", "symbol", "direction", "entry_ts", "exit_ts",
                "entry_price", "exit_price", "pnl_abs", "pnl_pct", "win_stage6", "exit_reason"
            ])
    
    def analyze(self) -> dict:
        """Run full attribution analysis."""
        self.load_data()
        
        # Build index of trades by (date, symbol)
        trades_index = defaultdict(list)
        for _, row in self.trades_df.iterrows():
            key = (str(row.get("date", "")), str(row.get("symbol", "")))
            trades_index[key].append(row)
        
        # Attribution counters
        attribution = {
            AttributionCategory.GOOD_AND_GOOD: {"count": 0, "examples": []},
            AttributionCategory.GOOD_BUT_NO_TRADE: {"count": 0, "examples": []},
            AttributionCategory.GOOD_BUT_LOSS_TRADE: {"count": 0, "examples": [], "exit_reasons": defaultdict(int)},
            AttributionCategory.BAD_BUT_TRADE: {"count": 0, "examples": [], "exit_reasons": defaultdict(int)},
            AttributionCategory.NO_SIGNAL: {"count": 0, "examples": []},
        }
        
        total_signals = 0
        directional_signals = 0
        ideal_wins = 0
        stage6_wins = 0
        stage6_trades = 0
        
        # Process each alpha signal
        for _, alpha_row in self.alpha_df.iterrows():
            date = alpha_row["date"]
            symbol = alpha_row["symbol"]
            direction = alpha_row["alpha_direction"]
            win_ideal = alpha_row["win_ideal"]
            
            total_signals += 1
            
            # Get trades for this (date, symbol)
            key = (date, symbol)
            trades = trades_index.get(key, [])
            
            had_signal = direction != "FLAT"
            had_ideal_win = win_ideal == 1
            had_trade = len(trades) > 0
            any_win_trade = any(t.get("win_stage6", t.get("pnl_pct", 0) > 0) == 1 for t in trades)
            
            if had_signal:
                directional_signals += 1
            if had_ideal_win:
                ideal_wins += 1
            if had_trade:
                stage6_trades += len(trades)
                for t in trades:
                    if t.get("win_stage6", t.get("pnl_pct", 0) > 0) == 1:
                        stage6_wins += 1
            
            # Classify
            if not had_signal:
                cat = AttributionCategory.NO_SIGNAL
            elif had_ideal_win and had_trade and any_win_trade:
                cat = AttributionCategory.GOOD_AND_GOOD
            elif had_ideal_win and not had_trade:
                cat = AttributionCategory.GOOD_BUT_NO_TRADE
            elif had_ideal_win and had_trade and not any_win_trade:
                cat = AttributionCategory.GOOD_BUT_LOSS_TRADE
                for t in trades:
                    reason = t.get("exit_reason", "UNKNOWN")
                    attribution[cat]["exit_reasons"][reason] += 1
            elif not had_ideal_win and had_trade:
                cat = AttributionCategory.BAD_BUT_TRADE
                for t in trades:
                    reason = t.get("exit_reason", "UNKNOWN")
                    attribution[cat]["exit_reasons"][reason] += 1
            else:
                # Alpha wrong, no trade (expected, skip)
                continue
            
            attribution[cat]["count"] += 1
            if len(attribution[cat]["examples"]) < 3:
                attribution[cat]["examples"].append({
                    "date": date,
                    "symbol": symbol,
                    "alpha_direction": direction,
                    "win_ideal": win_ideal,
                    "trades": len(trades),
                })
        
        # Calculate winrates
        ideal_winrate = ideal_wins / directional_signals if directional_signals > 0 else 0
        stage6_winrate = stage6_wins / stage6_trades if stage6_trades > 0 else 0
        
        # Build result
        result = {
            "meta": {
                "analyzed_at": datetime.now().isoformat(),
                "alpha_file": str(self.alpha_path),
                "trades_file": str(self.trades_path),
            },
            "summary": {
                "total_signals": total_signals,
                "directional_signals": directional_signals,
                "ideal_wins": ideal_wins,
                "ideal_winrate": round(ideal_winrate * 100, 1),
                "stage6_trades": stage6_trades,
                "stage6_wins": stage6_wins,
                "stage6_winrate": round(stage6_winrate * 100, 1),
                "winrate_gap": round((ideal_winrate - stage6_winrate) * 100, 1),
            },
            "attribution": {}
        }
        
        # Convert attribution to serializable format
        for cat, data in attribution.items():
            result["attribution"][cat] = {
                "count": data["count"],
                "share_of_signals": round(data["count"] / total_signals * 100, 1) if total_signals > 0 else 0,
                "examples": data["examples"][:3],
            }
            if "exit_reasons" in data and data["exit_reasons"]:
                result["attribution"][cat]["exit_reasons"] = dict(data["exit_reasons"])
        
        return result
    
    def generate_markdown(self, result: dict) -> str:
        """Generate markdown report from analysis result."""
        md = []
        md.append("# Alpha vs Stage6 Attribution Report\n")
        md.append(f"*Generated: {result['meta']['analyzed_at']}*\n")
        
        # Summary
        s = result["summary"]
        md.append("## Summary\n")
        md.append("| Metric | Value |")
        md.append("|--------|-------|")
        md.append(f"| Total Signals | {s['total_signals']} |")
        md.append(f"| Directional Signals | {s['directional_signals']} |")
        md.append(f"| **Ideal Winrate** | **{s['ideal_winrate']}%** |")
        md.append(f"| Stage6 Trades | {s['stage6_trades']} |")
        md.append(f"| **Stage6 Winrate** | **{s['stage6_winrate']}%** |")
        md.append(f"| **Winrate Gap** | **{s['winrate_gap']}%** ⚠️ |")
        md.append("")
        
        # Attribution breakdown
        md.append("## Attribution Breakdown\n")
        md.append("| Category | Count | Share |")
        md.append("|----------|-------|-------|")
        for cat, data in result["attribution"].items():
            emoji = "✅" if cat == AttributionCategory.GOOD_AND_GOOD else "❌" if "LOSS" in cat else "⚠️"
            md.append(f"| {emoji} {cat} | {data['count']} | {data['share_of_signals']}% |")
        md.append("")
        
        # Exit reasons for loss categories
        for cat in [AttributionCategory.GOOD_BUT_LOSS_TRADE, AttributionCategory.BAD_BUT_TRADE]:
            data = result["attribution"].get(cat, {})
            if data.get("exit_reasons"):
                md.append(f"### {cat} - Exit Reasons\n")
                md.append("| Reason | Count |")
                md.append("|--------|-------|")
                for reason, count in sorted(data["exit_reasons"].items(), key=lambda x: -x[1]):
                    md.append(f"| {reason} | {count} |")
                md.append("")
        
        # Key findings
        md.append("## Key Findings\n")
        
        good_no_trade = result["attribution"].get(AttributionCategory.GOOD_BUT_NO_TRADE, {}).get("count", 0)
        good_loss = result["attribution"].get(AttributionCategory.GOOD_BUT_LOSS_TRADE, {}).get("count", 0)
        
        if good_no_trade > 0:
            md.append(f"1. **{good_no_trade}** times alpha was RIGHT but we didn't trade (filters/limits)")
        if good_loss > 0:
            md.append(f"2. **{good_loss}** times alpha was RIGHT but Stage6 LOST (SL/execution)")
        
        md.append("")
        return "\n".join(md)


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze Alpha vs Stage6 Attribution")
    parser.add_argument("--alpha-path", type=str, default="reports/alpha_signals_daily.csv")
    parser.add_argument("--trades-path", type=str, default="reports/stage6_trades.csv")
    args = parser.parse_args()
    
    root = Path(__file__).parent.parent
    alpha_path = root / args.alpha_path
    trades_path = root / args.trades_path
    
    # Run analysis
    analyzer = AlphaStage6Analyzer(alpha_path, trades_path)
    result = analyzer.analyze()
    
    # Save JSON
    json_path = root / "reports" / "alpha_vs_stage6_attribution.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved: {json_path}")
    
    # Save Markdown
    md_content = analyzer.generate_markdown(result)
    md_path = root / "reports" / "alpha_vs_stage6_attribution.md"
    with open(md_path, "w") as f:
        f.write(md_content)
    logger.info(f"Saved: {md_path}")
    
    # Print summary
    s = result["summary"]
    print("\n" + "="*60)
    print("ALPHA VS STAGE6 ATTRIBUTION")
    print("="*60)
    print(f"Ideal Winrate:   {s['ideal_winrate']}%")
    print(f"Stage6 Winrate:  {s['stage6_winrate']}%")
    print(f"Winrate Gap:     {s['winrate_gap']}% ⚠️")
    print("="*60)
    
    for cat, data in result["attribution"].items():
        print(f"  {cat}: {data['count']} ({data['share_of_signals']}%)")
    print("="*60)


if __name__ == "__main__":
    main()
