#!/usr/bin/env python3
"""
Alpha Search v2.0 - Night Pipeline: Stage 1 + Stage 2

Runs complete overnight research pipeline:
  Stage 0 configs → Stage 1 (60d ranking) → Stage 2 (3y realistic) → Final summary

Usage:
  python alpha_night_run_stage1_stage2.py [--top-n 100]
"""

import asyncio
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import argparse

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/tmp/alpha_night_run.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"
REPORTS_DIR = BASE_DIR / "reports" / "alpha_search_v2"

# Config
DEFAULT_TOP_N = 100
PYTHON_BIN = "/root/trade_stage6_venv/bin/python"


def log_marker(marker: str):
    """Log important markers."""
    logger.info("\n" + "="*70)
    logger.info(f"*** {marker} ***")
    logger.info("="*70 + "\n")


async def run_stage1() -> Path:
    """Run Stage 1: 60-day ranking backtest."""
    log_marker("STAGE1_START")
    
    input_file = REPORTS_DIR / "stage0_passed.csv"
    output_file = REPORTS_DIR / "stage1_results_ranked.csv"
    
    if not input_file.exists():
        raise FileNotFoundError(f"Stage 0 input not found: {input_file}")
    
    logger.info(f"Input: {input_file}")
    logger.info(f"Output: {output_file}")
    
    # Run Stage 1 script
    cmd = [
        PYTHON_BIN,
        str(SCRIPTS_DIR / "alpha_backtest_stage1_60d.py"),
        "--input", str(input_file),
        "--output", str(output_file)
    ]
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    if process.returncode != 0:
        logger.error(f"Stage 1 failed with code {process.returncode}")
        logger.error(f"STDERR: {process.stderr}")
        raise RuntimeError("Stage 1 failed")
    
    logger.info(f"STDOUT: {process.stdout}")
    
    if not output_file.exists():
        raise FileNotFoundError(f"Stage 1 output not created: {output_file}")
    
    log_marker("STAGE1_DONE")
    return output_file


def select_top_n_for_stage2(stage1_file: Path, top_n: int) -> Path:
    """Select top N candidates from Stage 1 for Stage 2."""
    logger.info(f"\nSelecting top {top_n} candidates for Stage 2...")
    
    df = pd.read_csv(stage1_file)
    logger.info(f"Loaded {len(df)} results from Stage 1")
    
    # Filter candidates only
    candidates = df[df['is_candidate'] == True].copy()
    logger.info(f"Candidates: {len(candidates)}")
    
    if len(candidates) == 0:
        raise ValueError("No candidates found in Stage 1!")
    
    # Sort by score and take top N
    candidates = candidates.sort_values('score_stage1', ascending=False)
    top_candidates = candidates.head(top_n)
    
    logger.info(f"Selected top {len(top_candidates)} for Stage 2")
    
    # Save selection
    output_file = REPORTS_DIR / f"stage1_top{top_n}_for_stage2.csv"
    top_candidates.to_csv(output_file, index=False)
    
    logger.info(f"Saved to: {output_file}")
    logger.info(f"\nTop 5 scores:")
    top5 = top_candidates.head(5)[['alpha_id', 'score_stage1', 'total_return_pct_60d', 'winrate_trades', 'profit_factor']]
    logger.info("\n" + top5.to_string(index=False))
    
    return output_file


async def run_stage2(stage2_input: Path) -> Path:
    """Run Stage 2: 3-year realistic backtest."""
    log_marker("STAGE2_START")
    
    output_dir = REPORTS_DIR / "stage2"
    
    logger.info(f"Input: {stage2_input}")
    logger.info(f"Output dir: {output_dir}")
    
    # Run Stage 2 script
    cmd = [
        PYTHON_BIN,
        str(SCRIPTS_DIR / "alpha_backtest_stage2_3y.py"),
        "--input", str(stage2_input),
        "--output-dir", str(output_dir)
    ]
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    if process.returncode != 0:
        logger.error(f"Stage 2 failed with code {process.returncode}")
        logger.error(f"STDERR: {process.stderr}")
        raise RuntimeError("Stage 2 failed")
    
    logger.info(f"STDOUT: {process.stdout}")
    
    output_file = output_dir / "stage2_summary.csv"
    if not output_file.exists():
        raise FileNotFoundError(f"Stage 2 output not created: {output_file}")
    
    log_marker("STAGE2_DONE")
    return output_file


def generate_final_summary(stage1_file: Path, stage2_file: Path):
    """Generate final summary combining Stage 1 and Stage 2 results."""
    log_marker("GENERATING_FINAL_SUMMARY")
    
    # Load both stages
    stage1_df = pd.read_csv(stage1_file)
    stage2_df = pd.read_csv(stage2_file)
    
    logger.info(f"Stage 1 results: {len(stage1_df)}")
    logger.info(f"Stage 2 tested: {len(stage2_df)}")
    
    # Merge
    merged = stage2_df.merge(
        stage1_df[['alpha_id', 'score_stage1', 'total_return_pct_60d']],
        on='alpha_id',
        how='left'
    )
    
    # Filter passed
    passed = merged[merged['pass_stage2'] == True]
    
    logger.info(f"Stage 2 passed: {len(passed)}")
    
    # Summary stats
    summary = {
        "generated_at": datetime.utcnow().isoformat(),
        "pipeline_version": "v2.0",
        "stats": {
            "total_stage0": len(stage1_df),
            "stage1_candidates": sum(stage1_df['is_candidate']),
            "stage2_tested": len(stage2_df),
            "stage2_passed": len(passed)
        },
        "top_by_cagr": [],
        "top_by_risk_adjusted": []
    }
    
    if len(passed) > 0:
        # Top 5 by CAGR
        top_cagr = passed.nlargest(5, 'cagr')
        summary["top_by_cagr"] = top_cagr[[
            'alpha_id', 'cagr', 'max_drawdown_pct', 'win_rate', 'profit_factor', 'num_trades'
        ]].to_dict('records')
        
        # Top 5 by risk-adjusted (CAGR / MaxDD)
        passed['risk_adjusted_score'] = passed['cagr'] / (passed['max_drawdown_pct'] + 1)
        top_risk_adj = passed.nlargest(5, 'risk_adjusted_score')
        summary["top_by_risk_adjusted"] = top_risk_adj[[
            'alpha_id', 'cagr', 'max_drawdown_pct', 'win_rate', 'profit_factor', 'risk_adjusted_score'
        ]].to_dict('records')
    
    # Save JSON
    json_path = REPORTS_DIR / "alpha_final_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved JSON: {json_path}")
    
    # Generate Markdown
    md_content = f"""# Alpha Search v2.0 - Final Summary

**Generated:** {summary['generated_at']}

---

## Pipeline Stats

- **Total Stage 0 configs:** {summary['stats']['total_stage0']:,}
- **Stage 1 candidates:** {summary['stats']['stage1_candidates']:,}
- **Stage 2 tested:** {summary['stats']['stage2_tested']}
- **Stage 2 passed:** {summary['stats']['stage2_passed']} ✅

---

## Results

"""
    
    if len(passed) > 0:
        md_content += f"### Top 5 by CAGR\n\n"
        for i, row in enumerate(summary['top_by_cagr'], 1):
            md_content += f"{i}. **Alpha {row['alpha_id']}**: CAGR {row['cagr']:.1f}%, MaxDD {row['max_drawdown_pct']:.1f}%, WR {row['win_rate']:.1f}%\n"
        
        md_content += f"\n### Top 5 by Risk-Adjusted Return\n\n"
        for i, row in enumerate(summary['top_by_risk_adjusted'], 1):
            md_content += f"{i}. **Alpha {row['alpha_id']}**: CAGR {row['cagr']:.1f}%, MaxDD {row['max_drawdown_pct']:.1f}%, Score {row['risk_adjusted_score']:.2f}\n"
    else:
        md_content += "**No strategies passed Stage 2 criteria.**\n\n"
        md_content += "This indicates that the current parameter space does not produce consistently profitable strategies over 3-year period.\n"
    
    md_content += "\n---\n\nGenerated by Alpha Search v2.0 Night Pipeline\n"
    
    md_path = REPORTS_DIR / "alpha_final_summary.md"
    with open(md_path, 'w') as f:
        f.write(md_content)
    logger.info(f"Saved Markdown: {md_path}")
    
    log_marker("SUMMARY_COMPLETE")
    
    # Log summary to console
    logger.info("\n" + md_content)


async def run_night_pipeline(top_n: int = DEFAULT_TOP_N):
    """Run complete overnight pipeline."""
    start_time = datetime.now()
    
    log_marker("NIGHT_PIPELINE_START")
    logger.info(f"Top N for Stage 2: {top_n}")
    logger.info(f"Start time: {start_time}")
    
    try:
        # Stage 1
        stage1_output = await run_stage1()
        
        # Select top N
        stage2_input = select_top_n_for_stage2(stage1_output, top_n)
        
        # Stage 2
        stage2_output = await run_stage2(stage2_input)
        
        # Final summary
        generate_final_summary(stage1_output, stage2_output)
        
        # Success
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 3600
        
        log_marker("NIGHT_PIPELINE_SUCCESS")
        logger.info(f"Duration: {duration:.2f} hours")
        logger.info(f"End time: {end_time}")
        
    except Exception as e:
        logger.error(f"\n{'='*70}")
        logger.error(f"NIGHT PIPELINE FAILED: {e}")
        logger.error(f"{'='*70}\n")
        raise


def main():
    parser = argparse.ArgumentParser(description="Alpha Search v2.0 - Night Pipeline")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N,
                       help=f"Number of top candidates for Stage 2 (default: {DEFAULT_TOP_N})")
    args = parser.parse_args()
    
    asyncio.run(run_night_pipeline(top_n=args.top_n))


if __name__ == "__main__":
    main()
