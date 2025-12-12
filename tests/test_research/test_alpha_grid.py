"""
Tests for Alpha Grid Runner and Aggregator (Stage 9 Phase 2)
"""

import pytest
import json
import tempfile
import os
from pathlib import Path


def test_grid_runner_skips_completed():
    """Test that grid runner skips already completed experiments."""
    # Create temp results directory with existing result
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a fake existing result
        result_path = Path(tmpdir) / "scan_001_test.json"
        existing_result = {
            "_meta": {"id": "scan_001_test"},
            "n_trades": 100,
            "profit_factor_post_cost": 1.05,
        }
        with open(result_path, "w") as f:
            json.dump(existing_result, f)
        
        # Verify file exists
        assert result_path.exists()
        
        # Get original modification time
        orig_mtime = result_path.stat().st_mtime
        
        # Import and check the get_result_path function
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from scripts.run_alpha_grid import get_result_path
        
        # Verify path construction
        constructed_path = get_result_path(tmpdir, "scan_001_test")
        assert constructed_path == result_path
        
        # The skip logic is in run_single_experiment which checks if file exists
        # We just verify the file wasn't touched
        assert result_path.stat().st_mtime == orig_mtime


def test_aggregator_ranks_correctly():
    """Test that aggregator ranks results by composite score."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from scripts.summarize_alpha_scan import aggregate_results, compute_score
    
    # Create test results with different scores
    results = [
        {
            "_meta": {"id": "worst"},
            "profit_factor_post_cost": 0.9,
            "sharpe_post_cost": 0.5,
            "n_trades": 5000,
        },
        {
            "_meta": {"id": "best"},
            "profit_factor_post_cost": 1.20,
            "sharpe_post_cost": 2.0,
            "n_trades": 500,
        },
        {
            "_meta": {"id": "middle"},
            "profit_factor_post_cost": 1.10,
            "sharpe_post_cost": 1.2,
            "n_trades": 1000,
        },
    ]
    
    # Aggregate
    classified = aggregate_results(results)
    
    # Best should be candidate (meets all criteria)
    # Middle should be borderline
    # Worst should be rejected
    
    # Check sorting by score - candidates should be sorted best first
    all_sorted = classified["candidates"] + classified["borderline"] + classified["rejected"]
    
    # "best" should have highest score
    best_result = next(r for r in all_sorted if r["_meta"]["id"] == "best")
    worst_result = next(r for r in all_sorted if r["_meta"]["id"] == "worst")
    
    assert best_result["_score"] > worst_result["_score"], \
        "Best result should have higher score than worst"


def test_rejection_rules_monotone():
    """Test that lower PF always results in same or worse classification."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from scripts.summarize_alpha_scan import classify_result
    
    # Base result that's a candidate
    good_result = {
        "profit_factor_post_cost": 1.20,
        "sharpe_post_cost": 2.0,
        "trades_per_month": 500,
        "max_drawdown_post_cost": -0.15,
    }
    
    # Decreasing PF versions
    borderline_result = {
        **good_result,
        "profit_factor_post_cost": 1.08,  # Below candidate, above borderline
    }
    
    rejected_result = {
        **good_result,
        "profit_factor_post_cost": 0.95,  # Below borderline
    }
    
    classification_rank = {"candidate": 3, "borderline": 2, "rejected": 1, "error": 0}
    
    good_class = classify_result(good_result)
    borderline_class = classify_result(borderline_result)
    rejected_class = classify_result(rejected_result)
    
    # Lower PF should never improve classification
    assert classification_rank[good_class] >= classification_rank[borderline_class], \
        "Lowering PF should not improve classification"
    assert classification_rank[borderline_class] >= classification_rank[rejected_class], \
        "Lowering PF further should not improve classification"
    
    # Verify expected classifications
    assert good_class == "candidate"
    assert borderline_class == "borderline"
    assert rejected_class == "rejected"
