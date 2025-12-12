#!/usr/bin/env python3
"""
Canary Executor v1 — AGI Level 0.5 Guarded Autonomy

Selects safe candidates, enables canary execution in Stage 6.
Auto-stops on DD limit or Diagnostics degradation.

Usage:
  python3 scripts/canary_executor.py --auto    # Enable best candidate
  python3 scripts/canary_executor.py --stop    # Stop all canary
  python3 scripts/canary_executor.py --status  # Show status
"""
import sys
import json
import argparse
import urllib.request
from datetime import datetime
from pathlib import Path

# Config paths
CONFIG_DIR = Path(__file__).parent.parent / "config"
CANARY_CONFIG = CONFIG_DIR / "stage6_canary_config.json"
CANARY_POLICY = CONFIG_DIR / "canary_policy.json"
AGI_HISTORY = Path("/app/data/agi/canary_history.jsonl") if Path("/app").exists() else Path(__file__).parent.parent / "data" / "canary_history.jsonl"

# SuperBrain API base
API_BASE = "http://localhost:8001"


def fetch_json(url: str) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return {"error": str(e)}


def load_config(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_config(path: Path, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def log_event(event_type: str, data: dict):
    AGI_HISTORY.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "ts": datetime.now().isoformat(),
        "type": event_type,
        **data
    }
    with open(AGI_HISTORY, "a") as f:
        f.write(json.dumps(event) + "\n")
    print(f"[{event_type}] logged")


def check_diagnostics() -> tuple[bool, str]:
    diag = fetch_json(f"{API_BASE}/diag/summary")
    if "error" in diag:
        return False, f"Diagnostics error: {diag[error]}"
    
    status = diag.get("global_status", "unknown")
    if status != "ok":
        return False, f"Diagnostics not ok: {status}"
    
    return True, "Diagnostics OK"


def get_safe_candidates(policy: dict) -> list:
    candidates = fetch_json(f"{API_BASE}/alpha/auto/candidates")
    if "error" in candidates:
        return []
    
    safe = []
    for c in candidates.get("candidates", []):
        if not c.get("unsinkable_ok", False):
            continue
        if c.get("status") != "candidate":
            continue
        
        sharpe = c.get("sharpe") or 0
        dd = c.get("max_dd_pct") or -100
        trades = c.get("trades") or 0
        
        if sharpe < policy.get("min_sharpe", 1.5):
            continue
        if dd < policy.get("max_dd", -8.0):
            continue
        if trades < policy.get("min_trades", 50):
            continue
        
        safe.append(c)
    
    # Sort by Sharpe
    return sorted(safe, key=lambda x: x.get("sharpe", 0), reverse=True)


def enable_canary(candidate: dict, canary_config: dict):
    strategy = {
        "id": candidate["strategy_id"] + "_canary",
        "base_strategy": candidate["strategy_id"],
        "symbols": [candidate.get("symbol", "BTCUSDT")],
        "timeframe": candidate.get("timeframe", "15m"),
        "risk_per_trade": 0.25,
        "started_at": datetime.now().isoformat(),
        "sharpe": candidate.get("sharpe"),
        "max_dd_pct": candidate.get("max_dd_pct")
    }
    
    canary_config["enabled"] = True
    canary_config["strategies"] = [strategy]
    
    save_config(CANARY_CONFIG, canary_config)
    
    log_event("canary_started", {
        "strategy_id": candidate["strategy_id"],
        "sharpe": candidate.get("sharpe"),
        "max_dd_pct": candidate.get("max_dd_pct"),
        "symbol": candidate.get("symbol"),
        "timeframe": candidate.get("timeframe")
    })
    
    print(f"✅ Canary ENABLED: {candidate[strategy_id]}")
    print(f"   Sharpe: {candidate.get(sharpe)}, DD: {candidate.get(max_dd_pct)}%")


def stop_canary(canary_config: dict, reason: str):
    if not canary_config.get("enabled"):
        print("ℹ️ Canary already disabled")
        return
    
    strategies = canary_config.get("strategies", [])
    strategy_id = strategies[0]["base_strategy"] if strategies else "unknown"
    
    canary_config["enabled"] = False
    canary_config["strategies"] = []
    
    save_config(CANARY_CONFIG, canary_config)
    
    log_event("canary_stopped", {
        "strategy_id": strategy_id,
        "reason": reason
    })
    
    print(f"🛑 Canary STOPPED: {strategy_id}")
    print(f"   Reason: {reason}")


def show_status():
    canary_config = load_config(CANARY_CONFIG)
    policy = load_config(CANARY_POLICY)
    
    print("\n=== Canary Executor Status ===")
    print(f"Enabled: {canary_config.get(enabled, False)}")
    print(f"Max strategies: {canary_config.get(max_canary_strategies, 1)}")
    print(f"Max DD: {canary_config.get(max_canary_dd_pct, -5.0)}%")
    
    strategies = canary_config.get("strategies", [])
    if strategies:
        print("\nActive canary strategies:")
        for s in strategies:
            print(f"  • {s.get(id)}: {s.get(symbols)} / {s.get(timeframe)}")
            print(f"    Started: {s.get(started_at)}")
            print(f"    Sharpe: {s.get(sharpe)}, DD: {s.get(max_dd_pct)}%")
    else:
        print("\nNo active canary strategies")
    
    # Check candidates
    safe = get_safe_candidates(policy)
    print(f"\nSafe candidates available: {len(safe)}")
    for c in safe[:3]:
        print(f"  • {c[strategy_id]}: Sharpe={c.get(sharpe)}, DD={c.get(max_dd_pct)}%")
    
    # Check diagnostics
    ok, msg = check_diagnostics()
    print(f"\nDiagnostics: {✅ if ok else ❌} {msg}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Canary Executor v1")
    parser.add_argument("--auto", action="store_true", help="Auto-enable best candidate")
    parser.add_argument("--stop", action="store_true", help="Stop all canary")
    parser.add_argument("--status", action="store_true", help="Show status")
    parser.add_argument("--check", action="store_true", help="Check and auto-stop if needed")
    
    args = parser.parse_args()
    
    canary_config = load_config(CANARY_CONFIG)
    policy = load_config(CANARY_POLICY)
    
    if args.status:
        show_status()
        return
    
    if args.stop:
        stop_canary(canary_config, "manual")
        return
    
    if args.check:
        # Auto-stop check
        ok, msg = check_diagnostics()
        if not ok and canary_config.get("enabled"):
            stop_canary(canary_config, f"diagnostics_{msg}")
        return
    
    if args.auto:
        # Check diagnostics first
        ok, msg = check_diagnostics()
        if not ok:
            print(f"❌ Cannot enable canary: {msg}")
            return
        
        # Get safe candidates
        safe = get_safe_candidates(policy)
        if not safe:
            print("❌ No safe candidates available")
            return
        
        # Check if already running
        if canary_config.get("enabled"):
            print("ℹ️ Canary already running")
            show_status()
            return
        
        # Enable best candidate
        best = safe[0]
        enable_canary(best, canary_config)


if __name__ == "__main__":
    main()
