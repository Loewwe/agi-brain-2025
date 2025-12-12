"""Candidate Generator P1 — Generate Stage6 candidate configs from incubate strategies."""
import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List

CANDIDATES_DIR = Path("/root/Alien.agi/agi_brain/config/candidates")
RESULTS_DIR = Path("/root/Alien.agi/agi_brain/data/auto_research")
UNSINKABLE_CONFIG = Path("/root/Alien.agi/agi_brain/config/unsinkable_limits.json")


@dataclass
class Candidate:
    candidate_id: str
    strategy_id: str
    engine: str
    symbol: str
    timeframe: str
    params: Dict[str, Any]
    
    # Metrics from backtest
    sharpe: float
    max_dd_pct: float
    trades: int
    pnl_pct: float
    
    # Status
    status: str = "candidate"  # candidate, approved, sandbox, canary, rejected
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None
    
    # UNSINKABLE check
    unsinkable_ok: bool = False
    unsinkable_notes: str = ""
    
    created_at: str = ""
    
    def to_dict(self):
        return asdict(self)


def load_unsinkable_limits() -> Dict:
    """Load UNSINKABLE limits."""
    if UNSINKABLE_CONFIG.exists():
        with open(UNSINKABLE_CONFIG) as f:
            return json.load(f)
    # Defaults
    return {
        "max_daily_loss_pct": 2.0,
        "max_position_size_pct": 5.0,
        "max_dd_allowed_pct": 10.0,
        "min_sharpe_for_canary": 1.5,
        "max_leverage": 10
    }


def check_unsinkable(candidate: Candidate, limits: Dict) -> tuple[bool, str]:
    """Check if candidate passes UNSINKABLE constraints."""
    issues = []
    
    if candidate.max_dd_pct < -limits.get("max_dd_allowed_pct", 10):
        issues.append(f"DD {candidate.max_dd_pct}% exceeds limit -{limits[max_dd_allowed_pct]}%")
    
    if candidate.sharpe < limits.get("min_sharpe_for_canary", 1.5):
        issues.append(f"Sharpe {candidate.sharpe} below {limits[min_sharpe_for_canary]}")
    
    if issues:
        return False, "; ".join(issues)
    return True, "All UNSINKABLE checks passed"


def generate_candidate_from_result(result: Dict) -> Optional[Candidate]:
    """Generate candidate from backtest result."""
    if result.get("rating") != "incubate":
        return None
    
    candidate_id = "cand_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + result["strategy_id"]
    
    candidate = Candidate(
        candidate_id=candidate_id,
        strategy_id=result["strategy_id"],
        engine=result.get("engine", "unknown"),
        symbol=result.get("symbol", "BTCUSDT"),
        timeframe=result.get("timeframe", "15m"),
        params={},
        sharpe=result.get("sharpe", 0),
        max_dd_pct=result.get("max_dd_pct", 0),
        trades=result.get("trades", 0),
        pnl_pct=result.get("pnl_pct", 0),
        created_at=datetime.now().isoformat()
    )
    
    # Check UNSINKABLE
    limits = load_unsinkable_limits()
    ok, notes = check_unsinkable(candidate, limits)
    candidate.unsinkable_ok = ok
    candidate.unsinkable_notes = notes
    
    return candidate


def save_candidate(candidate: Candidate):
    """Save candidate to config/candidates/."""
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    
    path = CANDIDATES_DIR / (candidate.candidate_id + ".json")
    with open(path, "w") as f:
        json.dump(candidate.to_dict(), f, indent=2)
    
    return path


def load_candidate(candidate_id: str) -> Optional[Candidate]:
    """Load candidate by ID."""
    path = CANDIDATES_DIR / (candidate_id + ".json")
    if not path.exists():
        return None
    
    with open(path) as f:
        data = json.load(f)
    return Candidate(**data)


def list_candidates(status: str = None) -> List[Candidate]:
    """List all candidates."""
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    candidates = []
    
    for f in CANDIDATES_DIR.glob("cand_*.json"):
        try:
            with open(f) as fp:
                c = Candidate(**json.load(fp))
                if status is None or c.status == status:
                    candidates.append(c)
        except:
            continue
    
    return sorted(candidates, key=lambda x: x.created_at, reverse=True)


def approve_candidate(candidate_id: str, approved_by: str = "manual") -> bool:
    """Approve candidate for sandbox/canary."""
    candidate = load_candidate(candidate_id)
    if not candidate:
        return False
    
    if not candidate.unsinkable_ok:
        return False
    
    candidate.status = "approved"
    candidate.approved_by = approved_by
    candidate.approved_at = datetime.now().isoformat()
    
    save_candidate(candidate)
    return True


def promote_to_sandbox(candidate_id: str) -> bool:
    """Promote approved candidate to sandbox."""
    candidate = load_candidate(candidate_id)
    if not candidate or candidate.status != "approved":
        return False
    
    candidate.status = "sandbox"
    save_candidate(candidate)
    return True
