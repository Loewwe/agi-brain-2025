
import asyncio
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any


import yaml
from scripts.mass_screening_runner import run_screening_programmatic

from .hybrid_portfolio_builder import HybridPortfolioBuilder

logger = logging.getLogger(__name__)

class AutoResearchPipeline:
    """
    Nightly Auto-Research Pipeline.
    Phases:
    1. Generation (Stage 0) -> Handled by Mass Screening for now (Catalog)
    2. Mass Screening (Quick Filter)
    3. Deep Validation (Backtest & Stress Test)
    4. Hybrid Construction (Portfolio Assembly)
    """
    
    def __init__(self, config_path: str = "src/configs/auto_research.json"):
        self.config_path = config_path
        self.builder = HybridPortfolioBuilder()
        self.core_config = self._load_yaml("src/configs/daily_alpha_core.yml")
        self.event_config = self._load_yaml("src/configs/event_strategies.yml")
        
    def _load_yaml(self, path: str) -> Dict:
        try:
            with open(path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config {path}: {e}")
            return {}
        
    async def run_nightly_cycle(self):
        """Execute the full nightly research cycle."""
        logger.info("Starting Nightly Auto-Research Cycle...")
        cycle_id = datetime.now().strftime("%Y%m%d")
        
        try:
            # Phase 1 & 2: Mass Screening (Generation & Screening combined for now)
            logger.info(f"[{cycle_id}] Phase 1 & 2: Mass Screening")
            screened_candidates = await self._phase_screening()
            logger.info(f"Screened down to {len(screened_candidates)} candidates.")
            
            # Phase 3: Deep Validation
            logger.info(f"[{cycle_id}] Phase 3: Deep Validation")
            validated_strategies = await self._phase_validation(screened_candidates)
            logger.info(f"Validated {len(validated_strategies)} strategies.")
            
            # Phase 4: Hybrid Construction
            logger.info(f"[{cycle_id}] Phase 4: Hybrid Construction")
            portfolio = await self._phase_construction(validated_strategies)
            
            # Save Artifacts
            self._save_results(cycle_id, portfolio)
            logger.info(f"[{cycle_id}] Cycle Complete. Portfolio saved.")
            
        except Exception as e:
            logger.error(f"[{cycle_id}] Cycle Failed: {e}", exc_info=True)
            
    async def _phase_screening(self) -> List[Dict]:
        """Run Mass Screening to find candidates."""
        # Run the screening runner programmatically
        # It returns a list of dicts with metrics
        results = run_screening_programmatic() 
        
        candidates = []
        for r in results:
            # Classify and Filter
            cls = self._classify_strategy(r)
            if cls:
                r["class"] = cls
                candidates.append(r)
                
        return candidates

    def _classify_strategy(self, metrics: Dict) -> str:
        """Classify as A_CORE, A_EVENT or None based on configs."""
        # Check A_CORE
        if (metrics["avg_return"] >= self.core_config.get("min_avg_daily_return", 0.007) and
            metrics["profit_factor"] >= self.core_config.get("min_pf", 1.5) and
            metrics["win_rate"] >= self.core_config.get("min_win_rate", 0.58) and
            metrics["max_drawdown"] <= self.core_config.get("max_drawdown", 0.25) and
            metrics["total_events"] >= self.core_config.get("min_total_trades", 150)):
            return "A_CORE"
            
        # Check A_EVENT
        if (metrics["profit_factor"] >= self.event_config.get("min_pf", 1.8) and
            metrics["win_rate"] >= self.event_config.get("min_win_rate", 0.60) and
            metrics["max_drawdown"] <= self.event_config.get("max_drawdown", 0.35) and
            metrics["total_events"] >= self.event_config.get("min_total_trades", 30)):
            return "A_EVENT"
            
        return None

    async def _phase_validation(self, candidates: List[Dict]) -> List[Dict]:
        """Deep validation and stress testing."""
        # For now, we trust the Mass Screening metrics as "Validation" 
        # In future, this will run OOS / Stress tests.
        # We just calculate a hybrid score here.
        
        validated = []
        for c in candidates:
            # Simple score: PF * WR
            score = c["profit_factor"] * c["win_rate"]
            c["hybrid_score"] = score
            validated.append(c)
            
        return validated

    async def _phase_construction(self, strategies: List[Dict]) -> Dict:
        """Construct hybrid portfolio."""
        return self.builder.build(strategies)

    def _save_results(self, cycle_id: str, portfolio: Dict):
        """Save results to JSON and Markdown."""
        output_dir = Path("results/auto_research")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        with open(output_dir / f"hybrid_portfolio_{cycle_id}.json", "w") as f:
            json.dump(portfolio, f, indent=4)
            
        # Save Report
        with open(output_dir / f"morning_report_{cycle_id}.md", "w") as f:
            f.write(f"# Morning Report {cycle_id}\n\n")
            f.write("## Portfolio Summary\n")
            f.write(f"- Total Strategies: {portfolio['meta']['total_count']}\n")
            f.write(f"- Core Allocation: {portfolio['meta']['core_allocation']:.0%}\n")
            f.write(f"- Event Allocation: {portfolio['meta']['event_allocation']:.0%}\n")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pipeline = AutoResearchPipeline()
    asyncio.run(pipeline.run_nightly_cycle())
