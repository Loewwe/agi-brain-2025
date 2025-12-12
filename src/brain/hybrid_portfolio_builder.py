
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class HybridPortfolioBuilder:
    """
    Constructs a hybrid portfolio from validated strategies.
    Combines A_CORE (Daily Alpha) and A_EVENT (Event Alpha).
    """
    
    def __init__(self):
        self.min_core_weight = 0.5
        self.max_strategy_weight = 0.25
        self.target_strategies_count = (5, 15) # Min, Max
        
    def build(self, validated_strategies: List[Dict]) -> Dict[str, Any]:
        """
        Select strategies and allocate weights.
        """
        # 1. Classify Strategies
        a_core = [s for s in validated_strategies if s.get("class") == "A_CORE"]
        a_event = [s for s in validated_strategies if s.get("class") == "A_EVENT"]
        
        # 2. Selection (Simple Top-N for now)
        # Sort by hybrid score (assuming it exists)
        a_core.sort(key=lambda x: x.get("hybrid_score", 0), reverse=True)
        a_event.sort(key=lambda x: x.get("hybrid_score", 0), reverse=True)
        
        selected_core = a_core[:8] # Max 8 Core
        selected_event = a_event[:7] # Max 7 Event
        
        if not selected_core and not selected_event:
            logger.warning("No valid strategies found for portfolio.")
            return {"strategies": [], "meta": {"status": "empty"}}
            
        # 3. Weight Allocation
        # Simple allocation: Core gets 60%, Event gets 40% (normalized)
        total_core_score = sum(s.get("hybrid_score", 1) for s in selected_core)
        total_event_score = sum(s.get("hybrid_score", 1) for s in selected_event)
        
        portfolio_strategies = []
        
        # Allocate Core
        core_allocation = 0.6
        if not selected_event: core_allocation = 1.0
        
        for s in selected_core:
            weight = (s.get("hybrid_score", 1) / total_core_score) * core_allocation
            weight = min(weight, self.max_strategy_weight)
            s["recommended_capital_share"] = round(weight, 4)
            portfolio_strategies.append(s)
            
        # Allocate Event
        event_allocation = 0.4
        if not selected_core: event_allocation = 1.0
        
        for s in selected_event:
            weight = (s.get("hybrid_score", 1) / total_event_score) * event_allocation
            weight = min(weight, self.max_strategy_weight)
            s["recommended_capital_share"] = round(weight, 4)
            portfolio_strategies.append(s)
            
        return {
            "strategies": portfolio_strategies,
            "meta": {
                "core_count": len(selected_core),
                "event_count": len(selected_event),
                "total_count": len(portfolio_strategies),
                "core_allocation": core_allocation,
                "event_allocation": event_allocation
            }
        }
