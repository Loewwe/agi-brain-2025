import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.planning.brain import configure_brain, get_brain
from src.perception.trading_adapter import configure_trading_adapter, AdapterMode
from src.world.snapshot_builder import configure_snapshot_builder
from src.memory.owner_profile import OwnerProfile

async def main():
    # 1. Configure Components
    # Use Fixture Adapter with our new file
    positions_path = Path("fixtures/trading/user_positions.json").absolute()
    
    adapter = configure_trading_adapter(
        mode=AdapterMode.STUB  # We'll manually inject the fixture data via the class
    )
    # Hack: Replace the internal adapter with Fixture adapter
    from src.perception.trading_adapter import FixtureTradingAdapter
    global _trading_adapter
    import src.perception.trading_adapter as ta
    ta._trading_adapter = FixtureTradingAdapter(positions_path=str(positions_path))
    
    # Configure Tools Registry
    from src.tools.registry import configure_tools_registry
    from src.tools.base_tools import register_base_tools
    
    config_path = Path("src/configs/tools_registry.yaml").absolute()
    registry = configure_tools_registry(config_path)
    register_base_tools(registry)
    
    # Configure Brain
    configure_brain(
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        model="gpt-4o-mini"
    )
    
    brain = get_brain()
    
    # 2. Run Analysis
    print("🧠 Analyzing Portfolio Snapshot...")
    print(f"Loading positions from: {positions_path}")
    
    goal = "Проанализируй текущий портфель (XRP, DOGE, SOL, AVAX). Оцени риски и дай рекомендации по Stop-Loss."
    
    analysis = await brain.understand_goal(goal, client_id="owner")
    
    print("\n📋 Analysis Result:")
    print("-" * 50)
    print(f"Confidence: {analysis.confidence}")
    print(f"Complexity: {analysis.estimated_complexity}")
    print("-" * 50)
    
    # Create a plan manually to avoid LLM hallucinations
    from src.planning.brain import PlanResult, PlanStep
    from datetime import datetime
    
    steps = [
        PlanStep(
            id="step_1",
            tool="trading.fetch_positions",
            args={},
            level=0,
            description="Fetch current positions"
        ),
        PlanStep(
            id="step_2",
            tool="trading.analyze_risk",
            args={"include_recommendations": True},
            level=0,
            description="Analyze risk",
            depends_on=["step_1"]
        )
    ]
    
    plan_result = PlanResult(
        plan_id=f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        goal=goal,
        steps=steps,
        total_steps=len(steps),
        requires_approval=False,
        approval_reason=None,
        estimated_duration_seconds=10
    )

    # Execute the plan (simulation)
    from src.planning.executor import configure_executor, get_executor
    configure_executor()
    executor = get_executor()
    
    result = await executor.execute(plan_result)
    
    print("\n📝 Execution Summary:")
    print(result.summary)
    if result.output:
        print("\nOutput Details:")
        # Pretty print output if it's a dict
        import json
        try:
            print(json.dumps(result.output, indent=2, ensure_ascii=False))
        except:
            print(result.output)

if __name__ == "__main__":
    asyncio.run(main())
