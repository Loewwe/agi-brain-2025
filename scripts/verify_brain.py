import asyncio
import sys
from pathlib import Path

# Add project root to path (alien-agi)
sys.path.append(str(Path(__file__).parent.parent.parent))

from agi_brain.src.planning.brain import configure_brain, get_brain, BrainMode
from agi_brain.src.planning.executor import configure_executor, get_executor
from agi_brain.src.tools.registry import configure_tools_registry, get_tools_registry
from agi_brain.src.tools.base_tools import register_base_tools
from agi_brain.src.memory.episodic import configure_episodic_memory
from agi_brain.src.memory.semantic import configure_semantic_memory
from agi_brain.src.perception.trading_adapter import configure_trading_adapter, AdapterMode
from agi_brain.src.world.snapshot_builder import configure_snapshot_builder
from agi_brain.src.safety.acl import configure_acl
from agi_brain.src.safety.audit import configure_audit_logger
from agi_brain.src.safety.laws import get_core_laws

async def verify_components():
    print("🚀 Starting AGI-Brain verification...")
    
    # 1. Initialize Components
    print("\n1. Initializing components...")
    try:
        configure_episodic_memory()
        configure_semantic_memory()
        trading_adapter = configure_trading_adapter(mode=AdapterMode.STUB)
        configure_snapshot_builder(
            trading_adapter=trading_adapter,
            episodic_memory=None,
            semantic_memory=None,
            owner_profile=None
        )
        configure_acl()
        configure_audit_logger()
        
        # Config path
        base_path = Path(__file__).parent.parent
        config_path = base_path / "src" / "configs" / "tools_registry.yaml"
        
        registry = configure_tools_registry(config_path)
        register_base_tools(registry)
        
        configure_brain(mode=BrainMode.L0_READONLY)
        configure_executor()
        
        print("✅ Components initialized successfully")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return

    # 2. Verify Tools
    print("\n2. Verifying tools registry...")
    registry = get_tools_registry()
    tools = registry.list_all()
    print(f"   Found {len(tools)} tools: {', '.join(tools)}")
    if "world.snapshot" in tools and "trading.fetch_positions" in tools:
        print("✅ Base tools registered")
    else:
        print("❌ Missing base tools")

    # 3. Verify Laws
    print("\n3. Verifying core laws...")
    laws = get_core_laws()
    supreme = laws.get_supreme_law()
    print(f"   Supreme Law: {supreme.name}")
    print(f"   Total laws: {len(laws.get_all_laws())}")
    print("✅ Laws loaded")

    # 4. Verify Brain & Planning (Stub)
    print("\n4. Verifying Brain & Planning...")
    brain = get_brain()
    executor = get_executor()
    
    goal = "Check trading status"
    print(f"   Goal: {goal}")
    
    # Analyze
    analysis = await brain.understand_goal(goal)
    print(f"   Analysis: {analysis.understood_intent} (Tools: {analysis.required_tools})")
    
    # Plan
    plan = await brain.create_plan(goal, analysis)
    print(f"   Plan created: {plan.total_steps} steps")
    
    # Execute
    print("   Executing plan...")
    result = await executor.execute(plan)
    
    if result.success:
        print(f"✅ Execution successful: {result.summary}")
    else:
        print(f"❌ Execution failed: {result.summary}")

    print("\n✨ Verification complete!")

if __name__ == "__main__":
    asyncio.run(verify_components())
