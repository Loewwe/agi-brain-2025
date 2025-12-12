#!/usr/bin/env python3
"""
Start AGI-Brain Risk Advisor in Shadow Mode.

Usage:
    python scripts/start_shadow_mode.py --mock  # Test with mock data
    python scripts/start_shadow_mode.py         # Live with Stage6
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.risk.shadow_runner import ShadowRunner
from src.risk.advisor import configure_risk_advisor


async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="AGI-Brain Shadow Mode")
    parser.add_argument("--interval", type=int, default=5, help="Interval in minutes")
    parser.add_argument("--duration", type=float, default=None, help="Duration in hours")
    parser.add_argument("--mock", action="store_true", help="Use mock data")
    parser.add_argument("--no-telegram", action="store_true", help="Disable Telegram")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM (hard rules only)")
    
    args = parser.parse_args()
    
    # Configure advisor
    advisor = configure_risk_advisor(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        use_llm=not args.no_llm,
    )
    
    # Create runner
    runner = ShadowRunner(
        advisor=advisor,
        log_path=Path(__file__).parent.parent / "logs" / "shadow_decisions.jsonl",
        telegram_token=os.getenv("TELEGRAM_BOT_TOKEN"),
    )
    
    print("🧠 AGI-Brain Shadow Mode")
    print(f"   Interval: {args.interval} min")
    print(f"   Duration: {args.duration}h" if args.duration else "   Duration: ∞")
    print(f"   Mock mode: {args.mock}")
    print(f"   LLM: {not args.no_llm}")
    print(f"   Telegram: {not args.no_telegram}")
    print()
    print("Write /brain to the Telegram bot to start receiving updates!")
    print("Press Ctrl+C to stop.")
    print()
    
    await runner.run_continuous(
        interval_minutes=args.interval,
        duration_hours=args.duration,
        use_mock=args.mock,
        send_updates=not args.no_telegram,
    )


if __name__ == "__main__":
    asyncio.run(main())
