
import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.brain.pipeline import AutoResearchPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    parser = argparse.ArgumentParser(description="Auto-Research Brain (Nightly Cycle)")
    parser.add_argument("--mode", type=str, default="normal", choices=["normal", "aggressive", "defensive", "test"], help="Risk mode for screening")
    parser.add_argument("--hours", type=float, default=8.0, help="Max runtime in hours")
    parser.add_argument("--candidates", type=int, default=1000, help="Number of candidates to generate")
    
    args = parser.parse_args()
    
    logger.info(f"Starting Auto-Research Brain in {args.mode} mode for {args.hours} hours.")
    logger.info(f"Target candidates: {args.candidates}")
    
    # Initialize Pipeline
    # TODO: Pass args to pipeline configuration
    pipeline = AutoResearchPipeline()
    
    # Run Cycle
    await pipeline.run_nightly_cycle()

if __name__ == "__main__":
    asyncio.run(main())
