try:
    from src.world.model import RiskLevel, RiskViolation
    print("Import successful")
except Exception as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()
