"""
Filter Bypass Patch for Stage6 Experiment

This patch modifies validate_entry in the Decision Pipeline
to bypass all entry filters when experiment mode is enabled.

Apply by adding this code to core/decision/decision_pipeline.py
"""

# Add this method to DecisionPipeline class:
"""
def _bypass_filters_check(self) -> bool:
    '''Check if filters should be bypassed for experiment.'''
    if hasattr(self, 'experiment') and self.experiment:
        return self.experiment.get('filters_override_enabled', False)
    return False
"""

# Modify validate_entry to add bypass at the beginning:
"""
def validate_entry(self, symbol, df, side, cap):
    '''Validate entry conditions.'''
    
    # === EXPERIMENT BYPASS ===
    if self._bypass_filters_check():
        import logging
        logging.getLogger(__name__).warning(
            f"⚡ TEMP OVERRIDE: All entry filters DISABLED for {symbol}"
        )
        # Return immediate approval
        suggested_size = cap * 0.02  # 2% of capital
        return True, suggested_size, ["FILTERS_OVERRIDDEN"]
    
    # Original filter logic continues below...
"""

# Also add bypass for Unsinkable pre_trade_check in orchestrator.py run_cycle:
"""
# Find where pre_trade_check is called and wrap it:
if hasattr(self, 'experiment') and self.experiment.get('unsinkable_pretrade_override', False):
    logger.warning("⚡ TEMP OVERRIDE: Unsinkable pre_trade_check DISABLED")
    allowed = True
    reason = "UNSINKABLE_PRETRADE_OVERRIDE"
else:
    allowed, reason = unsinkable.pre_trade_check(...)
"""
