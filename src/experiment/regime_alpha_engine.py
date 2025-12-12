
import pandas as pd
import structlog
from pathlib import Path
from typing import Dict

from .alpha_engine import AlphaEngine, AlphaConfig
from .regime_detector import MarketRegimeDetector, RegimeConfig, Regime

logger = structlog.get_logger()

class RegimeAwareAlphaEngine:
    """
    AlphaEngine that switches models based on market regime.
    
    Wraps multiple AlphaEngine instances (one per regime).
    """
    
    def __init__(self, models_dir: Path | str, regime_config: RegimeConfig | None = None, allowed_regimes: list[str] | None = None):
        self.models_dir = Path(models_dir)
        self.detector = MarketRegimeDetector(regime_config or RegimeConfig(adx_threshold=25.0))
        self.models: Dict[str, AlphaEngine] = {}
        self.allowed_regimes = allowed_regimes  # If None, allow all
        
        # Load models
        self._load_models()
        
    def _load_models(self):
        """Load models for each regime."""
        for regime in Regime:
            model_path = self.models_dir / f"model_{regime.value}.pkl"
            if model_path.exists():
                try:
                    self.models[regime.value] = AlphaEngine.load(model_path)
                    logger.info("regime_engine.loaded_model", regime=regime.value)
                except Exception as e:
                    logger.error("regime_engine.load_failed", regime=regime.value, error=str(e))
            else:
                logger.warning("regime_engine.model_missing", regime=regime.value, path=str(model_path))
                
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate alpha predictions using regime-specific models.
        """
        if df.empty:
            return df
            
        # Detect regimes
        regimes = self.detector.detect_all(df)
        
        # Prepare result dataframe
        results = []
        
        # Group by regime and predict
        for regime_value, group_df in df.groupby(regimes):
            # Check allowed regimes
            if self.allowed_regimes is not None and regime_value not in self.allowed_regimes:
                # Neutral prediction
                pred = group_df.copy()
                pred["alpha_score"] = 0.5
                pred["alpha_direction"] = 0
                pred["alpha_confidence"] = 0.0
                pred["regime"] = regime_value
                results.append(pred)
                continue

            model = self.models.get(regime_value)
            
            if model:
                # Predict using specific model
                pred = model.predict(group_df)
                # Add regime info
                pred["regime"] = regime_value
                results.append(pred)
            else:
                # Fallback: No prediction (neutral)
                logger.warning("regime_engine.no_model", regime=regime_value)
                pred = group_df.copy()
                pred["alpha_score"] = 0.5
                pred["alpha_direction"] = 0
                pred["alpha_confidence"] = 0.0
                pred["regime"] = regime_value
                results.append(pred)
                
        # Combine and restore order
        if not results:
            return df
            
        final_df = pd.concat(results)
        final_df = final_df.reindex(df.index)
        
        return final_df

    def predict_single(self, features: dict[str, float]) -> dict:
        """Predict for a single row."""
        # Detect regime
        regime = self.detector.detect(features)
        
        # Check allowed regimes
        if self.allowed_regimes is not None and regime.value not in self.allowed_regimes:
            return {
                "alpha_score": 0.5,
                "alpha_direction": 0,
                "alpha_confidence": 0.0,
                "regime": regime.value
            }

        model = self.models.get(regime.value)
        
        if model:
            result = model.predict_single(features)
            result["regime"] = regime.value
            return result
        else:
            return {
                "alpha_score": 0.5,
                "alpha_direction": 0,
                "alpha_confidence": 0.0,
                "regime": regime.value
            }

    @property
    def feature_columns(self) -> list[str]:
        """Return feature columns used by models + regime detection cols."""
        cols = set()
        if self.models:
            cols.update(next(iter(self.models.values())).feature_columns)
        
        # Add regime detection columns
        cols.update(["close", "ema200", "adx"])
        
        return list(cols)
    
    @property
    def config(self) -> AlphaConfig | None:
        """Return config of first model."""
        if self.models:
            return next(iter(self.models.values())).config
        return None
