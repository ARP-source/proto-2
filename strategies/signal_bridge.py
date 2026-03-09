import pandas as pd
import numpy as np

class SignalTranslator:
    """
    Agent 9: Communicator - Signal Translator
    Interprets the continuous predictions output by the ML pipeline
    and normalizes them into discrete conviction vectors (Alpha signals).
    """
    
    @staticmethod
    def translate_to_alpha(predictions: pd.Series, top_n: int = 5) -> dict:
        """
        Converts raw N-day forward return predictions into bounded conviction scores.
        Only keeps the top N highest conviction long signals and bottom N short signals.
        Returns a dictionary mapping {symbol: conviction_weight}.
        """
        if predictions.empty:
            return {}
            
        # Convert Series back to dictionary for sorting (assuming index is symbols)
        pred_dict = predictions.to_dict()
        
        # Sort by predicted return (highest to lowest)
        sorted_preds = sorted(pred_dict.items(), key=lambda x: x[1], reverse=True)
        
        conviction_signals = {}
        total_assets = len(sorted_preds)
        
        # If we have very few assets (e.g., test environment), adjust N
        safe_top_n = min(top_n, total_assets // 2) if total_assets > 1 else total_assets
        
        if safe_top_n == 0:
            # Fallback if universe is tiny (e.g., 2 stocks)
            for sym, pred in sorted_preds:
                conviction_signals[sym] = 1.0 if pred > 0 else -1.0
            return conviction_signals

        # Extract Top N (Longs)
        top_longs = sorted_preds[:safe_top_n]
        # Extract Bottom N (Shorts)
        bottom_shorts = sorted_preds[-safe_top_n:]
        
        # Normalize conviction based on the magnitude of prediction (Softmax or proportional)
        # Using a simpler proportional rank weighting here for robustness
        for idx, (sym, pred) in enumerate(top_longs):
            if pred > 0:
                # Rank 0 gets highest conviction
                conviction_signals[sym] = 1.0 - (idx * (1.0 / safe_top_n))
                
        for idx, (sym, pred) in enumerate(reversed(bottom_shorts)):
            if pred < 0:
                # Rank 0 (most negative) gets lowest conviction target
                conviction_signals[sym] = -1.0 + (idx * (1.0 / safe_top_n))
                
        return conviction_signals
