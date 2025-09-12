# risk_assessment.py
import random
import statistics
from .utils import log_event

class RiskAssessment:
    """
    Quantifies financial risks.
    Provides Monte Carlo simulations and Value-at-Risk (VaR).
    """

    def monte_carlo_simulation(self, base_value, volatility, trials=1000):
        outcomes = [
            base_value * (1 + random.gauss(0, volatility))
            for _ in range(trials)
        ]
        mean = statistics.mean(outcomes)
        stdev = statistics.pstdev(outcomes)
        log_event("Monte Carlo simulation complete")
        return {"mean": mean, "stdev": stdev, "min": min(outcomes), "max": max(outcomes)}

    def value_at_risk(self, portfolio, confidence=0.95):
        """
        Calculate Value at Risk (VaR).
        """
        losses = sorted([-v for v in portfolio])
        index = int((1-confidence) * len(losses))
        var = losses[index] if index < len(losses) else 0
        log_event(f"Calculated VaR at {confidence} confidence: {var}")
        return var
