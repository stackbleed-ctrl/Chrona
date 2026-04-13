"""
examples/simulate.py
Run a what-if simulation: what happens if demand scales up 20%?

    python examples/simulate.py
"""

import numpy as np
from chrona import ChronaPredictor

rng = np.random.default_rng(42)
series = list(10 + np.cumsum(rng.normal(0, 0.3, 120)))

predictor = ChronaPredictor.from_scratch()

result = predictor.simulate(
    series=series,
    interventions=[
        {"type": "scale", "factor": 1.2, "start": 90, "end": 120},
    ],
    horizon=48,
)

print("Base forecast (first 5 steps):")
print(result["base"].mean[:5].round(3))

print("\nScenario forecast (first 5 steps):")
print(result["scenario"].mean[:5].round(3))

print("\nDelta (scenario − base):")
print(result["delta_mean"][:5].round(3))
