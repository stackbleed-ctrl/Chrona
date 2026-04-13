"""
examples/zero_shot_forecast.py
Run a zero-shot forecast on synthetic data — no training required.

    python examples/zero_shot_forecast.py
"""

import numpy as np
from chrona import ChronaPredictor

# 1. Generate some fake data (replace with your own)
t = np.linspace(0, 6 * np.pi, 300)
series = (
    10 * np.sin(t)
    + 3 * np.sin(3 * t)
    + np.random.default_rng(0).normal(0, 0.5, 300)
    + np.linspace(0, 4, 300)          # trend
)

# 2. Load predictor (untrained weights — swap checkpoint path for real weights)
predictor = ChronaPredictor.from_scratch()

# 3. Forecast
result = predictor.predict(series, horizon=48)

# 4. Print
df = result.to_dataframe()
print(df.to_string(index=False))

# 5. Optional: plot
try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(series[-100:], color="#7777ff", label="history")
    h = len(result.mean)
    x_fut = range(100, 100 + h)
    ax.plot(x_fut, result.mean, color="#00ffa3", label="forecast (p50)")
    ax.fill_between(x_fut, result.p10(), result.p90(), alpha=0.2, color="#00ffa3", label="p10–p90")
    ax.axvline(100, color="white", alpha=0.3, linestyle="--")
    ax.set_facecolor("#0a0f18")
    fig.patch.set_facecolor("#0a0f18")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#0a0f18", labelcolor="white")
    plt.tight_layout()
    plt.savefig("forecast.png", dpi=150)
    print("\nPlot saved → forecast.png")
except ImportError:
    print("\n(pip install matplotlib to generate a plot)")
