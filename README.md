# PyGLAM

**PyGLAM** is a high-performance Python framework for emulating probability distributions using **Generalized Lambda Distributions (GLD)**.

## Installation

```bash
pip install pyglam
```

## Quick Start
Here is a simple example of how to fit a distribution and generate new data using the FKML parameterization:

```python
import numpy as np
from pyglam import GlamFKML

# 1. Generate sample data (e.g., Normal Distribution)
x = np.random.normal(0, 1, 50000)
x_vals = np.linspace(-4, 4, 500)

# 2. Fit the Lambdas (The "Emulator" way)
g = GlamFKML()
sol = g.fit_lambdas(x, method="least_squares") # You can also use method="root"
print(f"Estimated Lambdas: {sol.x}")

# 3. Use the emulated model
# Initialize with the optimized lambdas
emulator = GlamFKML(*sol.x)

rvs_glam = emulator.rvs(size=1000)          # Random variates
pdf_glam = emulator.pdf(x_vals)             # Probability Density Function
cdf_glam = emulator.cdf(x_vals)             # Cumulative Distribution Function
ppf_glam = emulator.ppf(np.linspace(0.01, 0.99, 100)) # Percent Point Function
```

## Measuring how good the emulation is

`Performance` scores a predicted sample against a reference sample. Hand it the raw data and the
sample you generated with `rvs()`, and it reports the agreement between the two distributions:

```python
from pyglam import Performance

report = Performance().performance(x, rvs_glam)

print(report['kl_divergence'])  # 0.001284  - Kullback-Leibler divergence
print(report['ks_statistic'])   # 0.003820  - two-sample Kolmogorov-Smirnov
print(report['ks_pvalue'])      # 1.000000  - > 0.05 means agreement is not rejected
print(report['wasserstein'])    # 0.005086  - earth mover's distance, in data units
print(report['r2_pdf'])         # 0.999184  - R2 between the two densities
```

The report also carries `rel_err_p5`, `rel_err_p50`, `rel_err_p95` and the arrays `x_grid`,
`pdf_true` and `pdf_pred` for plotting. No metric ever raises: a comparison that cannot be carried
out comes back as `NaN`, so a sweep over many design points always runs to completion.

See [docs_performance.md](docs_performance.md) for the full reference and three worked examples.
