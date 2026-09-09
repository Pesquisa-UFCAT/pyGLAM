# PyGLAM

**PyGLAM** is a high-performance Python framework for emulating probability distributions using **Generalized Lambda Distributions (GLD)**.

## Installation

```bash
pip install pyglam
```

For local development and tests with `uv`, see [README-ENV.md](README-ENV.md).

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

rvs_glam = emulator.rvs(size=1000, random_state=42) # Reproducible random variates
pdf_glam = emulator.pdf(x_vals)             # Probability Density Function
cdf_glam = emulator.cdf(x_vals)             # Cumulative Distribution Function
ppf_glam = emulator.ppf(np.linspace(0.01, 0.99, 100)) # Percent Point Function
```

`rvs()` draws independent uniform probabilities and transforms them through the FKML
quantile function. Without `random_state`, each call draws a fresh sample. Passing the
same integer seed repeats a sample; reuse a NumPy generator for a reproducible sequence
of different samples:

```python
rng = np.random.default_rng(42)
sample_a = emulator.rvs(size=1000, random_state=rng)
sample_b = emulator.rvs(size=1000, random_state=rng)
```

Samples are returned in draw order, with exactly the requested size (or tuple shape).
The default `quantile_trim=0.001` samples the central 99.8% of the distribution,
excluding 0.1% from each tail to limit extreme draws. This is a truncated distribution:
its moments can differ from those of the full FKML, especially with heavy tails.
Set `quantile_trim=0.0` to sample the full distribution, subject to the quantile
function's numerical probability guard of `1e-12`.
Optional `lam1`–`lam4` arguments override instance parameters for that call.
`np.random.seed()` does not control the default generator; use `random_state` instead.

For the deterministic quantile grid returned by older versions, use
`emulator.ppf(np.linspace(0.001, 0.999, 1000))`.

`cdf()` and `pdf()` preserve the input shape and return a scalar for a scalar input.
Outside the FKML support, the CDF returns 0 or 1 and the PDF returns 0. NaN inputs
remain NaN; at finite support endpoints the PDF uses its one-sided limit.
The PDF's `qp_min` guard marks low quantile derivatives as NaN without dropping
positions; use `qp_min=0` to allow any positive derivative. For CDF/PDF, `tol`
sets the absolute inversion tolerance in probability space; smaller values may
be needed to resolve very small tail probabilities and densities.

## Measuring how good the emulation is

`Performance` scores a predicted sample against a reference sample. Hand it the raw data and the
sample you generated with `rvs()`, and it reports the agreement between the two distributions:

```python
from pyglam import Performance

report = Performance().performance(x, rvs_glam)

print(report['kl_divergence'])  # Kullback-Leibler divergence
print(report['ks_statistic'])   # Two-sample Kolmogorov-Smirnov statistic
print(report['ks_pvalue'])      # KS p-value; a large value does not prove equality
print(report['wasserstein'])    # Distance in data units
print(report['r2_pdf'])         # R2 between the estimated density curves
```

The report also carries `rel_err_p5`, `rel_err_p50`, `rel_err_p95` and the arrays `x_grid`,
`pdf_true` and `pdf_pred` for plotting. Unavailable metrics are reported as `NaN`;
invalid options such as an unknown `grid_range` raise `ValueError`. `r2_pdf` is
different from regression R2 for a neural network predicting lambda parameters.

See [docs_performance.md](docs_performance.md) for the full reference and three worked examples.

The Sphinx user guide includes an [introduction to FKML](docs/source/pyglam.rst),
a [fit-and-sample walkthrough](docs/source/quickstart.rst), and
[performance examples and API](docs/source/performance.rst). See
[README-ENV.md](README-ENV.md#documentação-sphinx) to build the HTML and execute its examples.
