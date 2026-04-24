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