# `Performance` — statistical performance metrics

The `Performance` class scores how close a **predicted** sample is to a **reference** sample. It
never fits anything: you hand it two datasets that are already drawn, and it reports how well they
agree. In the usual pyGLAM workflow the reference is the raw data and the prediction is the output
of `GlamFKML.rvs()`, but the class does not care where the samples came from — any two 1D samples
work.

Both densities are estimated with a Gaussian KDE on a shared grid and normalised to unit area before
the density-based metrics are integrated.

```python
from pyglam import Performance
```

---

## API

```python
Performance(n_grid: int = 400, eps: float = 1e-12)
```

| Parameter | Meaning |
|---|---|
| `n_grid` | Number of grid points used to integrate the KL divergence and R² |
| `eps` | Density floor that keeps the logarithm and the normalisation well defined |

```python
.performance(true, pred, grid_range="true", percentile_tol=1e-5) -> dict
```

| Parameter | Meaning |
|---|---|
| `true` | Reference dataset (the "real" sample) |
| `pred` | Predicted dataset (for example, the output of `GlamFKML.rvs()`) |
| `grid_range` | `"true"` spans the reference range; `"union"` spans both samples |
| `percentile_tol` | Reference percentiles below this magnitude are scored with the absolute error instead of the relative one |

### Returned dictionary

| Key | Meaning | Perfect agreement |
|---|---|---|
| `kl_divergence` | Kullback–Leibler divergence between the two densities | `0` |
| `ks_statistic` | Two-sample Kolmogorov–Smirnov statistic | `0` |
| `ks_pvalue` | p-value of the same test — `> 0.05` means agreement is not rejected | `1` |
| `wasserstein` | Wasserstein (earth mover's) distance, in the units of the data | `0` |
| `r2_pdf` | R² between the two densities on the grid | `1` |
| `rel_err_p5` / `p50` / `p95` | Relative error at the 5th / 50th / 95th percentile | `0` |
| `x_grid` | The shared integration grid | — |
| `pdf_true` / `pdf_pred` | Both densities on that grid, normalised to unit area | — |

**Nothing ever raises.** A comparison that cannot be carried out — empty sample, constant sample,
fewer than two points — comes back as `NaN`. That way a sweep over a whole dataset always runs to
completion and the failures show up as gaps in the resulting maps rather than as a crash.

---

## Example 1 — good agreement

Two independent samples from the same distribution. This is your calibration case: it shows what the
metrics look like when nothing is wrong.

```python
import numpy as np
from pyglam import Performance

rng = np.random.default_rng(42)
x_true = rng.normal(0, 1, 20000)
x_pred = rng.normal(0, 1, 20000)

report = Performance().performance(x_true, x_pred)
```

```
  kl_divergence: 0.000479
   ks_statistic: 0.004750
      ks_pvalue: 0.976952
    wasserstein: 0.006068
         r2_pdf: 0.999428
     rel_err_p5: -0.010104
    rel_err_p50: 2.015672
    rel_err_p95: -0.005291
```

KL essentially zero, R² essentially one, and the KS test does not reject (`p = 0.98`). The two
samples are indistinguishable, as they should be.

**Note the `rel_err_p50` of 2.02.** That is not a disagreement — it is the relative-error pitfall
described under [Caveats](#caveats). The median of `N(0, 1)` is about `0.003`, so a negligible
absolute difference divided by it produces a huge ratio. Read `wasserstein` (`0.006`, in data units)
instead.

---

## Example 2 — the full pyGLAM workflow

The real use case: fit a GLD to your data, generate from it, and measure how good the emulation is.

```python
import numpy as np
from pyglam import GlamFKML, Performance

rng = np.random.default_rng(7)
data = rng.normal(10, 2, 20000)

# 1. Fit the lambdas
g = GlamFKML()
sol = g.fit_lambdas(data)
print(sol.x)        # [9.98736  0.736143  0.13592  0.136153]
print(sol.success)  # True

# 2. Generate from the fitted model
emulator = GlamFKML(*sol.x)
generated = emulator.rvs(size=20000)

# 3. Score the emulation
report = Performance().performance(data, generated)
```

```
  kl_divergence: 0.005898
   ks_statistic: 0.004050
      ks_pvalue: 0.996453
    wasserstein: 0.013482
         r2_pdf: 0.999618
     rel_err_p5: 0.003337
    rel_err_p50: 0.001122
    rel_err_p95: -0.001430
```

The GLD reproduces the normal closely: KS does not reject, R² is `0.9996`, and all three percentiles
are within 0.4%. Here the percentile errors are trustworthy because the data is centred at 10, far
from zero.

> `sol.success` only tells you the optimiser converged — it is **not** a measure of fit quality. This
> report is. Always score the fit rather than trusting `success`.

---

## Example 3 — poor agreement, and a failed prediction

### 3a. Two genuinely different distributions

```python
import numpy as np
from pyglam import Performance

rng = np.random.default_rng(0)
normal_data = rng.normal(0, 1, 20000)
expon_data  = rng.exponential(1.0, 20000)

report = Performance().performance(normal_data, expon_data)
```

```
  kl_divergence: 6.698965
   ks_statistic: 0.502000
      ks_pvalue: 0.000000
    wasserstein: 0.998236
         r2_pdf: -0.301664
     rel_err_p5: 1.030614
    rel_err_p50: 117.496464
    rel_err_p95: 0.835018
```

Everything points the same way: KL of 6.7, KS rejects outright (`p = 0`), and a **negative** R²,
meaning the predicted density is a worse explanation of the reference density than a flat line would
be. A negative `r2_pdf` is always a red flag.

### 3b. A prediction that failed

`GlamFKML` returns an empty array when the fitted parameters are not a valid GLD (for example
`λ₂ ≤ 0`, where the quantile function stops increasing). The class absorbs that:

```python
report = Performance().performance(normal_data, np.array([]))
```

```
  kl_divergence: nan
   ks_statistic: nan
      ks_pvalue: nan
    wasserstein: nan
         r2_pdf: nan
     rel_err_p5: nan
    rel_err_p50: nan
    rel_err_p95: nan
```

No exception. `x_grid` and `pdf_true` are still populated (the reference side is fine), and
`pdf_pred` is all `NaN`. A loop over thousands of design points therefore completes, and you can
locate the failures afterwards with `np.isnan(...)`.

---

## Choosing `grid_range`

```python
Performance().performance(normal_data, expon_data, grid_range="true")   # default
Performance().performance(normal_data, expon_data, grid_range="union")
```

```
   true: grid=[-4.023, +3.946]  KL=6.6990  R2=-0.3017
  union: grid=[-4.023, +9.705]  KL=6.7179  R2=+0.0599
```

- **`"true"`** (default) spans the reference data's range. If the prediction does not cover that
  range, its density is clipped to the `eps` floor there and the KL divergence gets large. That is
  the intended reading: *the prediction puts almost no probability where the real data actually
  lives.*
- **`"union"`** spans both samples, so it also accounts for mass the prediction invents outside the
  reference range. Note how R² rises from `-0.30` to `+0.06` here — over the wider grid both
  densities are near zero across most of the span, which flatters the fit. `"true"` is the stricter
  and usually more honest choice.

---

## Plotting the comparison

The report carries everything needed to draw the two densities:

```python
import matplotlib.pyplot as plt

report = Performance().performance(data, generated)

plt.plot(report['x_grid'], report['pdf_true'], label='Reference', lw=2)
plt.plot(report['x_grid'], report['pdf_pred'], label='Predicted', lw=2, ls='--')
plt.fill_between(report['x_grid'], report['pdf_true'], report['pdf_pred'], alpha=0.2)
plt.title(f"KL = {report['kl_divergence']:.4f}   $R^2$ = {report['r2_pdf']:.4f}")
plt.legend()
plt.show()
```

---

## How to read the numbers

| Metric | Good | Suspicious |
|---|---|---|
| `kl_divergence` | `< 0.01` | `> 0.1` |
| `ks_pvalue` | `> 0.05` (not rejected) | `< 0.01` |
| `r2_pdf` | `> 0.99` | `< 0.9`, and **negative is a red flag** |
| `wasserstein` | small relative to the data's spread | comparable to the standard deviation |

The KS p-value is sample-size dependent: with very large `n` it rejects on differences too small to
matter in practice. With `n` in the hundreds of thousands, lean on `wasserstein` and `r2_pdf`
instead.

---

## Caveats

**Percentile errors near zero.** `rel_err_*` divides by the reference percentile, so it explodes when
that percentile sits near zero — Example 1 shows `rel_err_p50 = 2.02` for two samples from the *same*
distribution. Raise `percentile_tol` above the scale of the percentile you care about to switch to
absolute errors, or use `wasserstein`, which has no such pathology. This behaviour is inherited from
the original benchmark script, where the quantity of interest was far from zero.

**`GlamFKML.rvs()` is currently deterministic.** It builds an evenly spaced quantile grid rather than
drawing random variates, so two consecutive calls return identical arrays. That makes `ks_statistic`
in Example 2 a comparison between a random sample and a deterministic lattice, which is optimistic —
the real sampling noise is missing. Interpret the emulation scores as a best case until `rvs()`
performs true inverse-transform sampling.

**KDE smoothing.** Both densities are smoothed with a Gaussian KDE using Scott's rule. Sharp features
— hard bounds, discontinuities, strongly multimodal data — are blurred, which flatters `kl_divergence`
and `r2_pdf`. The sample-based metrics (`ks_statistic`, `wasserstein`) do not smooth anything and are
the safer read for such data.
