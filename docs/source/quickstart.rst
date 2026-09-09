Fit, sample and evaluate
========================

Install the released package with ``pip install pyglam``. For development from
a checkout, run ``uv sync --locked`` in the repository root, then launch Python
with ``uv run --locked python``. These examples describe the source in this
checkout; they include changes that may not yet be in the published release.

Fit a distribution
------------------

Here the observations come from a normal distribution with mean 10 and standard
deviation 2. A fixed seed makes this example reproducible.

.. doctest::

   >>> import numpy as np
   >>> from pyglam import GlamFKML, Performance
   >>> data = np.random.default_rng(7).normal(10, 2, 2000)
   >>> sol = GlamFKML().fit_lambdas(data)
   >>> bool(sol.success)
   True

The returned ``sol.x`` contains ``lam1``, ``lam2``, ``lam3`` and ``lam4``.
Fitting does not populate the original instance. Before using the four-moment
fit, check that the parameters are finite and belong to its valid domain:

.. doctest::

   >>> valid = (np.all(np.isfinite(sol.x)) and sol.x[1] > 0
   ...          and np.all(sol.x[2:] > -0.25))
   >>> if not sol.success or not valid:
   ...     raise ValueError("The moment fit did not produce valid FKML parameters.")
   >>> model = GlamFKML(*sol.x)

Sample and evaluate the model
-----------------------------

Use ``random_state`` for reproducibility. This example explicitly disables
trimming to compare a sample from the full fitted distribution with the reference
data. Omitting ``quantile_trim`` retains the default cut of 0.001 per tail.

.. doctest::

   >>> generated = model.rvs(size=2000, quantile_trim=0.0, random_state=42)
   >>> generated.shape
   (2000,)
   >>> probabilities = np.array([0.05, 0.5, 0.95])
   >>> percentiles = model.ppf(probabilities)
   >>> bool(np.allclose(model.cdf(percentiles), probabilities, atol=1e-10))
   True
   >>> density = model.pdf(percentiles)
   >>> bool(np.all(density >= 0))
   True

To obtain different samples within one reproducible experiment, create a single
``np.random.default_rng(seed)`` and pass that generator to successive calls.
Passing the same integer seed to every call repeats the same sample.

Measure agreement
-----------------

:class:`~pyglam.performance.Performance` compares two samples, without fitting
another model. Inspect several metrics because each measures a different aspect
of agreement.

.. doctest::

   >>> report = Performance(n_grid=200).performance(data, generated, grid_range="union")
   >>> for key in ("kl_divergence", "ks_statistic", "wasserstein", "r2_pdf"):
   ...     print(f"{key}: {report[key]:.4f}")
   kl_divergence: 0.0018
   ks_statistic: 0.0175
   wasserstein: 0.0488
   r2_pdf: 0.9949

These are the outputs for this example and environment, not acceptance
thresholds. Sampling and numerical-library versions can affect the scores.
See :doc:`performance` for their interpretation, comparison examples and
handling of unavailable metrics. Agreement with the same data used to fit the
model is descriptive; use held-out data when assessing predictive generalization.

