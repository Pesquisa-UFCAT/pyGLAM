Evaluate distributional agreement
==================================

:class:`~pyglam.performance.Performance` is the evaluation class for comparing a
reference sample with a predicted or generated sample. It complements
:class:`~pyglam.glam.GlamFKML`: the distribution class fits and evaluates a model,
while the evaluation class measures how closely two datasets agree.

The inputs are observations, not vectors of lambda parameters. In particular,
``r2_pdf`` compares estimated density curves. It is different from the regression
R² used to evaluate a neural network that predicts ``lam2`` or the other lambdas.

Read the metrics
----------------

.. list-table:: Returned scores
   :header-rows: 1
   :widths: 25 35 40

   * - Key
     - Meaning
     - Interpretation
   * - ``kl_divergence``
     - KL divergence from the reference KDE to the predicted KDE.
     - Smaller is better; depends on the grid, density floor and KDE smoothing.
   * - ``ks_statistic``
     - Largest difference between the two empirical CDFs.
     - Smaller is better; uses the samples directly.
   * - ``ks_pvalue``
     - Two-sample KS test p-value.
     - A large value does not establish equality. Sample size and fitting on the
       same data affect its interpretation.
   * - ``wasserstein``
     - First Wasserstein distance between the samples.
     - Smaller is better; expressed in the same units as the observations.
   * - ``r2_pdf``
     - R² of the two KDE density curves on their shared grid.
     - Closer to 1 indicates closer curves. The current implementation clips
       this score to [-1, 1], so very poor fits may all report -1.
   * - ``rel_err_p5``, ``rel_err_p50``, ``rel_err_p95``
     - Signed errors at the 5th, 50th and 95th percentiles.
     - Normally (predicted - reference) / abs(reference). When abs(reference)
       is below ``percentile_tol``, use the signed difference in data units.

The report also includes ``x_grid``, ``pdf_true`` and ``pdf_pred`` for inspecting
or plotting the estimated densities. By default the grid spans the reference
sample. ``grid_range="union"`` spans both samples. The KDEs are normalized on
that grid, so density scores depend on the selected range.

Example: matching and shifted samples
-------------------------------------

The :doc:`quickstart` covers fitting a model before evaluating it. Here a known
uniform FKML model isolates the evaluation step. The shifted sample has the
right shape but the wrong location.

.. doctest::

   >>> import numpy as np
   >>> from pyglam import GlamFKML, Performance
   >>> model = GlamFKML(10, 2, 1, 1)
   >>> reference = model.rvs(2000, quantile_trim=0.0, random_state=7)
   >>> predicted = model.rvs(2000, quantile_trim=0.0, random_state=42)
   >>> evaluator = Performance(n_grid=200)
   >>> matching = evaluator.performance(reference, predicted, grid_range="union")
   >>> shifted = evaluator.performance(reference, predicted + 3, grid_range="union")
   >>> matching['wasserstein'] < shifted['wasserstein']
   True
   >>> matching['ks_statistic'] < shifted['ks_statistic']
   True
   >>> matching['x_grid'].shape
   (200,)

For repeated sampling experiments, reuse a generator to obtain different
samples from a reproducible sequence. Report variability as well as average
scores; a single random sample does not characterize stability.

.. doctest::

   >>> rng = np.random.default_rng(123)
   >>> distances = []
   >>> for _ in range(3):
   ...     sample = model.rvs(2000, quantile_trim=0.0, random_state=rng)
   ...     result = evaluator.performance(reference, sample, grid_range="union")
   ...     distances.append(result['wasserstein'])
   >>> bool(np.isfinite(np.mean(distances)) and np.isfinite(np.std(distances)))
   True

Example: percentile errors near zero
------------------------------------

A relative error can become large when its reference percentile is near zero.
``percentile_tol`` controls when to use a signed difference instead. This changes
the units of the affected entries even though their keys retain ``rel_err_``.

.. doctest::

   >>> reference = np.array([-1.5, -0.5, 0.5, 1.5])
   >>> shifted = reference + 0.1
   >>> report = evaluator.performance(reference, shifted, percentile_tol=0.2)
   >>> round(report['rel_err_p50'], 3)
   0.1

Here the reference median is zero, so the median error is 0.1 in data units.
Choose the threshold in the context of your measurement scale.

Example: unavailable metrics
----------------------------

An empty sample cannot be compared; a constant sample has no usable KDE for the
density-based metrics. Available sample-based scores are retained.

.. doctest::

   >>> missing = evaluator.performance([], [1, 2, 3])
   >>> bool(np.isnan(missing['wasserstein']))
   True
   >>> constant = evaluator.performance([1, 1, 1], [2, 2, 2])
   >>> bool(np.isnan(constant['r2_pdf']))
   True
   >>> constant['wasserstein']
   1.0

This handling applies to unavailable metrics, not invalid configuration.
For example, an unknown ``grid_range`` raises ``ValueError``.

Class and method reference
--------------------------

.. autoclass:: pyglam.performance.Performance
   :members: performance
