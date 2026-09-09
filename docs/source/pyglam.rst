The FKML distribution family
============================

:class:`~pyglam.glam.GlamFKML` groups the methods for the Generalized Lambda
Distribution in the Freimer--Kollia--Mudholkar--Lin (FKML) parameterization.
This is a family of continuous distributions: changing its four parameters can
produce symmetric or asymmetric shapes, bounded distributions and heavy tails.
It can approximate many familiar distributions, although an approximation need
not reproduce a target distribution exactly.

The family is defined by its quantile function, which maps a probability
:math:`u` to a value :math:`Q(u)`:

.. math::

   Q(u) = \lambda_1 + \frac{\phi(u,\lambda_3)-\phi(1-u,\lambda_4)}{\lambda_2},
   \qquad 0 < u < 1,

.. math::

   \phi(u,\lambda) =
   \begin{cases}
      (u^\lambda - 1)/\lambda, & \lambda \ne 0,\\
      \log(u), & \lambda = 0.
   \end{cases}

.. list-table:: Parameters
   :header-rows: 1
   :widths: 15 25 60

   * - Parameter
     - Role
     - Interpretation
   * - ``lam1``
     - Location
     - Shifts the distribution; it is not generally the mean.
   * - ``lam2``
     - Inverse scale
     - Must be positive. With shape fixed, increasing it reduces the spread.
   * - ``lam3``
     - Left shape
     - Controls the left side, including whether its support is bounded.
   * - ``lam4``
     - Right shape
     - Controls the right side, including whether its support is bounded.

``lam2`` is not generally the inverse standard deviation: variance also depends
on ``lam3`` and ``lam4``. For example, ``GlamFKML(0, 2, 1, 1)`` is uniform on
[-0.5, 0.5], and ``GlamFKML(0, 1, 0, 0)`` is the standard logistic distribution.

Support and tails
-----------------

The *support* is the range of values the distribution can take. Its lower bound
is :math:`\lambda_1-1/(\lambda_2\lambda_3)` when :math:`\lambda_3>0`, and
:math:`-\infty` otherwise. Its upper bound is
:math:`\lambda_1+1/(\lambda_2\lambda_4)` when :math:`\lambda_4>0`, and
:math:`+\infty` otherwise.

Below the support, the CDF is 0; above it, the CDF is 1. The PDF is zero outside
the support. At finite endpoints, the PDF uses its one-sided density limit.
Both methods preserve the shape of arrays, return scalars for scalar input, and
propagate NaN values in their original positions.

``rvs()`` draws uniform probabilities and transforms them with the quantile
function. Its default ``quantile_trim=0.001`` excludes 0.1% of each tail. Set
``quantile_trim=1e-6`` for a smaller cut, or ``0.0`` to disable it. Trimming changes
the sampled distribution and may affect moments substantially for heavy tails.
The quantile function retains a numerical endpoint guard of ``1e-12``.

The ``qp_min`` guard belongs to ``pdf()``. If a quantile derivative is at or below
that threshold, the corresponding density is NaN. Use ``qp_min=0`` to allow any
positive derivative. This argument is not part of ``rvs()``.

Choosing a method
-----------------

.. list-table:: Public methods
   :header-rows: 1

   * - Method
     - Purpose
   * - :meth:`~pyglam.glam.GlamFKML.moments`
     - Summarize data with mean, variance, skewness and Pearson kurtosis.
   * - :meth:`~pyglam.glam.GlamFKML.fit_lambdas`
     - Estimate all four parameters by matching sample moments.
   * - :meth:`~pyglam.glam.GlamFKML.rvs`
     - Draw random samples, optionally with a reproducible seed.
   * - :meth:`~pyglam.glam.GlamFKML.ppf`
     - Convert probabilities to values, such as percentiles.
   * - :meth:`~pyglam.glam.GlamFKML.cdf`
     - Evaluate the probability of a value being at or below a threshold.
   * - :meth:`~pyglam.glam.GlamFKML.pdf`
     - Evaluate density; a density value is not a probability at a point.

The current fit matches four moments, which require
:math:`\lambda_3,\lambda_4>-1/4`. The optimizer does not yet enforce these
bounds. Verify the fitted parameters and evaluate distributional agreement
before using the model; optimizer success alone is insufficient.

Class and method reference
--------------------------

.. autoclass:: pyglam.glam.GlamFKML
   :members: moments, fit_lambdas, rvs, ppf, cdf, pdf

