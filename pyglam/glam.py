import scipy as sc
import numpy as np
 

from scipy.special import beta
from scipy.optimize import brentq

# Deterministic grid of shape pairs (lambda3, lambda4) for the multi-start. It
# covers the qualitative regimes of the GLD: bounded supports (positive shapes),
# near-exponential tails (shapes near zero), heavy tails (negative shapes),
# asymmetry (unequal shapes) and the mixed-sign RS regions 5 and 6. Shapes at or
# below -1/4 are left out because the kurtosis does not exist there.
#
# The order matters: n_starts takes the first entries, so the first fifteen
# visit every regime once before any regime is visited twice. A grid that
# listed one regime first would leave a small multi-start blind to the others;
# for RS, whose regions are separated by the sign of lambda2, the solver cannot
# cross from one region to another, so a start on each side is needed.
_SHAPE_GRID = (
    # one pass over every regime
    (0.135, 0.135),    # near normal
    (-0.10, -0.10),    # heavy tails
    (0.05, 0.05),      # near exponential tails
    (1.0, 1.0),        # uniform
    (0.1, 0.5),        # skewed, bounded
    (0.5, 0.1),
    (-0.05, -0.15),    # skewed, heavy tails
    (-0.15, -0.05),
    (-0.249, 8.0),     # RS region 5
    (8.0, -0.249),     # RS region 6
    (0.01, 0.01),
    (0.5, 0.5),
    (-0.18, -0.18),
    (0.2, 1.0),
    (1.0, 0.2),
    # second pass
    (0.2, 0.2),
    (1.8, 1.8),
    (2.0, 2.0),
    (-0.05, -0.05),
    (-0.15, -0.15),
    (-0.22, -0.22),
    (-0.10, -0.20),
    (-0.20, -0.10),
    (-0.12, -0.18),
    (-0.18, -0.12),
    (-0.240, 9.0),
    (-0.220, 14.0),
    (9.0, -0.240),
    (14.0, -0.220),
)


def _shape_candidates(n_candidates: int, seed=None) -> np.ndarray:
    """Return ``n_candidates`` shape pairs (lam3, lam4) for the multi-start.

    The fixed grid comes first, so a small number of starts is reproducible
    without a seed. Beyond the grid, shapes are drawn log-uniformly in
    ``lam + 1/4``, which samples the boundary of the moment domain and the
    large-shape region on comparable terms.
    """

    grid = np.array(_SHAPE_GRID, dtype=float)
    if n_candidates <= len(grid):
        return grid[:n_candidates]

    rng = np.random.default_rng(seed)
    extra = rng.uniform(np.log(0.05), np.log(5.0), size=(n_candidates - len(grid), 2))

    return np.vstack([grid, -0.25 + np.exp(extra)])

def _in_moment_domain(x, family="FKML") -> bool:
    """Verifica se os parâmetros pertencem ao domínio dos quatro momentos."""

    x = np.asarray(x, dtype=float)

    if x.size != 4 or not np.all(np.isfinite(x)):
        return False

    family_name = (
        family
        if isinstance(family, str)
        else getattr(family, "__name__", "")
    )

    l1, l2, l3, l4 = x

    if l2 == 0.0:
        return False

    # ==========================================================
    # FKML
    # ==========================================================
    if family_name in ("FKML", "GlamFKML"):
        return bool(
            l2 > 0.0
            and l3 > -0.25
            and l4 > -0.25
        )

    # ==========================================================
    # RS - Ramberg-Schmeiser
    # ==========================================================
    if family_name in ("RS", "GlamRS"):

        # Domínio dos quatro momentos
        if l3 <= -0.25 or l4 <= -0.25:
            return False

        # Região RS
        if l2 > 0.0:
            return bool(
                l3 >= 0.0
                and l4 >= 0.0
                and (l3 > 0.0 or l4 > 0.0)
            )

        if l2 < 0.0:

            # Região 4
            if (
                l3 <= 0.0
                and l4 <= 0.0
                and (l3 < 0.0 or l4 < 0.0)
            ):
                return True

            # Região 5
            if (
                -1.0 < l3 < 0.0
                and l4 > 1.0
            ):
                lhs_log = (
                    (1.0 - l3) * np.log(1.0 - l3)
                    + (l4 - 1.0) * np.log(l4 - 1.0)
                    - (l4 - l3) * np.log(l4 - l3)
                )

                rhs_log = np.log(-l3 / l4)

                if lhs_log < rhs_log:
                    return True

            # Região 6
            if (
                l3 > 1.0
                and -1.0 < l4 < 0.0
            ):
                lhs_log = (
                    (1.0 - l4) * np.log(1.0 - l4)
                    + (l3 - 1.0) * np.log(l3 - 1.0)
                    - (l3 - l4) * np.log(l3 - l4)
                )

                rhs_log = np.log(-l4 / l3)

                if lhs_log < rhs_log:
                    return True

        return False

    return False


def _solve_from_starts(residuals, starts, method: str):
    """Run the optimizer once per starting point and collect the solutions.

    A start that makes the optimizer fail is skipped: it must not invalidate
    the remaining ones. The failure is re-raised only when every start fails.
    """

    used, solutions = [], []
    failure = None
    for start in starts:
        try:
            if method == "root":
                sol = sc.optimize.root(residuals, start, method="hybr")
            else:
                # least_squares é mais robusto que root fora do mínimo
                sol = sc.optimize.least_squares(residuals, start, method="trf", max_nfev=5000, ftol=1e-5, xtol=1e-5, gtol=1e-5)
        except Exception as error:
            failure = error if failure is None else failure
            continue
        used.append(start)
        solutions.append(sol)

    if not solutions:
        if failure is not None:
            raise failure
        raise ValueError("at least one starting point is required.")

    return used, solutions


def _quantile_distance(family, sample, params) -> float:
    """Mean absolute gap between fitted and empirical quantiles.

    This is the L1 (Wasserstein) distance between the sample and the fitted
    distribution, evaluated on at most 2000 plotting positions. Parameters
    that cannot be turned into a usable model score as infinite.
    """

    try:
        model = family(*params)
        size = min(sample.size, 2000)
        probabilities = (np.arange(size) + 0.5) / size
        fitted = np.asarray(model.ppf(probabilities), dtype=float)
        if not np.all(np.isfinite(fitted)):
            return np.inf
        return float(np.mean(np.abs(fitted - np.quantile(sample, probabilities))))
    except Exception:
        return np.inf


def _has_monotone_quantile(family, params, n_points: int = 512) -> bool:
    """Report whether ``params`` describe a distribution at all.

    Four-moment matching is not injective, and some of its roots have a
    quantile function that decreases somewhere on (0, 1): the parameters
    reproduce the four moments to machine precision and still do not define a
    distribution. Optimizer termination cannot see this, so the quantile
    function is sampled and required to be finite and non-decreasing.
    """

    try:
        model = family(*params)
        probabilities = np.linspace(1e-6, 1.0 - 1e-6, n_points)
        quantiles = np.asarray(model.ppf(probabilities), dtype=float)
    except Exception:
        return False

    if not np.all(np.isfinite(quantiles)):
        return False

    # a tolerance relative to the spread absorbs round-off in the quantile
    # evaluation, not a real decrease
    tolerance = 1e-9 * max(float(np.max(np.abs(quantiles))), 1.0)

    return bool(np.all(np.diff(quantiles) >= -tolerance))


def _best_solution(solutions, residuals, starts, family, sample):
    """Pick the best solution and record what the other starts reached.
    Solutions that are distributions come first: a lower residual is worthless
    if the quantile function runs backwards, and a lower residual outside the
    moment domain describes parameters whose moments do not exist. Cost then selects the
    best, but four-moment matching is not injective — different parameters can
    reproduce the same four moments — so every cost numerically tied with the
    best is a candidate, and the tie goes to the smallest quantile distance to
    the data. Without that step a difference such as 1e-13 against 1e-27, which
    is optimizer noise, would decide the fit.

    The returned result carries ``starts``, ``candidates``, ``n_starts``,
    ``costs``, ``in_domain``, ``monotone``, ``distances`` and ``best_start``
    alongside the SciPy fields, so another selection rule is one line away.
    """

    costs = np.array([0.5 * float(np.sum(np.asarray(residuals(sol.x), dtype=float) ** 2)) for sol in solutions])
    in_domain = np.array([ _in_moment_domain(sol.x, family=family) for sol in solutions])
    #in_domain = np.array([_in_moment_domain(sol.x) for sol in solutions])
    monotone = np.array([_has_monotone_quantile(family, sol.x) for sol in solutions])
    distances = np.array([_quantile_distance(family, sample, sol.x) for sol in solutions])

    # Candidates that are not distributions are never preferred, whatever their
    # residual. The fallbacks keep a fit available when no candidate clears the
    # stricter screens, so a hard case still returns instead of raising.
    for mask in (in_domain & monotone, monotone, in_domain):
        if mask.any():
            pool = np.flatnonzero(mask)
            break
    else:
        pool = np.arange(costs.size)
    best_cost = float(costs[pool].min())
    tied = pool[costs[pool] <= best_cost + 1e-8 + 1e-2 * best_cost]
    best = int(tied[np.argmin(distances[tied])])

    sol = solutions[best]
    sol.starts = np.array(starts, dtype=float)
    sol.candidates = np.array([solution.x for solution in solutions], dtype=float)
    sol.n_starts = len(solutions)
    sol.costs = costs
    sol.in_domain = in_domain
    sol.monotone = monotone
    sol.distances = distances
    sol.best_start = best

    return sol


class GlamFKML:
    r"""Methods for the Generalized Lambda Distribution in its FKML parameterization.

    The Freimer--Kollia--Mudholkar--Lin (FKML) family describes continuous
    distributions through their quantile function. Changing four parameters
    controls location, scale, asymmetry and tail shape, allowing one family to
    approximate different distribution shapes.

    Use :meth:`fit_lambdas` to estimate parameters from data, then construct an
    instance with the fitted parameters to generate samples or evaluate the
    quantile, cumulative distribution and density functions. Use
    :class:`pyglam.performance.Performance` to assess agreement with reference data.

    :param lam1: location parameter; not generally the mean
    :param lam2: positive inverse-scale parameter; not generally the inverse
        standard deviation because the shape parameters also affect variance
    :param lam3: left-shape parameter
    :param lam4: right-shape parameter

    Parameters may be omitted when constructing an instance for fitting.
    Sampling and distribution evaluation require all four parameters.

    Examples:
        The parameters below describe a uniform distribution on [-0.5, 0.5].

        >>> from pyglam import GlamFKML
        >>> model = GlamFKML(lam1=0, lam2=2, lam3=1, lam4=1)
        >>> model.cdf(0)
        0.5
        >>> model.rvs(size=5, random_state=42).shape
        (5,)
    """
    def __init__(self, lam1: float | None = None, lam2: float | None = None, lam3: float | None = None, lam4: float | None = None) -> None:
        """Initialize the GLAM-FKML distribution."""
        self.lam1 = lam1
        self.lam2 = lam2
        self.lam3 = lam3
        self.lam4 = lam4

    def _validated_lambdas(self, **overrides):
        params = []
        for name in ("lam1", "lam2", "lam3", "lam4"):
            value = overrides.get(name)
            if value is None:
                value = getattr(self, name)
            if value is None or not np.isscalar(value):
                raise ValueError(f"{name} must be a finite scalar.")
            value = float(value)
            if not np.isfinite(value):
                raise ValueError(f"{name} must be a finite scalar.")
            params.append(value)
        if params[1] <= 0:
            raise ValueError("lam2 must be positive for the FKML distribution.")
        return tuple(params)

    def moments(self, data: list | np.ndarray):
        """Compute the first four moments of a dataset.
        
        :param data: dataset

        Variance uses divisor n, and kurtosis is Pearson kurtosis (a normal
        distribution has kurtosis 3), not excess kurtosis. Non-finite entries
        are removed and multidimensional input is flattened. Use nonempty data
        with positive variance to obtain all four finite moments.

        :return: mean, variance, skewness and kurtosis, in that order

        Examples:
            >>> import numpy as np
            >>> from pyglam import GlamFKML
            >>> (np.round(GlamFKML().moments([1, 2, 3]), 3) + 0.0).tolist()
            [2.0, 0.667, 0.0, 1.5]
        """

        x     = self._prepare_data(data)
        mean  = np.mean(x)
        var   = np.mean((x - mean)**2)
        sigma = np.sqrt(var)
        skew  = np.mean(((x - mean) / sigma)**3)
        kurt  = np.mean(((x - mean) / sigma)**4)

        return mean, var, skew, kurt    
    
    def _prepare_data(self, data: list | np.ndarray) -> np.ndarray:
        """Prepare data for analysis: convert to numpy array, ensure 1D, and remove NaN/inf values.

        :param data: dataset

        :return: cleaned 1D numpy array
        """

        x = np.asarray(data, dtype=float)
        if x.ndim != 1:
            x = x.ravel()
        x = x[np.isfinite(x)]

        return x    

    def _v1(self, lambda3: float, lambda4: float) -> float:     

        term1 = 1.0 / (lambda3 * (lambda3 + 1.0))
        term2 = 1.0 / (lambda4 * (lambda4 + 1.0))

        return term1 - term2

    def _v2(self, lambda3: float, lambda4: float) -> float:

        term1 = 1.0 / (lambda3**2 * (2.0 * lambda3 + 1.0))
        term2 = (2.0 / (lambda3 * lambda4)) * (sc.special.gamma(lambda3 + 1.0) * sc.special.gamma(lambda4 + 1.0)) / sc.special.gamma(lambda3 + lambda4 + 2.0)
        term3 = 1.0 / (lambda4**2 * (2.0 * lambda4 + 1.0))

        return term1 - term2 + term3
    
    def _v3(self, lambda3: float, lambda4: float) -> float:

        l3    = float(lambda3)
        l4    = float(lambda4)
        term1 = 1.0 / (l3**3 * (3.0*l3 + 1.0))
        term2 = (3.0 / (l3**2 * l4)) * (sc.special.gamma(2.0*l3 + 1.0) * sc.special.gamma(l4 + 1.0)) / sc.special.gamma(2.0*l3 + l4 + 2.0)
        term3 = (3.0 / (l3 * l4**2)) * (sc.special.gamma(l3 + 1.0) * sc.special.gamma(2.0*l4 + 1.0)) / sc.special.gamma(l3 + 2.0*l4 + 2.0)
        term4 = 1.0 / (l4**3 * (3.0*l4 + 1.0))

        return term1 - term2 + term3 - term4
    
    def _v4(self, lambda3: float, lambda4: float) -> float:

        l3    = float(lambda3)
        l4    = float(lambda4)
        term1 = 1.0 / (l3**4 * (4.0*l3 + 1.0))
        term2 = (4.0 / (l3**3 * l4)) * (sc.special.gamma(3.0*l3 + 1.0) * sc.special.gamma(l4 + 1.0)) / sc.special.gamma(3.0*l3 + l4 + 2.0)
        term3 = (6.0 / (l3**2 * l4**2)) * (sc.special.gamma(2.0*l3 + 1.0) * sc.special.gamma(2.0*l4 + 1.0)) / sc.special.gamma(2.0*l3 + 2.0*l4 + 2.0)
        term4 = (4.0 / (l3 * l4**3)) * (sc.special.gamma(l3 + 1.0) * sc.special.gamma(3.0*l4 + 1.0)) / sc.special.gamma(l3 + 3.0*l4 + 2.0)
        term5 = 1.0 / (l4**4 * (4.0*l4 + 1.0))

        return term1 - term2 + term3 - term4 + term5

    def _theoretical_moments(self, lambda1: float, lambda2: float, lambda3: float, lambda4: float):
        
        v1 = self._v1(lambda3, lambda4)
        v2 = self._v2(lambda3, lambda4)
        v3 = self._v3(lambda3, lambda4)
        v4 = self._v4(lambda3, lambda4)

        # mean
        mean = lambda1 - (1.0 / lambda2) * (1.0 / (lambda3 + 1.0) - 1.0 / (lambda4 + 1.0))

        # variance
        a2 = (v2 - v1**2)
        var = a2 / (lambda2**2)

        if a2 <= 0 or var <= 0:
            return mean, np.nan, np.nan, np.nan

        # skewness
        skew = (v3 - 3.0*v1*v2 + 2.0*(v1**3)) / (a2 ** 1.5)

        # kurtosis
        kurt = (v4 - 4.0*v1*v3 + 6.0*(v1**2)*v2 - 3.0*(v1**4)) / (a2 ** 2)

        return mean, var, skew, kurt

    def _start_from_shapes(self, lambda3: float, lambda4: float, mean_hat: float, var_hat: float):
        """Complete a shape pair into a full starting point.

        With the shapes fixed, matching the sample mean and variance is a
        closed-form problem, so the candidate already reproduces two of the
        four targets and the optimizer only has to work on skewness and
        kurtosis. Returns ``None`` when the shapes have no positive variance.
        """

        try:
            v1 = self._v1(lambda3, lambda4)
            v2 = self._v2(lambda3, lambda4)
            a2 = v2 - v1**2
            if not np.isfinite(a2) or a2 <= 0:
                return None
            lambda2 = np.sqrt(a2 / var_hat)
            lambda1 = mean_hat + (1.0 / lambda2) * (1.0 / (lambda3 + 1.0) - 1.0 / (lambda4 + 1.0))
        except (ZeroDivisionError, FloatingPointError, ValueError):
            return None

        start = np.array([lambda1, lambda2, lambda3, lambda4], dtype=float)

        return start if np.all(np.isfinite(start)) and lambda2 > 0 else None

    def fit_lambdas(self, data, x0=None, method="least_squares", n_starts=1, seed=None):
        """Fit the GLAM-FKML distribution parameters to the data using moment matching.

        Match the sample mean, variance, skewness and Pearson kurtosis. The
        result contains four parameters in ``sol.x``; fitting does not update
        the instance. Create ``GlamFKML(*sol.x)`` to use the fitted model.

        :param data: observations with positive variance after removing non-finite values
        :param x0: optional initial guess [lam1, lam2, lam3, lam4]; the default
            is [sample mean, inverse sample standard deviation, 0.5, 0.5]
        :param method: ``"least_squares"`` (default) or ``"root"``
        :param n_starts: number of starting points; 1 (default) fits from ``x0``
            alone, larger values add starts spread over the shape plane
        :param seed: seed or NumPy Generator used only when ``n_starts`` exceeds
            the fixed grid of shapes and extra starts must be drawn
        :return: SciPy OptimizeResult with parameters, residuals and convergence status

        ``sol.success`` describes optimizer termination, not distributional
        agreement. The current optimizer does not enforce the FKML moment
        domain: check finite parameters, lam2 > 0 and lam3, lam4 > -1/4 before
        interpreting a four-moment fit. Evaluate the fitted distribution with
        :class:`pyglam.performance.Performance` as well.

        Moment matching has several local minima, and a single start converges
        to whichever one it begins next to. With ``n_starts`` above 1 the fit
        restarts from points spread over the shape plane, each completed so
        that it already matches the sample mean and variance, and returns the
        best result: solutions inside the moment domain come first, then the
        lowest cost, and costs that are numerically tied are separated by the
        quantile distance to the data, because different parameters can share
        the same four moments. The attempts are reported in ``sol.costs``,
        ``sol.in_domain`` and ``sol.distances``, their parameters in
        ``sol.candidates``, their starting points in ``sol.starts``, and the
        winner in ``sol.best_start``.

        A better moment match is not always a better distributional fit: on
        skewed data an exact four-moment solution can sit far from the sample,
        while a start that ends with a larger cost stays close to it. That is a
        limit of moment matching, not of the search. Compare ``sol.distances``
        before accepting a multi-start result, and select by another rule with
        ``sol.candidates[int(np.argmin(sol.distances))]`` when adherence
        matters more than the moments.

        Examples:
            >>> import numpy as np
            >>> from pyglam import GlamFKML
            >>> data = np.random.default_rng(7).normal(10, 2, 2000)
            >>> sol = GlamFKML().fit_lambdas(data)
            >>> bool(sol.success)
            True
            >>> model = GlamFKML(*sol.x)
            >>> model.rvs(size=100, random_state=42).shape
            (100,)

            Ten starts spread over the shape plane, with every attempt kept:

            >>> multi = GlamFKML().fit_lambdas(data, n_starts=10)
            >>> multi.n_starts
            10
            >>> multi.candidates.shape
            (10, 4)
        """

        mean_hat, var_hat, skew_hat, kurt_hat = self.moments(data)
        target = np.array([mean_hat, var_hat, skew_hat, kurt_hat], dtype=float)

        if x0 is None:
            sd = np.sqrt(var_hat) if var_hat > 0 else 1.0
            x0 = np.array([mean_hat, 1.0/max(sd, 1e-8), 0.5, 0.5], dtype=float)
        else:
            x0 = np.array(x0, dtype=float)

        n_starts = int(n_starts)
        if n_starts < 1:
            raise ValueError("n_starts must be at least 1.")

        def residuals(x):
            lambda1, lambda2, lambda3, lambda4 = x

            # domínios mínimos para evitar divisões por zero e singularidades triviais
            if lambda2 == 0 or lambda3 in (0.0,) or lambda4 in (0.0,):
                return np.ones(4) * 1e6

            mean, var, skew, kurt = self._theoretical_moments(lambda1, lambda2, lambda3, lambda4)
            r = np.array([mean, var, skew, kurt], dtype=float) - target

            # se algum virou nan (região inválida), penaliza
            if not np.all(np.isfinite(r)):
                return np.ones(4) * 1e6

            # (opcional) escala: evita a variância dominar tudo
            scale = np.array([max(abs(mean_hat), 1.0),
                              max(abs(var_hat), 1.0),
                              1.0,
                              max(abs(kurt_hat), 1.0)], dtype=float)
            return r / scale

        starts = [x0]
        for lambda3, lambda4 in _shape_candidates(n_starts - 1, seed):
            start = self._start_from_shapes(lambda3, lambda4, mean_hat, var_hat)
            if start is not None:
                starts.append(start)

        used, solutions = _solve_from_starts(residuals, starts, method)

        return _best_solution(solutions, residuals, used, type(self), np.sort(self._prepare_data(data)))
        
     # Função auxiliar φ(u,λ) usada na parametrização FKML da GLD.
    # Para λ ≠ 0: φ(u,λ) = (u^λ − 1)/λ
    # Para λ → 0: φ(u,λ) → log(u) (limite analítico).
    # expm1 evita cancelamento numérico quando λ ≈ 0.      expm1(x)=e^x−1
    def _phi(self, u, lam):
        log_u = np.log(u)
        if lam == 0.0:
            return log_u
        return np.expm1(lam * log_u) / lam

    def _gld_fkml_quantile(self, u, l1, l2, l3, l4):
        u = np.clip(u, 1e-12, 1-1e-12)
        return l1 + ( self._phi(u, l3) - self._phi(1-u, l4) ) / l2

    def _gld_fkml_qprime(self, u, l2, l3, l4):
        u = np.asarray(u, dtype=float)
        # Endpoint limits are needed by pdf(); infinite derivatives give zero density.
        with np.errstate(divide="ignore", over="ignore"):
            return (u**(l3 - 1.0) + (1.0 - u)**(l4 - 1.0)) / l2
    
    def rvs(
        self,
        size: int | tuple[int, ...] = 1000,
        quantile_trim: float = 0.0,
        lam1: float | None = None,
        lam2: float | None = None,
        lam3: float | None = None,
        lam4: float | None = None,
        *,
        random_state: int | np.random.Generator | np.random.RandomState | None = None,
    ) -> np.ndarray:
        """Draw random variates by inverse-transform sampling.

        :param size: number of samples, or a tuple giving the output shape
        :param quantile_trim: optional probability removed from each tail;
            must satisfy 0 <= quantile_trim < 0.5. The default 0.0 keeps the
            tails, apart from the quantile function's numerical guard of 1e-12;
            0.001, for example, samples the central 99.8% of the distribution.
        :param lam1: optional location override for this call
        :param lam2: optional positive inverse-scale override for this call
        :param lam3: optional left-shape override for this call
        :param lam4: optional right-shape override for this call
        :param random_state: integer seed, NumPy Generator or RandomState, or None.
            An integer repeats the sample; a generator advances its state;
            None creates a fresh generator independently of np.random.seed().

        Samples retain their draw order and requested shape. Invalid parameters
        raise ValueError; non-finite samples raise FloatingPointError instead of
        being silently discarded. Parameter overrides do not modify the instance.

        Examples:
            >>> import numpy as np
            >>> from pyglam import GlamFKML
            >>> model = GlamFKML(0, 2, 1, 1)
            >>> first = model.rvs(size=100, random_state=42)
            >>> np.array_equal(first, model.rvs(size=100, random_state=42))
            True
            >>> rng = np.random.default_rng(42)
            >>> a = model.rvs(size=100, random_state=rng)
            >>> b = model.rvs(size=100, random_state=rng)
            >>> np.array_equal(a, b)
            False
            >>> model.rvs(size=(2, 3), quantile_trim=1e-6, random_state=42).shape
            (2, 3)
        """
        params = self._validated_lambdas(lam1=lam1, lam2=lam2, lam3=lam3, lam4=lam4)

        a = float(quantile_trim)
        if not np.isfinite(a) or not 0.0 <= a < 0.5:
            raise ValueError("quantile_trim must satisfy 0 <= quantile_trim < 0.5.")
        rng = (random_state if isinstance(random_state, np.random.RandomState)
               else np.random.default_rng(random_state))
        u = rng.uniform(a, 1.0 - a, size=size)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            x = self._gld_fkml_quantile(u, *params)
        if not np.all(np.isfinite(x)):
            raise FloatingPointError("FKML sampling produced non-finite values for these parameters.")
        return x
    
    def ppf(self, q: float | np.ndarray, tol: float = 1e-12) -> np.ndarray:
        """Percent point function (inverse CDF) for the GLAM-FKML distribution.
        
        :param q: quantiles to evaluate
        :param tol: tolerance to avoid extreme tails (values close to 0 or 1)
        
        :return: values corresponding to the given quantiles

        Probabilities are clipped to [tol, 1 - tol], and the internal quantile
        function also applies a 1e-12 endpoint guard. ``ppf(0)`` and ``ppf(1)``
        therefore evaluate guarded tail quantiles, not exact support endpoints.

        Examples:
            >>> import numpy as np
            >>> from pyglam import GlamFKML
            >>> model = GlamFKML(0, 2, 1, 1)
            >>> np.round(model.ppf([0.1, 0.5, 0.9]), 3).tolist()
            [-0.4, 0.0, 0.4]
            >>> grid = model.ppf(np.linspace(0.001, 0.999, 100))
            >>> grid.shape
            (100,)
        """

        q = np.asarray(q, dtype=float)
        q_clipped = np.clip(q, tol, 1 - tol)
        
        return self._gld_fkml_quantile(q_clipped, self.lam1, self.lam2, self.lam3, self.lam4)
    
    
    @staticmethod
    def _support_bounds(params):
        l1, l2, l3, l4 = params
        lower = l1 - (1.0 / l3) / l2 if l3 > 0 else -np.inf
        upper = l1 + (1.0 / l4) / l2 if l4 > 0 else np.inf

        return lower, upper

    def _cdf_array(self, x, tol, params):
        if not np.isfinite(tol) or not 0 < tol < 0.5:
            raise ValueError("tol must satisfy 0 < tol < 0.5.")
        lower, upper = self._support_bounds( params)
        result = np.full(x.shape, np.nan, dtype=float)
        result[x <= lower] = 0.0
        result[x >= upper] = 1.0
        interior = np.flatnonzero(np.isfinite(x) & (x > lower) & (x < upper))
        l1, l2, l3, l4 = params

        def quantile(u):
            if u == 0.0:
                return lower
            if u == 1.0:
                return upper
            # Inversion must use the full quantile, not the clipped ppf used
            # for sampling, or finite tail observations can fall outside its bracket.
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                return l1 + (self._phi(u, l3) - self._phi(1.0 - u, l4)) / l2

        for index in interior:
            xi = x.flat[index]
            result.flat[index] = sc.optimize.brentq(
                lambda u: quantile(u) - xi, 0.0, 1.0, xtol=tol, maxiter=200
            )
        return result

    def cdf(self, x: float | np.ndarray, tol: float = 1e-12) -> float | np.ndarray:
        """Evaluate the CDF, preserving the input shape (scalar in, scalar out).

        Values at or below the lower support bound return 0; values at or above
        the upper bound return 1. NaN inputs remain NaN.

        :param x: values to evaluate
        :param tol: absolute tolerance for inversion in probability space,
            with 0 < tol < 0.5. Tail probabilities smaller than tol may round to 0 or 1.

        Examples:
            >>> from pyglam import GlamFKML
            >>> model = GlamFKML(0, 2, 1, 1)
            >>> model.cdf(0)
            0.5
            >>> model.cdf([-1, 0, 1]).tolist()
            [0.0, 0.5, 1.0]
        """
        params = self._validated_lambdas()
        x = np.asarray(x, dtype=float)
        result = self._cdf_array(x, tol, params)
        return result.item() if x.ndim == 0 else result

    def pdf(self, x: float | np.ndarray, tol: float = 1e-12, qp_min: float = 1e-8) -> float | np.ndarray:
        """Evaluate the PDF, preserving the input shape (scalar in, scalar out).

        Outside the support and at +/-infinity the density is zero. At finite
        support endpoints, use the one-sided density limit. NaN inputs remain NaN.

        :param x: values to evaluate
        :param tol: absolute tolerance for CDF inversion in probability space;
            reduce it when resolving very small tail densities
        :param qp_min: nonnegative minimum quantile derivative; points at or below
            this threshold return NaN, preserving their positions. Set to 0 to
            allow any positive derivative.

        Examples:
            >>> import numpy as np
            >>> from pyglam import GlamFKML
            >>> model = GlamFKML(0, 2, 1, 1)
            >>> model.pdf(0)
            1.0
            >>> model.pdf([-1, 0, 1]).tolist()
            [0.0, 1.0, 0.0]
            >>> narrow = GlamFKML(0, 1e12, 1, 1)
            >>> bool(np.isnan(narrow.pdf(0)))
            True
            >>> narrow.pdf(0, qp_min=0)
            500000000000.0
        """
        if not np.isfinite(qp_min) or qp_min < 0:
            raise ValueError("qp_min must be finite and nonnegative.")
        params = self._validated_lambdas()
        x = np.asarray(x, dtype=float)
        lower, upper = self._support_bounds(params)
        u = self._cdf_array(x, tol, params)
        result = np.full(x.shape, np.nan, dtype=float)
        result[(x < lower) | (x > upper) | np.isinf(x)] = 0.0
        inside = np.isfinite(x) & (x >= lower) & (x <= upper)
        qp = self._gld_fkml_qprime(u[inside], *params[1:])
        density = np.full(qp.shape, np.nan, dtype=float)
        np.divide(1.0, qp, out=density, where=qp > qp_min)
        result[inside] = density
        return result.item() if x.ndim == 0 else result


 

class GlamRS:
    """
    Generalized Lambda Distribution (GLD)
    - Ramberg-Schmeiser (RS)

    Função quantil:

        Q(u) = lambda1
             + [u^lambda3 - (1-u)^lambda4] / lambda2

    Regiões utilizadas:

    Região 1:
        lambda2 > 0
        lambda3 > 0
        lambda4 > 0

    Região 2:
        lambda2 < 0
        lambda3 < 0
        lambda4 < 0

    A parametrização é ajustada pelos quatro primeiros momentos:
        média
        variância
        assimetria
        curtose de Pearson
    """

    def __init__(
        self,
        lam1=None,
        lam2=None,
        lam3=None,
        lam4=None,
        name="GlamRS",
        random_state=None,
    ):
        self.lam1 = lam1
        self.lam2 = lam2
        self.lam3 = lam3
        self.lam4 = lam4

        self.name = name
        self.random_state = random_state
    def _rs_region(self, params=None):

        if params is None:
          params = (
            self.lam1,
            self.lam2,
            self.lam3,
            self.lam4,
        )

        l1, l2, l3, l4 = map(float, params)

        if not np.all(np.isfinite([l1, l2, l3, l4])):
           return None

        if l2 == 0.0:
            return None

    # ==========================================================
    # Região 1
    # l2 < 0, l3 < -1, l4 > 1
    # ==========================================================
        if (
        l2 < 0.0
        and l3 < -1.0
        and l4 > 1.0):
          return 1

    # ==========================================================
    # Região 2
    # l2 < 0, l3 > 1, l4 < -1
    # ==========================================================
        if (
        l2 < 0.0
        and l3 > 1.0
        and l4 < -1.0
    ):
          return 2

    # ==========================================================
    # Região 3
    # l2 > 0, l3 >= 0, l4 >= 0
    # ==========================================================
        if (
        l2 > 0.0
        and l3 >= 0.0
        and l4 >= 0.0
        and (l3 > 0.0 or l4 > 0.0)
    ):
          return 3

    # ==========================================================
    # Região 4
    # l2 < 0, l3 <= 0, l4 <= 0
    # ==========================================================
        if (
        l2 < 0.0
        and l3 <= 0.0
        and l4 <= 0.0
        and (l3 < 0.0 or l4 < 0.0)
    ):
          return 4

    # ==========================================================
    # Região 5
    # l2 < 0
    # -1 < l3 < 0
    # l4 > 1
    #
    # condição adicional
    # ==========================================================
        if (
           l2 < 0.0
           and -1.0 < l3 < 0.0
           and l4 > 1.0 ):

           lhs_log = (
            (1.0 - l3) * np.log(1.0 - l3)
            + (l4 - 1.0) * np.log(l4 - 1.0)
            - (l4 - l3) * np.log(l4 - l3))

           rhs_log = np.log(-l3 / l4)

           if lhs_log < rhs_log:
             return 5

    # ==========================================================
    # Região 6
    # l2 < 0
    # l3 > 1
    # -1 < l4 < 0
    #
    # condição adicional
    # ==========================================================
        if (
           l2 < 0.0
           and l3 > 1.0
           and -1.0 < l4 < 0.0):

            lhs_log = (
            (1.0 - l4) * np.log(1.0 - l4)
            + (l3 - 1.0) * np.log(l3 - 1.0)
            - (l3 - l4) * np.log(l3 - l4))

            rhs_log = np.log(-l4 / l3)

            if lhs_log < rhs_log:
              return 6

        return None
    
    # ==============================================================
    # VALIDAÇÃO DOS PARÂMETROS
    # ==============================================================

    def _validated_lambdas(self, **overrides):

        params = [
        overrides.get("lam1", self.lam1),
        overrides.get("lam2", self.lam2),
        overrides.get("lam3", self.lam3),
        overrides.get("lam4", self.lam4),]

        if any(p is None for p in params):
            raise ValueError(
            "Todos os parâmetros lambda devem estar definidos.")

        params = tuple(float(p) for p in params)

        if not np.all(np.isfinite(params)):
            raise ValueError(
            "Todos os parâmetros lambda devem ser finitos.")

        lam1, lam2, lam3, lam4 = params

        if lam2 == 0.0:
            raise ValueError("lambda2 cannot be zero.")

        region = self._rs_region(params)

        if region is None:
            raise ValueError(
            "Os parâmetros lambda não pertencem a uma região "
            "válida da GLD-RS.")

        return params
    # ==============================================================
    # MOMENTOS DA AMOSTRA
    # ==============================================================

    def moments(self, data):

        data = np.asarray(data, dtype=float)

        mean_hat = np.mean(data)
        var_hat = np.var(data, ddof=0)

        centered = data - mean_hat

        if var_hat <= 0:
            raise ValueError("Variance must be positive.")

        m3 = np.mean(centered**3)
        m4 = np.mean(centered**4)

        skew_hat = m3 / var_hat**1.5
        kurt_hat = m4 / var_hat**2

        return mean_hat, var_hat, skew_hat, kurt_hat

    def _prepare_data(self, data):

        data = np.asarray(data, dtype=float)

        if data.ndim != 1:
            data = data.ravel()

        if len(data) < 4:
            raise ValueError(
                "At least four observations are required."
            )

        if not np.all(np.isfinite(data)):
            raise ValueError(
                "Data contain non-finite values."
            )

        return data

    # ==============================================================
    # COEFICIENTES DOS MOMENTOS RS
    # ==============================================================

    def _A(self, lambda3, lambda4):

        return (
            1.0 / (1.0 + lambda3)
            - 1.0 / (1.0 + lambda4)
        )

    def _B(self, lambda3, lambda4):

        return (
            1.0 / (1.0 + 2.0 * lambda3)
            + 1.0 / (1.0 + 2.0 * lambda4)
            - 2.0 * beta(
                1.0 + lambda3,
                1.0 + lambda4
            )
        )

    def _C(self, lambda3, lambda4):

        return (
            1.0 / (1.0 + 3.0 * lambda3)
            - 1.0 / (1.0 + 3.0 * lambda4)
            - 3.0 * beta(
                1.0 + 2.0 * lambda3,
                1.0 + lambda4
            )
            + 3.0 * beta(
                1.0 + lambda3,
                1.0 + 2.0 * lambda4
            )
        )

    def _D(self, lambda3, lambda4):

        return (
            1.0 / (1.0 + 4.0 * lambda3)
            + 1.0 / (1.0 + 4.0 * lambda4)
            - 4.0 * beta(
                1.0 + 3.0 * lambda3,
                1.0 + lambda4
            )
            + 6.0 * beta(
                1.0 + 2.0 * lambda3,
                1.0 + 2.0 * lambda4
            )
            - 4.0 * beta(
                1.0 + lambda3,
                1.0 + 3.0 * lambda4
            )
        )

    # ==============================================================
    # MOMENTOS TEÓRICOS
    # ==============================================================

    def _theoretical_moments(
        self,
        lambda1,
        lambda2,
        lambda3,
        lambda4,
    ):

        # ----------------------------------------------------------
        # Quarto momento exige:
        #
        # lambda3 > -1/4
        # lambda4 > -1/4
        # ----------------------------------------------------------

        if lambda3 <= -0.25 or lambda4 <= -0.25:
            return np.nan, np.nan, np.nan, np.nan

        # ----------------------------------------------------------
        # Restrição original utilizada pela implementação
        # ----------------------------------------------------------

        #if lambda3 > 4.0 or lambda4 > 4.0:
        #    return np.nan, np.nan, np.nan, np.nan

        # ----------------------------------------------------------
        # lambda2 não pode ser zero
        # ----------------------------------------------------------

        if lambda2 == 0:
            return np.nan, np.nan, np.nan, np.nan

        try:

            a = self._A(lambda3, lambda4)
            b = self._B(lambda3, lambda4)
            c = self._C(lambda3, lambda4)
            d = self._D(lambda3, lambda4)

        except Exception:

            return np.nan, np.nan, np.nan, np.nan

        a2 = b - a**2

        if not np.isfinite(a2) or a2 <= 0:
            return np.nan, np.nan, np.nan, np.nan

        mean = lambda1 + a / lambda2

        var = a2 / lambda2**2

        if not np.isfinite(var) or var <= 0:
            return np.nan, np.nan, np.nan, np.nan

        try:
            skew = (
                c
                - 3.0 * a * b
            + 2.0 * a**3) / (np.sign(lambda2) * a2**1.5)
            #skew = (
            #    c
            #    - 3.0 * a * b
            #    + 2.0 * a**3
            #) / a2**1.5

            kurt = (
                d
                - 4.0 * a * c
                + 6.0 * a**2 * b
                - 3.0 * a**4
            ) / a2**2

        except Exception:

            return np.nan, np.nan, np.nan, np.nan

        values = np.array(
            [mean, var, skew, kurt],
            dtype=float
        )

        if not np.all(np.isfinite(values)):
            return np.nan, np.nan, np.nan, np.nan

        return mean, var, skew, kurt

    # ==============================================================
    # ESTIMATIVA INICIAL
    # ==============================================================

    def _start_from_shapes(
        self,
        mean_hat,
        var_hat,
        lambda3,
        lambda4,
    ):

        a = self._A(lambda3, lambda4)

        b = self._B(lambda3, lambda4)

        a2 = b - a**2

        if a2 <= 0:
            raise ValueError(
                "Invalid shape parameters: variance coefficient <= 0."
            )

        scale_mag = np.sqrt(a2 / var_hat)

        # ==========================================================
        # SINAL DE lambda2 SEGUNDO A REGIÃO RS
        # ==========================================================

        # Região 3
        if (
            lambda3 >= 0.0
            and lambda4 >= 0.0
            and (lambda3 > 0.0 or lambda4 > 0.0)
        ):
            lambda2 = scale_mag

        # Regiões 4, 5 e 6
        elif (
            (
                lambda3 <= 0.0
                and lambda4 <= 0.0
                and (lambda3 < 0.0 or lambda4 < 0.0)
            )
            or
            (
                -1.0 < lambda3 < 0.0
                and lambda4 > 1.0
            )
            or
            (
                lambda3 > 1.0
                and -1.0 < lambda4 < 0.0
            )
        ):
            lambda2 = -scale_mag

        else:
            raise ValueError(
                "Os parâmetros lambda3 e lambda4 "
                "não pertencem a uma região RS válida."
            )

        lambda1 = mean_hat - a / lambda2

        return lambda1, lambda2

     
    def fit_lambdas(
        self,
        data,
        method='least_squares',
        n_starts=10,
        seed=None):

        if method not in ("least_squares", "root"):
            raise ValueError(
                f"Método '{method}' não suportado pela GlamRS. "
                "Use 'least_squares' ou 'root'."
            )

        data = self._prepare_data(data)

        mean_hat, var_hat, skew_hat, kurt_hat = self.moments(data)

        target = np.array(
            [mean_hat, var_hat, skew_hat, kurt_hat],
            dtype=float
        )

        # ==========================================================
        # RESÍDUOS
        # ==========================================================

        def residuals(params):

            lambda1, lambda2, lambda3, lambda4 = params

            params = np.asarray(
                [lambda1, lambda2, lambda3, lambda4],
                dtype=float
            )

            # ------------------------------------------------------
            # Verificação básica
            # ------------------------------------------------------

            if not np.all(np.isfinite(params)):
                return np.ones(4) * 1e6

            if lambda2 == 0.0:
                return np.ones(4) * 1e6

            # ------------------------------------------------------
            # Domínio dos quatro momentos + regiões RS
            # ------------------------------------------------------

            if not _in_moment_domain(
                params,
                family=type(self)
            ):
                return np.ones(4) * 1e6

            # ------------------------------------------------------
            # Momentos teóricos
            # ------------------------------------------------------

            try:

                mean, var, skew, kurt = self._theoretical_moments(
                    lambda1,
                    lambda2,
                    lambda3,
                    lambda4,
                )

            except Exception:
                return np.ones(4) * 1e6

            theoretical = np.array(
                [mean, var, skew, kurt],
                dtype=float
            )

            if not np.all(np.isfinite(theoretical)):
                return np.ones(4) * 1e6

            # ------------------------------------------------------
            # Escalas dos resíduos
            # ------------------------------------------------------

            scales = np.array(
                [
                    max(abs(mean_hat), 1.0),
                    max(abs(var_hat), 1.0),
                    1.0,
                    max(abs(kurt_hat), 1.0),
                ],
                dtype=float
            )

            return (theoretical - target) / scales
        # ==========================================================
        # CANDIDATOS DE FORMA
        # ==========================================================

        # ATENÇÃO:
        # _shape_candidates é função GLOBAL.
        # Portanto NÃO usar self._shape_candidates(...)

        candidates = _shape_candidates(
            n_starts,
            seed=seed
        )

        starts = []

        for lambda3, lambda4 in candidates:

            try:

                lambda1, lambda2 = self._start_from_shapes(
                    mean_hat,
                    var_hat,
                    lambda3,
                    lambda4,
                )

                start = np.array(
                    [
                        lambda1,
                        lambda2,
                        lambda3,
                        lambda4,
                    ],
                    dtype=float
                )

                if np.all(np.isfinite(start)):
                    starts.append(start)

            except ValueError:
                # a shape pair outside every RS region (the random draws beyond
                # the grid often are) cannot start a fit; it is skipped silently
                continue

        if not starts:
            raise RuntimeError(
                "Nenhum ponto inicial válido foi encontrado."
            )

        # ==========================================================
        # MULTI-START
        # ==========================================================

        # _solve_from_starts também é função GLOBAL

        used, solutions = _solve_from_starts(
            residuals,
            starts,
            method
        )

        # ==========================================================
        # MELHOR SOLUÇÃO
        # ==========================================================

        # _best_solution também é função GLOBAL

        best = _best_solution(
            solutions,
            residuals,
            used,
            type(self),
            np.sort(data)
        )

        # ==========================================================
        # ATUALIZA O OBJETO
        # ==========================================================

        (
            self.lam1,
            self.lam2,
            self.lam3,
            self.lam4,
        ) = best.x

        return best
    # ==============================================================
    # QUANTIL RS
    # ==============================================================

    def _gld_rs_quantile(
        self,
        u,
        lambda1,
        lambda2,
        lambda3,
        lambda4,
    ):

        u = np.asarray(u, dtype=float)

        u = np.clip(
            u,
            1e-12,
            1.0 - 1e-12,
        )

        return (
            lambda1
            + (
                u**lambda3
                - (1.0 - u)**lambda4
            ) / lambda2
        )

    # ==============================================================
    # DERIVADA DO QUANTIL
    # ==============================================================

    def _gld_rs_qprime(
        self,
        u,
        lambda1,
        lambda2,
        lambda3,
        lambda4,
    ):

        u = np.asarray(u, dtype=float)

        u = np.clip(
            u,
            1e-12,
            1.0 - 1e-12,
        )

        return (
            lambda3 * u**(lambda3 - 1.0)
            + lambda4 * (1.0 - u)**(lambda4 - 1.0)
        ) / lambda2

    # ==============================================================
    # RVS
    # ==============================================================

    def rvs(
        self,
        size=None,
        random_state=None,
        quantile_trim=None,
    ):

        params = self._validated_lambdas()

        rng = np.random.default_rng(
            self.random_state
            if random_state is None
            else random_state
        )

        u = rng.random(size)

        if quantile_trim is not None:

            q = float(quantile_trim)

            # 0 keeps the tails, matching GlamFKML.rvs
            if not (0.0 <= q < 0.5):
                raise ValueError(
                    "quantile_trim must satisfy 0 <= quantile_trim < 0.5."
                )

            u = q + (1.0 - 2.0 * q) * u

        return self._gld_rs_quantile(
            u,
            *params,
        )

    # ==============================================================
    # PPF
    # ==============================================================

    def ppf(self, q):

        q = np.asarray(q, dtype=float)

        if np.any((q < 0) | (q > 1)):
            raise ValueError(
                "q must be in [0, 1]."
            )

        params = self._validated_lambdas()

        q_clipped = np.clip(
            q,
            1e-12,
            1.0 - 1e-12,
        )

        return self._gld_rs_quantile(
            q_clipped,
            *params,
        )

    # ==============================================================
    # SUPORTE
    # ==============================================================

    def _support_bounds(self):

        l1, l2, l3, l4 = self._validated_lambdas()

        region = self._rs_region((l1, l2, l3, l4))

        if region == 1:
            # (-inf, lambda1 + 1/lambda2]
            return (
                -np.inf,
                l1 + 1.0 / l2,
            )

        if region == 2:
            # [lambda1 - 1/lambda2, +inf)
            return (
                l1 - 1.0 / l2,
                np.inf,
            )

        if region == 3:

            if l3 == 0.0 and l4 > 0.0:
                # [lambda1, lambda1 + 1/lambda2]
                return (
                    l1,
                    l1 + 1.0 / l2,
                )

            if l3 > 0.0 and l4 == 0.0:
                # [lambda1 - 1/lambda2, lambda1]
                return (
                    l1 - 1.0 / l2,
                    l1,
                )

            if l3 > 0.0 and l4 > 0.0:
                # [lambda1 - 1/lambda2,
                #  lambda1 + 1/lambda2]
                return (
                    l1 - 1.0 / l2,
                    l1 + 1.0 / l2,
                )

        if region == 4:

            if l3 == 0.0 and l4 < 0.0:
                # [lambda1, +inf)
                return (
                    l1,
                    np.inf,
                )

            if l3 < 0.0 and l4 == 0.0:
                # (-inf, lambda1]
                return (
                    -np.inf,
                    l1,
                )

            if l3 < 0.0 and l4 < 0.0:
                # (-inf, +inf)
                return (
                    -np.inf,
                    np.inf,
                )

        if region == 5:
            # (-inf, lambda1 + 1/lambda2]
            return (
                -np.inf,
                l1 + 1.0 / l2,
            )

        if region == 6:
            # [lambda1 - 1/lambda2, +inf)
            return (
                l1 - 1.0 / l2,
                np.inf,
            )

        raise ValueError(
            "Região RS inválida para determinação do suporte."
        )

    # ==============================================================
    # CDF
    # ==============================================================

    def _cdf_array(self, x):

        x = np.asarray(x, dtype=float)

        l1, l2, l3, l4 = self._validated_lambdas()

        lower, upper = self._support_bounds()

        # NaN inputs stay NaN instead of leaving uninitialized memory behind
        result = np.full(x.shape, np.nan, dtype=float)

        result[x <= lower] = 0.0
        result[x >= upper] = 1.0

        # the root search is bracketed by the guarded quantile function, so
        # points of an unbounded support beyond that bracket round to 0 or 1
        q_low = self._gld_rs_quantile(1e-12, l1, l2, l3, l4)
        q_high = self._gld_rs_quantile(1.0 - 1e-12, l1, l2, l3, l4)
        result[(x > lower) & (x <= q_low)] = 0.0
        result[(x < upper) & (x >= q_high)] = 1.0

        mask = (x > max(lower, q_low)) & (x < min(upper, q_high))

        for index in np.ndindex(x.shape):

            if not mask[index]:
                continue

            xi = x[index]

            def func(u):

                return (
                    self._gld_rs_quantile(
                        u,
                        l1,
                        l2,
                        l3,
                        l4,
                    )
                    - xi
                )

            try:

                result[index] = brentq(
                    func,
                    1e-12,
                    1.0 - 1e-12,
                )

            except Exception:

                result[index] = np.nan

        return result

    def cdf(self, x):

        x = np.asarray(x, dtype=float)

        result = self._cdf_array(x)

        if result.ndim == 0:
            return float(result)

        return result

    # ==============================================================
    # PDF
    # ==============================================================

    def pdf(self, x):

        x = np.asarray(x, dtype=float)

        # _cdf_array keeps the array shape, so scalar input can be masked too
        F = self._cdf_array(x)

        l1, l2, l3, l4 = self._validated_lambdas()

        result = np.zeros_like(
            x,
            dtype=float
        )

        mask = (
            (F > 0.0)
            & (F < 1.0)
            & np.isfinite(F)
        )

        if np.any(mask):

            qp = self._gld_rs_qprime(
                F[mask],
                l1,
                l2,
                l3,
                l4,
            )

            valid = (
                np.isfinite(qp)
                & (qp != 0)
            )

            temp = np.full_like(
                qp,
                np.nan,
                dtype=float
            )

            temp[valid] = 1.0 / qp[valid]

            result[mask] = temp

        result[np.isnan(x)] = np.nan

        if result.ndim == 0:
            return float(result)

        return result