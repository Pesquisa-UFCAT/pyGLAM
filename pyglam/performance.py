import numpy as np
from scipy.integrate import simpson
from scipy.stats import gaussian_kde, ks_2samp, wasserstein_distance

__all__ = ['Performance']

# Metric keys that are set to NaN when a comparison cannot be carried out.
_METRIC_KEYS = (
                    'kl_divergence',
                    'ks_statistic',
                    'ks_pvalue',
                    'wasserstein',
                    'r2_pdf',
                    'rel_err_p5',
                    'rel_err_p50',
                    'rel_err_p95',
               )


def _prepare_data(data: list | np.ndarray) -> np.ndarray:
    """Prepare data for analysis: convert to numpy array, ensure 1D, and remove NaN/inf values.

    Mirrors ``GlamFKML._prepare_data`` so both classes clean their inputs the same way.

    :param data: dataset

    :return: cleaned 1D numpy array
    """

    x = np.asarray(data, dtype=float)
    if x.ndim != 1:
        x = x.ravel()
    x = x[np.isfinite(x)]

    return x


class Performance:
    r"""Statistical performance metrics between a reference sample and a predicted sample.

    Compares two datasets that are already drawn — typically the raw data on one side and a sample
    produced by :meth:`pyglam.glam.GlamFKML.rvs` on the other — and scores how close the predicted distribution
    is to the reference one. The class never fits anything: it only knows about the two samples it
    is handed, so it works just as well for any pair of samples, whatever produced them.

    Both densities are estimated with a Gaussian KDE on a shared grid, then normalised to unit area
    before the density-based metrics are integrated.

    Unavailable metrics are reported as NaN. Empty samples prevent sample-based
    comparisons; constant or single-point samples prevent KDE-based metrics,
    while some sample-based metrics may still be available. Invalid options
    such as an unknown ``grid_range`` raise ValueError.

    ``r2_pdf`` compares two estimated density curves. It is not the regression
    R² of predicted lambda parameters from a neural network.

    :param n_grid: number of grid points used to numerically integrate the KL divergence and R²
    :param eps: density floor used to keep the logarithm and the normalisation well defined

    Example:
        >>> import numpy as np
        >>> from pyglam import Performance
        >>> rng = np.random.default_rng(42)
        >>> a = rng.normal(10, 2, 2000)
        >>> b = rng.normal(10, 2, 2000)
        >>> report = Performance().performance(a, b)
        >>> report['wasserstein'] >= 0
        True
    """

    def __init__(self, n_grid: int = 400, eps: float = 1e-12) -> None:
        """Initialize the performance evaluator.

        :param n_grid: number of grid points used to numerically integrate the KL divergence and R²
        :param eps: density floor used to keep the logarithm and the normalisation well defined
        """

        self.n_grid = int(n_grid)
        self.eps    = float(eps)

    def performance(self, true: list | np.ndarray, pred: list | np.ndarray, grid_range: str = "true", percentile_tol: float = 1e-5) -> dict:
        r"""Score a predicted sample against a reference sample.

        KL divergence is integrated on a shared grid; :math:`R^2` compares density
        values on that grid. By default the grid spans the reference sample's
        range. Use ``grid_range="union"`` to include the ranges of both samples.
        Densities are floored at ``eps`` and normalized on the selected grid, so
        the grid and KDE smoothing affect the density-based scores.

        .. warning::
            The percentile errors are *relative*, so they blow up when the reference percentile sits
            near zero — two samples from the same ``N(0, 1)`` can report ``rel_err_p50`` above 2.0
            simply because the median is ~0.003 and a tiny absolute difference is divided by it.
            Raise ``percentile_tol`` above the scale of the percentile you care about to switch
            those to signed differences in data units, or read the Wasserstein distance instead, which has no such
            pathology.

        :param true: reference dataset (the "real" sample)
        :param pred: predicted dataset (for example, the output of :meth:`pyglam.glam.GlamFKML.rvs`)
        :param grid_range: ``"true"`` to span the reference range, ``"union"`` to span both samples
        :param percentile_tol: reference percentiles below this magnitude are scored with the signed difference in data units instead of the relative one

        :return: Dictionary with the KL divergence, KS statistic and p-value, Wasserstein distance, R² between the two densities, relative errors at P5/P50/P95, the shared grid, and both normalised densities

        Examples:
            Compare two independently drawn samples from a known FKML model.

            >>> from pyglam import GlamFKML, Performance
            >>> model = GlamFKML(10, 2, 0.2, 0.3)
            >>> reference = model.rvs(2000, quantile_trim=0.0, random_state=7)
            >>> predicted = model.rvs(2000, quantile_trim=0.0, random_state=42)
            >>> report = Performance(n_grid=200).performance(
            ...     reference, predicted, grid_range="union")
            >>> sorted(report)
            ['kl_divergence', 'ks_pvalue', 'ks_statistic', 'pdf_pred', 'pdf_true', 'r2_pdf', 'rel_err_p5', 'rel_err_p50', 'rel_err_p95', 'wasserstein', 'x_grid']
            >>> report['x_grid'].shape
            (200,)

            Missing data yields unavailable scores without discarding the report.

            >>> import numpy as np
            >>> missing = Performance().performance([], predicted)
            >>> bool(np.isnan(missing['wasserstein']))
            True
        """

        if grid_range not in ("true", "union"):
            raise ValueError(f"grid_range must be 'true' or 'union', got {grid_range!r}")

        x_true = _prepare_data(true)
        x_pred = _prepare_data(pred)

        x_grid   = self._grid(x_true, x_pred, grid_range)
        pdf_true = self._density(x_true, x_grid)
        pdf_pred = self._density(x_pred, x_grid)

        result = {key: np.nan for key in _METRIC_KEYS}
        result['x_grid']   = x_grid
        result['pdf_true'] = pdf_true if pdf_true is not None else np.full(self.n_grid, np.nan)
        result['pdf_pred'] = pdf_pred if pdf_pred is not None else np.full(self.n_grid, np.nan)

        # =========================
        # 1. Density-based metrics: KDE of the reference vs. KDE of the prediction
        # =========================
        if pdf_true is not None and pdf_pred is not None:
            p = pdf_true
            q = pdf_pred

            result['kl_divergence'] = max(float(simpson(p * np.log(p / q), x=x_grid)), 0.0)

            ss_res = float(np.sum((p - q)**2))
            ss_tot = float(np.sum((p - p.mean())**2))
            if ss_tot > 1e-14:
                result['r2_pdf'] = float(np.clip(1.0 - ss_res / ss_tot, -1.0, 1.0))

        # =========================
        # 2. Sample-based diagnostics: the two raw samples, no density estimation involved
        # =========================
        if x_true.size > 0 and x_pred.size > 0:
            ks = ks_2samp(x_true, x_pred)
            result['ks_statistic'] = float(ks.statistic)
            result['ks_pvalue']    = float(ks.pvalue)
            result['wasserstein']  = float(wasserstein_distance(x_true, x_pred))

            p5_true, p50_true, p95_true = np.percentile(x_true, [5, 50, 95])
            p5_pred, p50_pred, p95_pred = np.percentile(x_pred, [5, 50, 95])

            result['rel_err_p5']  = self._relative_error(p5_pred, p5_true, percentile_tol)
            result['rel_err_p50'] = self._relative_error(p50_pred, p50_true, percentile_tol)
            result['rel_err_p95'] = self._relative_error(p95_pred, p95_true, percentile_tol)

        return result

    def _grid(self, x_true: np.ndarray, x_pred: np.ndarray, grid_range: str) -> np.ndarray:
        """Build the shared integration grid.

        :param x_true: cleaned reference sample
        :param x_pred: cleaned predicted sample
        :param grid_range: ``"true"`` to span the reference range, ``"union"`` to span both samples

        :return: grid of ``n_grid`` points, or a NaN array when no sample has any data
        """

        if grid_range == "union":
            pool = [a for a in (x_true, x_pred) if a.size]
        else:
            pool = [a for a in (x_true, x_pred) if a.size][:1]

        if not pool:
            return np.full(self.n_grid, np.nan)

        stacked = np.concatenate(pool)
        low     = float(stacked.min())
        high    = float(stacked.max())

        # A constant sample has zero span; widen it slightly so the grid is still usable.
        if low == high:
            span = max(abs(low), 1.0) * 1e-6
            low, high = low - span, high + span

        return np.linspace(low, high, self.n_grid)

    def _density(self, x: np.ndarray, x_grid: np.ndarray) -> np.ndarray | None:
        """Estimate a normalised density on the shared grid via Gaussian KDE.

        :param x: cleaned sample
        :param x_grid: shared integration grid

        :return: density normalised to unit area, or ``None`` when the KDE is not defined
        """

        if x.size < 2 or np.std(x) <= 0.0 or not np.all(np.isfinite(x_grid)):
            return None

        try:
            density = gaussian_kde(x)(x_grid)
        except Exception:
            # A singular covariance matrix (near-constant sample) is a failed comparison, not a crash.
            return None

        density = np.nan_to_num(density, nan=0.0, posinf=0.0, neginf=0.0)
        density = np.clip(density, self.eps, None)
        area    = float(simpson(density, x=x_grid))

        if not np.isfinite(area) or area <= 0.0:
            return None

        return density / area

    @staticmethod
    def _relative_error(model: float, real: float, tol: float = 1e-5) -> float:
        """Relative error, falling back to the absolute error when the reference is close to zero.

        :param model: predicted value
        :param real: reference value
        :param tol: threshold below which the reference is treated as zero

        :return: relative error, or absolute error when ``abs(real) < tol``
        """

        return float(model - real) if abs(real) < tol else float((model - real) / abs(real))
