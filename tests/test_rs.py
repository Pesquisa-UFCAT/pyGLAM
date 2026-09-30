import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy import integrate, stats

from pyglam import GlamRS


def numerical_moments(model):
    """Mean, variance, skewness and Pearson kurtosis by integrating Q(u)^k over (0, 1)."""
    l1, l2, l3, l4 = model._validated_lambdas()

    def raw(k):
        q = lambda u: model._gld_rs_quantile(u, l1, l2, l3, l4) ** k
        return integrate.quad(q, 0.0, 1.0, limit=400, epsabs=1e-12, epsrel=1e-10)[0]

    m1, m2, m3, m4 = (raw(k) for k in range(1, 5))
    var = m2 - m1**2
    skew = (m3 - 3 * m1 * m2 + 2 * m1**3) / var**1.5
    kurt = (m4 - 4 * m1 * m3 + 6 * m1**2 * m2 - 3 * m1**4) / var**2
    return m1, var, skew, kurt


class RambergSchmeiserTests(unittest.TestCase):
    def test_uniform_special_case_and_scalar_results(self):
        # Q(u) = 0.5 + (u - (1 - u)) / 2 = u
        model = GlamRS(0.5, 2, 1, 1)
        values = np.array([-1.0, 0.0, 0.25, 0.5, 0.75, 1.0, 2.0])
        assert_allclose(model.cdf(values), stats.uniform.cdf(values), atol=1e-10)
        assert_allclose(model.pdf([0.25, 0.5, 0.75]), 1.0, atol=1e-10)
        for method in (model.cdf, model.pdf):
            self.assertIsInstance(method(0.5), float)
            self.assertTrue(np.isnan(method(np.nan)))
        assert_allclose(model.cdf([np.nan, 0.5]), [np.nan, 0.5], atol=1e-10)

    def test_sampling_reproducibility_and_full_tails(self):
        model = GlamRS(0.5, 2, 1, 1)
        sample = model.rvs(100_000, random_state=7)
        self.assertEqual(sample.shape, (100_000,))
        assert_allclose(sample, model.rvs(100_000, random_state=7))
        self.assertLess(stats.kstest(sample, "uniform").statistic, 0.01)
        self.assertLess(sample.min(), 0.001)
        self.assertGreater(sample.max(), 0.999)
        rng = np.random.default_rng(1)
        self.assertFalse(np.array_equal(model.rvs(10, random_state=rng), model.rvs(10, random_state=rng)))
        trimmed = model.rvs(10_000, quantile_trim=0.1, random_state=7)
        self.assertTrue(np.all((trimmed >= 0.1) & (trimmed <= 0.9)))

    def test_heavy_tailed_region_round_trip_and_density(self):
        model = GlamRS(0.0, -0.25, -0.14, -0.14)
        self.assertEqual(model._support_bounds(), (-np.inf, np.inf))
        u = np.linspace(0.001, 0.999, 41)
        assert_allclose(model.cdf(model.ppf(u)), u, atol=1e-9)
        x = model.ppf(np.linspace(0.01, 0.99, 50))
        self.assertTrue(np.all(np.diff(model.cdf(x)) > 0))
        self.assertTrue(np.all(model.pdf(x) > 0))
        # beyond the guarded quantile bracket the tails round to 0 and 1
        assert_allclose(model.cdf([-1e12, 1e12]), [0.0, 1.0])
        mass = integrate.quad(model.pdf, *model.ppf([1e-7, 1 - 1e-7]), limit=200)[0]
        assert_allclose(mass, 1.0, atol=1e-4)

    def test_theoretical_moments_match_integration_in_every_sign_of_lambda2(self):
        for params in [(0.3, 0.2, 0.13, 0.4), (0.0, -0.25, -0.05, -0.15), (1.0, -0.3, -0.2, -0.08)]:
            model = GlamRS(*params)
            assert_allclose(model._theoretical_moments(*params), numerical_moments(model),
                            rtol=1e-6, atol=1e-9, err_msg=str(params))

    def test_fit_returns_a_valid_distribution(self):
        rng = np.random.default_rng(3)
        for sample in (rng.normal(2.0, 0.5, 2000), rng.gumbel(size=2000), rng.uniform(size=2000)):
            sol = GlamRS().fit_lambdas(sample, n_starts=15, seed=1)
            model = GlamRS(*sol.x)
            self.assertIsNotNone(model._rs_region())
            self.assertTrue(sol.monotone[sol.best_start])
            assert_allclose(model._theoretical_moments(*sol.x)[:2],
                            GlamRS().moments(sample)[:2], rtol=1e-3, atol=1e-3)

    def test_parameters_outside_every_region_raise(self):
        with self.assertRaises(ValueError):
            GlamRS(0, 1, -0.1, 0.5).ppf(0.5)
        with self.assertRaises(ValueError):
            GlamRS(0, 0, 1, 1).ppf(0.5)


if __name__ == "__main__":
    unittest.main()
