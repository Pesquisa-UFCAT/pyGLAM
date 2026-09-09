import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy import stats

from pyglam import GlamFKML


class RandomVariatesTests(unittest.TestCase):
    def setUp(self):
        self.model = GlamFKML(0.0, 1.0, 0.2, 0.3)

    def test_seed_reproducibility_and_fresh_samples(self):
        first = self.model.rvs(1000, random_state=42)
        assert_array_equal(first, self.model.rvs(1000, random_state=42))
        self.assertFalse(np.array_equal(first, self.model.rvs(1000, random_state=43)))
        self.assertFalse(np.array_equal(self.model.rvs(1000), self.model.rvs(1000)))
        self.assertTrue(np.any(np.diff(first) < 0))
        self.assertTrue(np.any(np.diff(first) > 0))

    def test_generators_advance_and_can_replay_a_sequence(self):
        for factory in (np.random.default_rng, np.random.RandomState):
            with self.subTest(factory=factory):
                rng = factory(42)
                first = self.model.rvs(100, random_state=rng)
                second = self.model.rvs(100, random_state=rng)
                self.assertFalse(np.array_equal(first, second))
                replay = self.model.rvs(200, random_state=factory(42))
                assert_array_equal(np.concatenate((first, second)), replay)

    def test_shape_empty_and_invalid_sizes(self):
        self.assertEqual(self.model.rvs().shape, (1000,))
        self.assertEqual(self.model.rvs((2, 3), random_state=42).shape, (2, 3))
        self.assertEqual(self.model.rvs(0).shape, (0,))
        for size in (-1, 2.5, (2, -1)):
            with self.subTest(size=size), self.assertRaises((ValueError, TypeError)):
                self.model.rvs(size)

    def test_uniform_special_case_has_sampling_noise_and_keeps_tails(self):
        # FKML(0, 2, 1, 1) is Uniform(-0.5, 0.5).
        model = GlamFKML(0, 2, 1, 1)
        sample = model.rvs(100_000, quantile_trim=0.0, random_state=7)
        self.assertLess(stats.kstest(sample, "uniform", args=(-0.5, 1)).statistic, 0.01)
        counts, _ = np.histogram(sample, bins=10, range=(-0.5, 0.5))
        self.assertGreater(np.std(counts), 20)
        self.assertLess(sample.min(), -0.499)
        self.assertGreater(sample.max(), 0.499)
        trimmed = model.rvs(10_000, quantile_trim=0.1, random_state=7)
        self.assertTrue(np.all((trimmed >= -0.4) & (trimmed <= 0.4)))
        default = model.rvs(100_000, random_state=7)
        self.assertTrue(np.all((default >= -0.499) & (default <= 0.499)))
        assert_array_equal(default, model.rvs(100_000, quantile_trim=0.001, random_state=7))
        self.assertLess(stats.kstest(default, "uniform", args=(-0.499, 0.998)).statistic, 0.01)

    def test_logistic_special_case_including_zero_and_near_zero_shapes(self):
        # FKML(3, 2, 0, 0) is Logistic(loc=3, scale=0.5).
        with np.errstate(all="raise"):
            sample = GlamFKML(3, 2, 0, 0).rvs(50_000, quantile_trim=0.0, random_state=42)
            near_zero = GlamFKML(3, 2, 1e-10, -1e-10).rvs(50_000, quantile_trim=0.0, random_state=42)
        self.assertLess(stats.kstest(sample, "logistic", args=(3, 0.5)).statistic, 0.01)
        assert_allclose(near_zero, sample, rtol=0, atol=1e-8)

    def test_asymmetric_heavy_tail_quantiles(self):
        model = GlamFKML(1, 2, -0.3, 0.2)
        sample = model.rvs(100_000, quantile_trim=0.0, random_state=42)
        probabilities = np.array([0.01, 0.1, 0.5, 0.9, 0.99])
        # Assess probability errors using binomial sampling uncertainty; an
        # absolute tolerance in data units is unsuitable for a heavy tail.
        observed = np.mean(sample[:, None] <= model.ppf(probabilities), axis=0)
        standard_error = np.sqrt(probabilities * (1 - probabilities) / sample.size)
        self.assertTrue(np.all(np.abs(observed - probabilities) < 6 * standard_error))

    def test_overrides_apply_without_mutating_instance(self):
        overrides = dict(lam1=4, lam2=2, lam3=0, lam4=-0.1)
        expected = GlamFKML(**overrides).rvs(100, random_state=42)
        assert_array_equal(self.model.rvs(100, random_state=42, **overrides), expected)
        assert_array_equal(GlamFKML().rvs(100, random_state=42, **overrides), expected)
        self.assertEqual((self.model.lam1, self.model.lam2, self.model.lam3, self.model.lam4),
                         (0.0, 1.0, 0.2, 0.3))

    def test_invalid_parameters_and_trim_raise(self):
        for overrides in ({"lam2": 0}, {"lam2": -1}, {"lam1": np.nan},
                          {"lam3": np.inf}, {"lam4": [0.1]}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.model.rvs(10, **overrides)
        with self.assertRaises(ValueError):
            GlamFKML().rvs(10)
        for trim in (-0.1, 0.5, 1, np.nan, np.inf):
            with self.subTest(trim=trim), self.assertRaises(ValueError):
                self.model.rvs(10, quantile_trim=trim)

    def test_small_scale_does_not_discard_draws(self):
        model = GlamFKML(0, 1e12, 1, 1)
        sample = model.rvs(100, random_state=42)
        self.assertEqual(sample.shape, (100,))
        self.assertTrue(np.all(np.isfinite(sample)))

    def test_overflow_raises_instead_of_shortening_sample(self):
        with self.assertRaises(FloatingPointError):
            GlamFKML(0, 1, -1000, -1000).rvs(100, random_state=42)


if __name__ == "__main__":
    unittest.main()
