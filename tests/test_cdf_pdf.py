import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy import integrate, stats

from pyglam import GlamFKML


class DistributionEvaluationTests(unittest.TestCase):
    def test_uniform_support_and_scalar_results(self):
        model = GlamFKML(0, 2, 1, 1)
        values = np.array([-np.inf, -50, -0.5, -0.2, 0, 0.5, 50, np.inf, np.nan])
        assert_allclose(model.cdf(values), stats.uniform.cdf(values, loc=-0.5), atol=1e-12)
        assert_allclose(model.pdf(values), stats.uniform.pdf(values, loc=-0.5), atol=1e-12)
        for method in (model.cdf, model.pdf):
            self.assertIsInstance(method(0.0), float)
            self.assertIsInstance(method(np.array(0.0)), float)

    def test_multidimensional_empty_and_nonfinite_inputs(self):
        model = GlamFKML(0, 2, 1, 1)
        values = [[-1, 0, 1], [-np.inf, np.inf, np.nan]]
        assert_allclose(model.cdf(values), [[0, 0.5, 1], [0, 1, np.nan]])
        assert_allclose(model.pdf(values), [[0, 1, 0], [0, 0, np.nan]])
        for method in (model.cdf, model.pdf):
            self.assertEqual(method([]).shape, (0,))
            self.assertEqual(method(np.empty((2, 0))).shape, (2, 0))
            self.assertTrue(np.isnan(method(np.nan)))

    def test_logistic_reference_including_unbounded_tails(self):
        model = GlamFKML(3, 2, 0, 0)
        values = np.linspace(-7, 13, 101)
        assert_allclose(model.cdf(values), stats.logistic.cdf(values, loc=3, scale=0.5), atol=1e-12)
        assert_allclose(model.pdf(values), stats.logistic.pdf(values, loc=3, scale=0.5), atol=2e-12)
        assert_array_equal(model.cdf([-np.inf, np.inf]), [0, 1])
        assert_array_equal(model.pdf([-np.inf, np.inf]), [0, 0])
        # A finite observation beyond ppf's numerical guard is still inside support.
        tail_model = GlamFKML(0, 1, 0, 0)
        assert_allclose(tail_model.cdf(-50, tol=1e-25), stats.logistic.cdf(-50), rtol=1e-3)
        assert_allclose(tail_model.pdf(-50, tol=1e-25), stats.logistic.pdf(-50), rtol=1e-3)

    def test_round_trip_for_bounded_one_sided_and_heavy_tailed_models(self):
        probabilities = np.array([1e-8, 0.001, 0.1, 0.5, 0.9, 0.999, 1 - 1e-8])
        for shapes in ((0.2, 0.3), (0, 0.5), (0.5, 0), (-0.3, 0.2), (-0.4, -0.1)):
            with self.subTest(shapes=shapes):
                model = GlamFKML(2, 1.5, *shapes)
                values = model.ppf(probabilities)
                assert_allclose(model.cdf(values), probabilities, atol=1e-12)
                self.assertTrue(np.all(model.pdf(values) >= 0))
                self.assertTrue(np.all(np.diff(model.cdf(values)) > 0))
        self.assertEqual(GlamFKML(2, 1, 0.5, 0).cdf(-50), 0)
        self.assertEqual(GlamFKML(2, 1, 0, 0.5).cdf(50), 1)

    def test_finite_endpoint_density_limits(self):
        for shape, expected_density in ((0.5, 0), (1, 1), (2, 2)):
            with self.subTest(shape=shape):
                model = GlamFKML(0, 2, shape, shape)
                endpoints = np.array([-1, 1]) / (2 * shape)
                assert_array_equal(model.cdf(endpoints), [0, 1])
                assert_allclose(model.pdf(endpoints), expected_density)

    def test_pdf_threshold_preserves_positions(self):
        model = GlamFKML(0, 1e9, 0.5, 0.5)
        values = model.ppf([0.0001, 0.5, 0.9999])
        density = model.pdf(values)
        self.assertEqual(density.shape, values.shape)
        assert_array_equal(np.isnan(density), [False, True, False])
        self.assertTrue(np.all(np.isfinite(model.pdf(values, qp_min=0))))
        self.assertTrue(np.isnan(model.pdf(0)))

    def test_pdf_integrates_to_one(self):
        model = GlamFKML(0, 1, 0.5, 0.5)
        mass, error = integrate.quad(model.pdf, -2, 2, epsabs=1e-8)
        self.assertAlmostEqual(mass, 1.0, places=7)
        self.assertLess(error, 1e-7)

    def test_invalid_parameters_and_tolerances(self):
        for model in (GlamFKML(), GlamFKML(0, 0, 1, 1),
                      GlamFKML(0, -1, 1, 1), GlamFKML(0, 1, np.nan, 1)):
            for method in (model.cdf, model.pdf):
                with self.subTest(model=model, method=method), self.assertRaises(ValueError):
                    method(0)
        model = GlamFKML(0, 1, 1, 1)
        for tol in (0, -1, 0.5, np.nan, np.inf):
            for method in (model.cdf, model.pdf):
                with self.subTest(tol=tol, method=method), self.assertRaises(ValueError):
                    method(0, tol=tol)
        for threshold in (-1, np.nan, np.inf):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                model.pdf(0, qp_min=threshold)


if __name__ == "__main__":
    unittest.main()
