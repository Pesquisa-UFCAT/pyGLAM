import scipy as sc
import numpy as np

class Glam:
    def __init__(self) -> None:
        pass

    def moments(self, data: list | np.ndarray):
        """Compute the first four moments of a dataset.
        
        :param data: dataset

        :return: [0] = mean, [1] = variance, [2] = skewness, [3] = kurtosis
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

    def _fit_lambdas(self, data, x0=None, method="least_squares"):
        """
        Ajusta (lambda1..lambda4) via método dos momentos:
        resolve sistema: theory_moments(lam) - sample_moments = 0
        """

        mean_hat, var_hat, skew_hat, kurt_hat = self.moments(data)
        target = np.array([mean_hat, var_hat, skew_hat, kurt_hat], dtype=float)

        if x0 is None:
            sd = np.sqrt(var_hat) if var_hat > 0 else 1.0
            x0 = np.array([mean_hat, 1.0/max(sd, 1e-8), 0.5, 0.5], dtype=float)
        else:
            x0 = np.array(x0, dtype=float)

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

        if method == "root":
            sol = sc.optimize.root(residuals, x0, method="hybr")
            return sol

        # default: least_squares (mais robusto)
        sol = sc.optimize.least_squares(residuals, x0, method="trf")

        return sol
