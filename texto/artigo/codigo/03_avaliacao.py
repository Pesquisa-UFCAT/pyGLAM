# desvios quantílicos, em m/s
p = np.array([0.05, 0.50, 0.95])
delta = gld.ppf(p) - np.quantile(v, p)

# distância de Kolmogorov-Smirnov entre a acumulada empírica e a do modelo
x = np.sort(v)
n = x.size
F = gld.cdf(x)
i = np.arange(1, n + 1)
Dn = max(np.max(i / n - F), np.max(F - (i - 1) / n))
