import numpy as np
from pyglam import GlamFKML

# GLD-FKML com lambda = (0; 1; 0,14; 0,14), forma próxima da normal
modelo = GlamFKML(0.0, 1.0, 0.14, 0.14)

# quantis lidos diretamente da função quantil Q(u)
mediana, p95 = modelo.ppf([0.50, 0.95])

# amostragem pela transformação inversa, X = Q(U)
amostra = modelo.rvs(size=1000, random_state=42)
