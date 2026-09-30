import pandas as pd

dados = pd.read_csv("estacao_A001_diario.csv")
v = dados["vento_ms"].dropna().to_numpy()      # velocidades médias diárias (m/s)

# método dos momentos a partir de 15 pontos iniciais
sol = GlamFKML().fit_lambdas(v, n_starts=15, seed=0)
gld = GlamFKML(*sol.x)                         # modelo ajustado
l1, l2, l3, l4 = sol.x

# limites do suporte, como na equação do suporte
a = l1 - 1 / (l2 * l3) if l3 > 0 else -np.inf
b = l1 + 1 / (l2 * l4) if l4 > 0 else np.inf
fora = np.sum((v < a) | (v > b))               # observações fora do suporte
