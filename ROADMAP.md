# Roadmap: do pyGLAM ao paper

Documento de acompanhamento. Marque os itens (`- [ ]` → `- [x]`) à medida que forem sendo feitos.

---

## Alvo editorial

| Decisão | Escolha |
|---|---|
| **Revista** | *Journal of Computational Science* (Elsevier) |
| **Custo** | **US$ 0** — revista híbrida; ao recusar a opção open access na submissão não há APC (o artigo fica atrás de paywall) |
| **Indexação** | JCR, IF ~3, Q1/Q2, internacional |
| **Argumento de novidade** | Comparação sistemática de estimadores de GLD sob um arcabouço único, com seleção automática guiada por aderência |
| **Estudo de caso** | Dados sintéticos (estudo Monte Carlo fatorial) |

**Precedente:** o **UQpy** foi publicado no JCS (Olivier, Giovanis & Shields, 2020), e **Ketson dos Santos — coautor do pyGLAM — é desenvolvedor do UQpy.**

**Fallback, em ordem:** *Computational Statistics* (Springer, híbrida, IF ~1.3) → *J. of Statistical Computation and Simulation* (T&F, híbrida).
**Alvo de prestígio para uma v2.0 madura:** *Journal of Statistical Software* (diamond OA, IF 14.3, zero taxa — foi onde saiu o GLDEX).

**Evitar, dado o critério de não pagar:** SoftwareX (APC), IEEE Access (~US$ 2.045, onde saiu o pymoo), MDPI (APC). O JOSS é grátis mas **não tem JCR**.

> Atenção às premissas: o **mealpy não saiu na SoftwareX** — saiu no *Journal of Systems Architecture*. Os dois exemplos que serviram de inspiração (pymoo e mealpy) não são modelos de "JCR + de graça"; a rota que serve é a de assinatura em revista híbrida.

---

## Painel de progresso

| Fase | Itens | Status |
|---|---|---|
| ✅ Base existente | 6/6 | completa |
| ✅ Fase 3 (parcial) — aderência | 4/9 | **em andamento** |
| ⬜ Fase 0 — bugs bloqueantes | 0/8 | não iniciada |
| ⬜ Fase 1 — infraestrutura | 0/11 | não iniciada |
| ⬜ Fase 2 — núcleo científico | 0/11 | não iniciada |
| ⬜ Fase 4 — Monte Carlo | 0/5 | não iniciada |
| ⬜ Fase 5 — validação cruzada | 0/4 | não iniciada |
| ⬜ Fase 6 — docs e release | 0/7 | não iniciada |
| ⬜ Fase 7 — manuscrito | 0/10 | não iniciada |

**Estimativa até a submissão: ~9 meses.** Some 6–12 meses de revisão. Publicação realista: fim de 2027.

---

## ✅ O que já está pronto

### Base existente
- [x] Parametrização FKML (Freimer–Kollia–Mudholkar–Lin) — [`pyglam/glam.py`](pyglam/glam.py)
- [x] Função quantílica, `ppf`, `cdf`, `pdf`
- [x] Momentos teóricos (v₁–v₄) e momentos amostrais
- [x] Ajuste por **método dos momentos** (`fit_lambdas`, com `least_squares` e `root`)
- [x] Publicação no PyPI (`pyglam`, 5 releases até a 0.2.2)
- [x] Notebooks exploratórios (`teste.ipynb`, `teste_normal.ipynb`; `learn_pyglam.ipynb` e `pyglam_test.ipynb` na `renata_branch`)

### Módulo de aderência — parcial
- [x] Classe `Performance` com o método `performance(true, pred)` — [`pyglam/performance.py`](pyglam/performance.py)
- [x] Métricas: divergência KL, KS (estatística **e** p-valor), distância de Wasserstein, R² entre densidades, erros relativos em P5/P50/P95
- [x] Contrato de falha sem exceção (amostra vazia, constante ou com n < 2 → `NaN`, para a varredura terminar e as falhas virarem buracos nos mapas)
- [x] Documentação com 3 exemplos executáveis e saídas reais — [`docs_performance.md`](docs_performance.md)

---

## Fase 0 — Bugs bloqueantes `[~1 semana]`

Defeitos que invalidam a tese central do software. Um parecerista pega todos na primeira leitura.

- [ ] **`rvs()` não gera números aleatórios.** [`glam.py:177`](pyglam/glam.py#L177) usa `np.linspace(a, 1-a, size)` — grade determinística. Não existe RNG algum no pacote; duas chamadas devolvem o mesmo vetor. Trocar por amostragem por transformada inversa com `np.random.default_rng(seed)`, expondo `seed`/`random_state`.
  > **Prioridade máxima.** Enquanto isso não for feito, todo score de emulação (inclusive os da `Performance`) é otimista: compara amostra aleatória com malha fixa, sem o ruído amostral.
- [ ] Remover o código morto `f` em [`glam.py:181-183`](pyglam/glam.py#L181-L183)
- [ ] **`rvs()` aceita `lam1..lam4` e os ignora em silêncio** ([`glam.py:168`](pyglam/glam.py#L168)) — o corpo só usa `self.lam*`. Honrar ou remover
- [ ] **`pdf()` muda o tamanho do vetor.** A máscara em [`glam.py:249`](pyglam/glam.py#L249) descarta pontos sem avisar; `len(pdf(x)) != len(x)` quebra qualquer `plt.plot`. Devolver `nan`/`0.0` preservando o formato; escalar na entrada → escalar na saída
- [ ] **`cdf()`/`pdf()` levantam exceção fora do suporte.** `brentq` sem checagem de mudança de sinal: `cdf(50.0)` estoura em vez de devolver `1.0`. Tratar as caudas analiticamente
- [ ] **Vetorizar `cdf`/`pdf`.** Hoje é um `brentq` por elemento em laço Python — o oposto do "high-performance" anunciado. Usar busca binária vetorizada sobre a `ppf` ou interpolação monotônica
- [ ] **Validar a região de suporte no ajuste.** Sem restringir `λ2 > 0` nem a região de existência dos momentos (curtose exige `λ3, λ4 > -1/4`), as Gammas são avaliadas em polos. Hoje ajustar uma lognormal devolve `success=True` com `λ3 ≈ 27.9` — solução degenerada reportada como sucesso. Adicionar `bounds` e um método `region_of_validity()`
- [ ] **Metadados errados:** `numpy` é importado mas não declarado em [`pyproject.toml`](pyproject.toml); `pandas` é declarado e nunca usado. Adicionar `__all__` (hoje `from pyglam import *` vaza `np` e `sc`) e `__version__`

---

## Fase 1 — Infraestrutura mínima de credibilidade `[~2 semanas]`

Sem isto, *desk reject* em qualquer revista séria.

- [ ] **Arquivo `LICENSE`** (MIT, coerente com o `pyproject.toml`). Hoje **não existe** — a API do GitHub reporta `license: null`. É bloqueio absoluto
- [ ] Suíte `tests/` com pytest — propriedades matemáticas (`cdf(ppf(q)) == q`, monotonicidade da `ppf`, `pdf ≥ 0`, integral da `pdf` ≈ 1)
- [ ] Testes de casos-limite (`λ3 → 0` cai em `log`) e de recuperação de λ conhecidos
- [ ] Testes de regressão para cada bug da Fase 0
- [ ] Cobertura ≥ 90% (`pytest --cov=pyglam`)
- [ ] CI em `.github/workflows/tests.yml` — pytest em Python 3.10–3.13 × Linux/macOS/Windows
- [ ] `ruff` + `mypy` no CI
- [ ] **Consolidar as branches.** `main`, `joao_paulo/documentação` (Sphinx) e `renata_branch` (melhor material didático) estão divergentes e órfãs
- [ ] Remover `dist/*.whl` e `dist/*.tar.gz` do versionamento (estão commitados) e adicionar ao `.gitignore`
- [ ] Renomear `teste.ipynb`/`teste_normal.ipynb` para exemplos em inglês; corrigir o traceback `ValueError` salvo na última célula do `teste_normal.ipynb`; traduzir comentários mistos PT/EN
- [ ] `CITATION.cff`, `CONTRIBUTING.md`, `CHANGELOG.md`

---

## Fase 2 — Núcleo científico `[~3 meses]`

**A fase que dá substância ao paper.** É onde o trabalho do colega entra.

### 2.1 Parametrizações — `pyglam/parameterizations.py`
Refatorar `GlamFKML` para uma classe base `GLD` + subclasses, mantendo a API atual como fachada retrocompatível.

- [ ] **RS** — Ramberg–Schmeiser *(obrigatória: é a que o `gld` e o `GLDEX` implementam; sem ela não há comparação justa)*
- [ ] **VSL** — van Staden–Loots (baseada em L-momentos)
- [ ] **Tukey lambda** — caso particular de 1 parâmetro; serve para validar contra `scipy.stats.tukeylambda`
- [ ] `region_of_validity()` e condições de existência de momentos para cada uma

### 2.2 Estimadores — `pyglam/estimators.py` *(o coração do paper)*
Interface uniforme `fit(data, method=..., **kwargs) -> GLDFitResult`, carregando λ̂, diagnóstico de convergência **e** as métricas da `Performance`.

- [ ] **L-momentos** (Karvanen & Nuutinen 2008; Asquith 2007) — *prioridade máxima: o método dos momentos é o estimador mais fraco da GLD*
- [ ] **Percentis** (Karian & Dudewicz 1999)
- [ ] **Máxima verossimilhança** (via forma densidade-quantil)
- [ ] **MLE discretizada** (Su 2007 — é o que o GLDEX faz)
- [ ] **Starship** (King & MacGillivray 1999)
- [ ] Mínimos quadrados de quantis *(barato — e é o que os notebooks já **afirmam** fazer, embora o código case momentos; um parecerista pega essa inconsistência)*

### 2.3 Integração
- [ ] Interface compatível com `scipy.stats.rv_continuous` — barato, e é a vantagem que pacotes de R não podem oferecer

---

## Fase 3 — Módulo de aderência `[~1 mês]` — **em andamento**

- [x] Divergência KL entre densidades
- [x] Kolmogorov–Smirnov (estatística e p-valor)
- [x] Distância de Wasserstein
- [x] R² entre densidades + erros de percentis
- [ ] **Anderson–Darling**
- [ ] **Cramér–von Mises**
- [ ] **Qui-quadrado**
- [ ] **KS reamostrado por bootstrap** *(o diferencial do GLDEX)*
- [ ] **Log-verossimilhança → AIC / BIC** *(permite comparar estimadores formalmente — indispensável para a Fase 4)*
- [ ] **`pyglam.autofit(data)`** — roda todos os estimadores × todas as parametrizações, pontua por aderência e devolve o melhor ajuste com relatório diagnóstico.
  > **É este o entregável que fecha o argumento de novidade.** Nenhum concorrente (`gld`, `GLDEX`, `gldpy`) oferece isso num único comando.

### Pendência técnica registrada
- [ ] O `_relative_error` da `Performance` explode quando o percentil de referência fica perto de zero (duas amostras da mesma `N(0,1)` reportam `rel_err_p50 ≈ 2.0`). Mitigado com o parâmetro `percentile_tol`, mas vale decidir se o padrão deve virar uma tolerância relativa à dispersão dos dados
- [ ] Extrair `_prepare_data`, hoje duplicada em `glam.py` e `performance.py`, para um `pyglam/utils.py`

---

## Fase 4 — Estudo Monte Carlo `[~2 meses]` — os resultados do paper

`benchmarks/monte_carlo_study.py`, desenho fatorial completo.

> **Risco a gerenciar:** o JCS valoriza impacto aplicado. Um paper só com dados sintéticos passa **se e somente se** este estudo for rigoroso. "Ajustei uma normal e plotei" é rejeição certa. Plano B, se um parecerista pedir dado real: apêndice com um conjunto público (retornos financeiros são o domínio clássico da GLD, e foi como o GLDEX se demonstrou).

- [ ] **Distribuições-mãe (12+):** normal, lognormal, gama, Weibull, exponencial, beta, t de Student(5), Gumbel, uniforme, triangular, mistura bimodal, Pareto (cauda pesada)
- [ ] **Tamanhos amostrais:** n ∈ {50, 100, 500, 1.000, 5.000}
- [ ] **Réplicas:** R = 1.000 por célula, semente fixa
- [ ] **Métricas por célula:** viés e RMSE de λ̂; distância KS entre CDF ajustada e verdadeira; Wasserstein; **taxa de falha de convergência**; tempo de execução
- [ ] **Saídas:** heatmap estimador × distribuição-mãe; curvas de convergência em n; **tabela de recomendação prática** ("qual método usar para qual formato de dado") — que vira a regra do `autofit`

---

## Fase 5 — Validação cruzada contra as referências `[~3 semanas]`

Exigência explícita do JSS e expectativa forte no JCS. **A concorrência já publicou:** `GLDEX` (R) saiu no JSS, `gld` (R) tem 3 parametrizações no CRAN, e o **`gldpy` já existe no PyPI com 3 parametrizações**. Hoje o pyGLAM tem menos funcionalidade que os três.

- [ ] Gerar em R, uma única vez, valores de referência do `gld` e do `GLDEX` (mesmos λ, mesmos quantis) e **commitar como fixtures CSV** em `tests/fixtures/` — evita dependência de `rpy2` no CI
- [ ] Testes de concordância numérica com tolerância explícita para `pdf`/`cdf`/`ppf` e para λ̂ nas parametrizações compartilhadas
- [ ] **Benchmark de velocidade** contra `gldpy` (Python) e `gld` (R) — necessário para sustentar as alegações de "high-performance" e "fast emulation", hoje infundadas
- [ ] Seção do paper posicionando o pyGLAM frente a `gld`, `GLDEX` e `gldpy`, com vantagens **e** desvantagens

---

## Fase 6 — Documentação e release citável `[~3 semanas]`

- [ ] Mergear o Sphinx de `joao_paulo/documentação` para `main` e publicar no GitHub Pages
- [ ] Incluir `fit_lambdas`, `moments` e `Performance` no autodoc (o `pyglam.rst` atual documenta só `rvs/ppf/pdf/cdf`)
- [ ] Páginas de teoria: definição matemática da GLD, cada parametrização, cada estimador, com bibliografia
- [ ] Guia do usuário + galeria de exemplos + referência de API
- [ ] README reescrito: definição matemática, badges (PyPI, CI, cobertura, DOI), citação, referências
- [ ] Release **v1.0.0** no PyPI
- [ ] **Arquivamento no Zenodo para obter DOI**. Corrigir o `pyproject.toml` para que a metadata do wheel não descarte 4 dos 5 autores (o poetry-core mantém só o primeiro)

---

## Fase 7 — Manuscrito `[~2 meses, em paralelo às Fases 5–6]`

- [ ] 1. **Introduction** — por que distribuições quantílicas flexíveis; a lacuna no ecossistema Python
- [ ] 2. **Related software** — `gld`, `GLDEX`, `gldpy`, `scipy.stats`; o que falta em cada um
- [ ] 3. **Mathematical background** — GLD, parametrizações, regiões de validade, existência de momentos
- [ ] 4. **Estimation methods** — os 6–7 estimadores sob notação unificada
- [ ] 5. **Software architecture** — API, design, integração com scipy, reprodutibilidade
- [ ] 6. **Goodness-of-fit and automatic model selection** — o `autofit`
- [ ] 7. **Numerical study** — o Monte Carlo da Fase 4 *(seção mais longa)*
- [ ] 8. **Validation and performance** — Fase 5
- [ ] 9. **Conclusions and future work**
- [ ] **Preprint no arXiv (stat.CO)** antes de submeter — grátis, dá visibilidade e carimbo de data

---

## Autoria — resolver antes da submissão

Há um descompasso que as revistas checam:

| Pessoa | No `pyproject.toml` | Commits no git |
|---|---|---|
| Wanderlei M. Pereira Junior | ✅ | 18 |
| Marcos Luiz Henrique | ✅ | 1 |
| Renata Maria Pensin | ✅ | 4 |
| **João Paulo Lopes** | ❌ **ausente** | **5** (toda a documentação Sphinx) |
| **Ketson R. M. dos Santos** | ✅ | **0** |
| **Victor Hugo M. C. Souza** | ✅ | **0** |

- [ ] Definir a autoria final e registrar um **CRediT contributor statement**

**Sugestão de alocação:** Fase 2.2 (estimadores) para o colega que já está nisso · Fase 3 (aderência) para o Wanderlei · Fase 4 (Monte Carlo) para os estudantes · Fases 5 e 7 para os professores.

---

## Cronograma

| Fase | Duração | Acumulado |
|---|---|---|
| 0 — Bugs bloqueantes | 1 sem | 1 sem |
| 1 — Infraestrutura | 2 sem | 3 sem |
| 2 — Parametrizações + estimadores | 3 meses | ~4 meses |
| 3 — Aderência | 1 mês | ~5 meses |
| 4 — Monte Carlo | 2 meses | ~7 meses |
| 5 — Validação cruzada | 3 sem | ~8 meses |
| 6 — Docs + DOI | 3 sem | ~8,5 meses |
| 7 — Manuscrito | 2 meses (paralelo) | **~9 meses até submeter** |

---

## Critérios de verificação

Como comprovar que cada fase está pronta:

- [ ] **Fase 0** — duas chamadas de `rvs()` com sementes diferentes produzem vetores diferentes; `len(pdf(x)) == len(x)`; `cdf(1e6) == 1.0` sem exceção
- [ ] **Fase 1** — CI verde nas 12 combinações de SO × Python; cobertura ≥ 90%; `ruff check` e `mypy pyglam/` sem erro
- [ ] **Fase 2** — para cada parametrização, `cdf(ppf(q)) ≈ q` em `1e-10`; cada estimador recupera λ conhecidos com n = 10.000; Tukey lambda bate com `scipy.stats.tukeylambda`
- [ ] **Fase 3** — em dados normais, o KS não rejeita a GLD ajustada (p > 0,05); em dados bimodais mal ajustados, rejeita; `autofit` escolhe o método que a Fase 4 indica
- [ ] **Fase 4** — `python benchmarks/monte_carlo_study.py --seed 42` reproduz bit-a-bit todas as tabelas e figuras
- [ ] **Fase 5** — fixtures contra `gld`/`GLDEX` passam na tolerância declarada; tabela de tempos gerada contra `gldpy`
- [ ] **Fase 6** — site no ar; `pip install pyglam==1.0.0` em ambiente limpo roda o quick-start; DOI do Zenodo resolve
- [ ] **Fase 7** — um **script único de replicação** regenera todos os resultados do manuscrito do zero

---

## Resposta curta

**Publicável hoje?** Não — em nenhuma revista com JCR. São ~250 linhas no núcleo, uma parametrização, um estimador, zero testes, sem arquivo de licença, e a função de geração de amostras não gera nada aleatório.

**Publicável em ~9 meses?** Sim, com boa chance, se as Fases 2 a 4 forem executadas. O tema é legítimo, a lacuna em Python é real, e a equipe tem um coautor com precedente publicado exatamente na revista-alvo.
