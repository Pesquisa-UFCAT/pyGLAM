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
| ✅ Fase 3 (parcial) — aderência | 4/10 | **em andamento** |
| 🟨 Fase 0 — bugs bloqueantes | 5/8 | **em andamento** |
| 🟨 Fase 1 — infraestrutura | 1/12 | **em andamento** |
| 🟨 Fase 2 — núcleo científico | 1/11 | **em andamento** |
| ⬜ Fase 4 — Monte Carlo | 0/5 | não iniciada |
| ⬜ Fase 5 — validação cruzada | 0/4 | não iniciada |
| 🟨 Fase 6 — docs e release | 4/9 | **em andamento** |
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
- [x] Métricas indisponíveis retornam `NaN`: amostras vazias impedem comparações amostrais; constantes ou com n < 2 impedem KDE. Métricas amostrais ainda podem existir nesses últimos casos; opções inválidas podem levantar `ValueError`.
- [x] Documentação com 3 exemplos executáveis e saídas reais — [`docs_performance.md`](docs_performance.md)

### Atualização técnica — 09/09/2026

- `rvs()` passou a gerar amostras aleatórias, com semente ou gerador em `random_state`, mantendo ordem e tamanho solicitado. Os argumentos `lam1..lam4` são respeitados.
- O padrão final permanece **`quantile_trim=0.001`**, excluindo 0,1% de cada cauda. `0.0` desativa esse corte; `1e-6` pode ser usado explicitamente, mas não foi adotado como padrão. O corte altera a distribuição amostrada e pode afetar os momentos.
- `qp_min` foi removido de **`rvs()`**, junto com o aviso de depreciação. Ele permanece ativo na **`pdf()`**: derivadas abaixo ou iguais ao limite retornam `NaN` na posição original.
- A função auxiliar de quantis usa `expm1` perto de zero e o limite logarítmico quando o parâmetro de forma é zero.
- `cdf()` e `pdf()` tratam os limites analíticos do suporte, entradas escalares, matrizes, vetores vazios, infinitos e `NaN`. A inversão usa a função quantil sem o corte numérico da `ppf`; `tol` controla a precisão absoluta em probabilidade.
- Ambiente local criado com **uv**, `.venv`, `.python-version` (Python 3.11) e `uv.lock`. NumPy declarado diretamente no `pyproject.toml`; instruções em [`README-ENV.md`](README-ENV.md).
- **18 testes passando** com `uv run --locked python -W error -m unittest discover -s tests -v`: 10 de amostragem e 8 de CDF/PDF, incluindo casos uniforme e logístico, caudas pesadas, limites do suporte, `cdf(ppf(q))` e integral da densidade.
- **Sphinx integrado ao checkout:** fontes adaptadas da branch `joao_paulo/documentação`, tema Read the Docs e grupo `docs` no TOML/lock. Introdução à família FKML antes da referência de classe, fórmula quantílica, parâmetros, suporte e corte de caudas.
- **Exemplos em todos os métodos públicos documentados:** `moments`, `fit_lambdas`, `rvs`, `ppf`, `cdf`, `pdf` e `Performance.performance`, além dos exemplos de criação das classes.
- **Guia de Performance:** fluxo ajuste → amostragem → avaliação, amostras semelhantes/deslocadas, repetições com sementes, percentis perto de zero e métricas indisponíveis. Documentada a diferença entre `r2_pdf` e o R² de uma rede que prevê lambdas.
- **Validação da documentação:** HTML gerado sem avisos, 100 verificações de exemplos aprovadas pelo builder `doctest` e 18 testes da biblioteca passando. Publicação do site ainda pendente.
- **GitHub Actions configurado:** `.github/workflows/docs.yml` valida testes, exemplos e HTML com uv/lock em pushes e PRs para `main`; publica no GitHub Pages após validação bem-sucedida de `main`. Execução manual disponível. A primeira execução remota e a conferência do site aguardam o push e a configuração de Pages como GitHub Actions.
- **Workflow validado localmente:** `actionlint` sem erros, sincronização `uv sync --locked --group docs` aprovada, 18 testes e 100 verificações dos exemplos passando, HTML sem avisos.

### Parametrização RS (Ramberg–Schmeiser) — 09/09/2026

- Nova classe **`GlamRS`** em [`pyglam/glam.py`](pyglam/glam.py), com a mesma interface pública da FKML: `moments`, `fit_lambdas`, `rvs`, `ppf`, `cdf`, `pdf`. A função quantílica é `Q(u) = λ1 + (u^λ3 − (1−u)^λ4)/λ2`, sem o limite logarítmico que a FKML precisa em `λ3 = 0` ou `λ4 = 0`. As duas famílias coincidem em `λ3 = λ4 = 1` (uniforme).
- Momentos teóricos pelos coeficientes A, B, C e D de Ramberg–Schmeiser; ajuste por casamento dos quatro momentos, como na FKML — e com a mesma limitação: o otimizador ainda não impõe `λ2 > 0` nem `λ3, λ4 > -1/4`.
- Página [`docs/source/glam_rs.rst`](docs/source/glam_rs.rst) com fórmula, tabela de parâmetros, suporte/caudas, escolha de método e referência da classe, já incluída no `toctree`.
- **142 verificações de exemplos** aprovadas pelo builder `doctest` (eram 100) e HTML sem avisos. Os 18 testes de `tests/` continuam passando, mas **ainda não cobrem a `GlamRS`**.
- Doctest de `moments` ajustado: somar `0.0` normaliza o `-0.0` da assimetria em amostras simétricas.
- **Documentação com texto justificado:** [`docs/source/_static/custom.css`](docs/source/_static/custom.css) registrado por `html_static_path`/`html_css_files` no `conf.py`. Justifica parágrafos, listas e definições da coluna de conteúdo; células de tabela, legendas e blocos de código mantêm o alinhamento do tema.

### Pendências para o próximo lançamento

- [ ] Cobrir a `GlamRS` com testes equivalentes aos da FKML: `rvs` com semente, `cdf(ppf(q))`, formato e suporte de `pdf`/`cdf`, e o caso uniforme em que RS e FKML coincidem.
- [ ] Unificar `GlamFKML` e `GlamRS` numa base comum — hoje as duas classes duplicam `_prepare_data`, ajuste, amostragem e inversão da CDF (~415 linhas repetidas).
- [ ] Restringir o ajuste à região válida dos quatro momentos e verificar os resíduos antes de aceitar os lambdas. Multi-start continua sendo uma melhoria a avaliar para a estabilidade dos alvos da rede.
- [ ] Concluir metadados/exportações e adicionar `LICENSE`, versão e notas de mudança, incluindo a nova amostragem aleatória e as alterações de `qp_min`/`tol`.
- [ ] Validar a primeira execução remota do workflow de testes/Sphinx já configurado; ampliar a cobertura dos ajustes e a matriz de versões de Python/SO.
- [ ] Selecionar GitHub Actions como fonte de Pages, conferir as permissões do ambiente `github-pages` para `main` e verificar a documentação após a primeira publicação automática.
- [ ] Gerar e instalar wheel/sdist em ambiente limpo, executar os exemplos e só então publicar a versão escolhida.

As pendências de vetorização, novos estimadores e estudo científico completo
continuam nas fases abaixo; os itens concluídos de documentação referem-se à
API FKML/Performance disponível neste checkout.

---

## Fase 0 — Bugs bloqueantes `[~1 semana]`

Defeitos que invalidam a tese central do software. Um parecerista pega todos na primeira leitura.

- [x] **Amostragem aleatória em `rvs()`.** Transformada inversa com `np.random.default_rng`, argumento `random_state` para semente ou gerador, preservação da ordem sorteada e do tamanho solicitado. Mantido o corte padrão `quantile_trim=0.001` (0,1% em cada cauda); `quantile_trim=0.0` desativa o corte, além da proteção numérica da função quantil.
- [x] Remover o código morto `f`, a filtragem por derivada e o argumento `qp_min` de `rvs()`.
- [x] **Honrar `lam1..lam4` em `rvs()`.** Parâmetros fornecidos substituem os da instância apenas naquela chamada; validar parâmetros finitos e `lam2 > 0`.
- [x] **Preservar o formato de `pdf()` e `cdf()`.** Escalar na entrada → escalar na saída; arrays mantêm dimensões e posições. `pdf()` retorna `NaN` nos pontos bloqueados por `qp_min`, sem removê-los.
- [x] **Tratar `cdf()`/`pdf()` fora do suporte.** Limites analíticos da FKML: CDF retorna 0/1 e PDF retorna zero fora do suporte. Nos extremos finitos, PDF usa o limite lateral da densidade. Inversão sem o corte da `ppf` evita falhas de intervalo em observações nas caudas.
- [ ] **Vetorizar `cdf`/`pdf`.** Hoje é um `brentq` por elemento em laço Python — o oposto do "high-performance" anunciado. Usar busca binária vetorizada sobre a `ppf` ou interpolação monotônica
- [ ] **Validar a região de suporte no ajuste.** Sem restringir `λ2 > 0` nem a região de existência dos momentos (curtose exige `λ3, λ4 > -1/4`), as Gammas são avaliadas em polos. Hoje ajustar uma lognormal devolve `success=True` com `λ3 ≈ 27.9` — solução degenerada reportada como sucesso. Adicionar `bounds` e um método `region_of_validity()`
- [ ] **Metadados e exportações — parcial:** NumPy já declarado em [`pyproject.toml`](pyproject.toml). Falta revisar a dependência de `pandas`, adicionar `__all__` (hoje `from pyglam import *` vaza `np` e `sc`) e `__version__`.

---

## Fase 1 — Infraestrutura mínima de credibilidade `[~2 semanas]`

Sem isto, *desk reject* em qualquer revista séria.

- [x] **Ambiente de desenvolvimento com uv.** `.venv`, `uv.lock`, Python 3.11 selecionado em `.python-version` e comandos de sincronização/testes documentados em `README-ENV.md`.
- [ ] **Arquivo `LICENSE`** (MIT, coerente com o `pyproject.toml`). Hoje **não existe** — a API do GitHub reporta `license: null`. É bloqueio absoluto
- [ ] Suíte `tests/` com pytest — **parcial:** 18 testes com `unittest` já passam, incluindo `cdf(ppf(q))`, monotonicidade da CDF, `pdf ≥ 0` e integral da `pdf` ≈ 1. Ampliar propriedades e integrar a execução/cobertura no CI.
- [ ] Testes de casos-limite e recuperação de λ conhecidos — **parcial:** formas iguais/próximas de zero, caudas pesadas e limites do suporte cobertos; recuperação dos parâmetros pelo ajuste ainda pendente.
- [ ] Testes de regressão para cada bug da Fase 0 — **parcial:** geração aleatória, parâmetros por chamada, formato da PDF/CDF e comportamento fora do suporte cobertos.
- [ ] Cobertura ≥ 90% (`pytest --cov=pyglam`)
- [ ] CI multiplataforma — **parcial:** `.github/workflows/docs.yml` executa unittest, exemplos Sphinx e HTML em Linux com o Python de `.python-version`; falta ampliar para Python 3.10–3.13 × Linux/macOS/Windows e confirmar a primeira execução remota.
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

- [x] **RS** — Ramberg–Schmeiser: classe `GlamRS` com a interface completa (`moments`, `fit_lambdas`, `rvs`, `ppf`, `cdf`, `pdf`) e página própria na documentação. *Falta a refatoração:* hoje é uma classe paralela à `GlamFKML`, não uma subclasse de uma base `GLD` comum, e o ajuste ainda não valida a região de existência dos momentos.
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

Exigência explícita do JSS e expectativa forte no JCS. **A concorrência já publicou:** `GLDEX` (R) saiu no JSS, `gld` (R) tem 3 parametrizações no CRAN, e o **`gldpy` já existe no PyPI com 3 parametrizações**. Com a RS, o pyGLAM passa a ter duas parametrizações e um único estimador — ainda atrás dos três.

- [ ] Gerar em R, uma única vez, valores de referência do `gld` e do `GLDEX` (mesmos λ, mesmos quantis) e **commitar como fixtures CSV** em `tests/fixtures/` — evita dependência de `rpy2` no CI
- [ ] Testes de concordância numérica com tolerância explícita para `pdf`/`cdf`/`ppf` e para λ̂ nas parametrizações compartilhadas
- [ ] **Benchmark de velocidade** contra `gldpy` (Python) e `gld` (R) — necessário para sustentar as alegações de "high-performance" e "fast emulation", hoje infundadas
- [ ] Seção do paper posicionando o pyGLAM frente a `gld`, `GLDEX` e `gldpy`, com vantagens **e** desvantagens

---

## Fase 6 — Documentação e release citável `[~3 semanas]`

- [x] Incorporar e adaptar as fontes Sphinx de `joao_paulo/documentação` ao checkout de `main`, com configuração e dependências reproduzíveis via uv. Artefatos gerados ficam fora do versionamento.
- [ ] Publicação no GitHub Pages — **workflow pronto** em `.github/workflows/docs.yml`; falta confirmar a fonte GitHub Actions nas configurações do repositório e validar o site após o primeiro push para `main`.
- [x] Incluir `fit_lambdas`, `moments` e `Performance` no autodoc, com introdução às classes e exemplos nos métodos públicos.
- [ ] Páginas de teoria — **parcial:** FKML e RS, função quantil, parâmetros e suporte documentados; faltam bibliografia científica completa e páginas dos futuros estimadores/parametrizações.
- [x] Guia do usuário, exemplos executáveis e referência da API atual: `quickstart.rst`, `pyglam.rst`, `glam_rs.rst` e `performance.rst`, com o texto do site justificado por CSS próprio.
- [x] Validar HTML sem avisos e executar os exemplos Sphinx: 142 verificações aprovadas pelo builder `doctest`. Instruções em `README-ENV.md`.
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

- [x] **Regressões de amostragem e avaliação da Fase 0** — sementes diferentes geram vetores diferentes; PDF/CDF preservam formato; CDF retorna 0/1 e PDF zero fora de suportes limitados. A Fase 0 ainda tem as pendências de vetorização, ajuste e metadados listadas acima.
- [ ] **Fase 1** — CI verde nas 12 combinações de SO × Python; cobertura ≥ 90%; `ruff check` e `mypy pyglam/` sem erro
- [ ] **Fase 2** — para cada parametrização, `cdf(ppf(q)) ≈ q` em `1e-10`; cada estimador recupera λ conhecidos com n = 10.000; Tukey lambda bate com `scipy.stats.tukeylambda`
- [ ] **Fase 3** — em dados normais, o KS não rejeita a GLD ajustada (p > 0,05); em dados bimodais mal ajustados, rejeita; `autofit` escolhe o método que a Fase 4 indica
- [ ] **Fase 4** — `python benchmarks/monte_carlo_study.py --seed 42` reproduz bit-a-bit todas as tabelas e figuras
- [ ] **Fase 5** — fixtures contra `gld`/`GLDEX` passam na tolerância declarada; tabela de tempos gerada contra `gldpy`
- [ ] **Fase 6** — site no ar; `pip install pyglam==1.0.0` em ambiente limpo roda o quick-start; DOI do Zenodo resolve
- [ ] **Fase 7** — um **script único de replicação** regenera todos os resultados do manuscrito do zero

---

## Resposta curta

**Pronto para submissão hoje?** Ainda faltam validação científica e infraestrutura: há duas parametrizações (FKML e RS) e um único estimador, sem arquivo de licença ou estudo Monte Carlo completo. A geração aleatória já foi corrigida, o ambiente uv está configurado e 18 testes de amostragem e avaliação passam; as demais etapas continuam pendentes.

**Publicável em ~9 meses?** Sim, com boa chance, se as Fases 2 a 4 forem executadas. O tema é legítimo, a lacuna em Python é real, e a equipe tem um coautor com precedente publicado exatamente na revista-alvo.
