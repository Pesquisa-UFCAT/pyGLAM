# Plano do artigo para a Revista Brasileira de Ensino de Física

**Título provisório:** *Modelagem probabilística de dados físicos com a distribuição lambda generalizada: uma abordagem didática em Python*

**Destino:** Revista Brasileira de Ensino de Física (RBEF), seção Artigos Gerais.  
**Público:** estudantes e docentes de graduação em física e engenharia, com conhecimentos básicos de probabilidade, cálculo e Python.  
**Natureza da contribuição:** exposição didática de um procedimento de modelagem, com dois exemplos de dados físicos reais e materiais computacionais reproduzíveis.  
**Data deste plano:** 19 de setembro de 2026.

## Orientação para a aluna

O artigo ensinará a ir das observações de uma grandeza física ao ajuste e à interpretação de uma distribuição de probabilidade. A distribuição lambda generalizada (GLD), na parametrização FKML, será apresentada como uma ferramenta flexível. Distribuições clássicas servirão de referência para discutir adequação, simplicidade e significado físico.

**Objetivo geral sugerido para o manuscrito:**

> Apresentar uma abordagem didática para a modelagem probabilística de dados físicos reais utilizando a distribuição lambda generalizada, explicando seu ajuste pelo método dos momentos e comparando sua representação dos dados com distribuições clássicas pertinentes aos exemplos estudados.

**Pergunta que organiza o trabalho:** como ajustar, avaliar e interpretar uma distribuição flexível em dados físicos reais, e o que sua comparação com modelos clássicos permite aprender?

A contribuição deverá ficar visível na explicação dos conceitos, nas escolhas justificadas e na leitura dos resultados. A GLD pode apresentar vantagens, desempenho semelhante ou limitações: as conclusões serão escritas depois das análises. Um ajuste semelhante ao de uma família mais simples também é um resultado útil para a discussão.

O enquadramento proposto é compatível com a descrição de Artigos Gerais como exposição didática de física teórica, experimental ou computacional; a adequação final depende da contribuição e da avaliação editorial. [Escopo e instruções da RBEF](https://www.scielo.br/journal/rbef/about/?ilang=pt_BR).

### Estrutura e proporções sugeridas

As extensões abaixo são referências de organização do trabalho, não exigências da revista.

| Seção | Função no argumento | Extensão indicativa |
|---|---|---|
| 1. Introdução | Contextualizar o tema, a trajetória histórica e a proposta didática | 800–1.000 palavras |
| 2. A distribuição lambda generalizada | Ensinar função quantil, FKML e ajuste por momentos | 1.300–1.700 palavras |
| 3. Grandezas físicas reais e modelagem por distribuições | Relacionar hipóteses físicas, dados e modelos de referência | 650–850 palavras |
| 4. Ajuste e interpretação de dados de vento | Desenvolver o procedimento completo | 1.100–1.400 palavras |
| 5. Ajuste e interpretação de dados de temperatura | Reaplicar o procedimento com outra grandeza | 700–1.000 palavras |
| 6. Considerações finais | Responder ao objetivo, discutir limites e apresentar os materiais | 350–500 palavras |

Faixa inicial: aproximadamente 4.900–6.450 palavras, além de referências e legendas. Ajustar ao conteúdo efetivamente necessário.

## 1. Introdução

**Objetivo da seção:** explicar por que vale ensinar distribuições flexíveis por meio de dados físicos reais e situar a GLD nesse percurso.

### 1.1. Distribuições no ensino de física e engenharia

Começar pela necessidade de descrever a variabilidade de medições e grandezas físicas: a média resume a posição, mas não informa sozinha a dispersão, a assimetria, os quantis ou a frequência de valores elevados e baixos.

Usar exemplos breves: velocidade do vento, temperatura, resistência de materiais e tempos de falha. Distinguir variabilidade física, erro de medição e incerteza sobre parâmetros; uma mesma distribuição observada pode refletir mais de uma dessas fontes.

Apresentar a normal como uma referência familiar e perguntar em quais situações sua simetria e seu suporte são adequados. Se o texto afirmar que outras distribuições são pouco exploradas no ensino, sustentar a afirmação em literatura educacional ou em livros identificados. Evitar generalizações sobre todos os cursos de física e engenharia.

### 1.2. Percurso histórico: das distribuições particulares às famílias flexíveis

Reservar cerca de três ou quatro parágrafos. Organizar o percurso pelo problema enfrentado em cada etapa, em vez de fazer uma cronologia extensa.

| Marco | Ideia a apresentar | Fonte inicial |
|---|---|---|
| Pearson, 1895 | Desenvolvimento de curvas para representar formas assimétricas, ampliando o repertório além da normal | [Nota original de Pearson sobre curvas assimétricas](https://www.nature.com/articles/052317a0) |
| Johnson, 1949 | Construção de sistemas de distribuições por transformações | [Artigo original em Biometrika](https://academic.oup.com/biomet/article-abstract/36/1-2/149/200775) |
| Tukey e a família lambda | Uso da função quantil para controlar a forma de uma distribuição | Localizar e conferir a fonte histórica antes de atribuir uma data; a [documentação da SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.tukeylambda.html) serve de apoio matemático, não de fonte histórica original |
| Ramberg e Schmeiser, 1974 | Generalização relacionada à geração aproximada de variáveis aleatórias assimétricas | *An approximate method for generating asymmetric random variables*, [DOI](https://doi.org/10.1145/360827.360840); conferir o texto integral na leitura bibliográfica |
| Freimer e colaboradores, 1988 | Parametrização que será adotada no artigo, conhecida como FKML | [A study of the generalized Tukey lambda family](https://doi.org/10.1080/03610928808829820) |

O percurso deve conduzir à pergunta prática: como representar diferentes formas de dados com uma mesma expressão de função quantil? Evitar prometer uma família universal ou apresentar toda distribuição clássica como caso particular exato da GLD.

### 1.3. Por que utilizar distribuições flexíveis?

Explicar a possibilidade de representar diferentes assimetrias e comportamentos de cauda, mantendo um procedimento computacional comum. Apresentar também o custo: mais parâmetros, estimação mais delicada e necessidade de verificar o suporte e a adequação.

Separar três questões que voltarão nos exemplos: proximidade dos dados, simplicidade do modelo e plausibilidade física. Nenhuma delas se resolve apenas contando parâmetros ou olhando um histograma.

### 1.4. Aplicações contemporâneas e ferramentas computacionais

Um parágrafo sobre aplicações atuais é suficiente. Exemplos documentados: representação de distribuições de saída de simuladores estocásticos, quantificação de incertezas e integração de simulações com diferentes níveis de fidelidade. Citar [Zhu e Sudret, 2021](https://doi.org/10.1137/20M1337302) e, como desenvolvimento recente, o [MF-GLaM, divulgado pelos autores em 2025](https://sudret.ibk.ethz.ch/news-events/RSUQ-news/2025/11/new-paper-published-in-computer-methods-in-applied-mechanics-and-engineering.html). Esses trabalhos contextualizam o uso da GLD; seus emuladores não serão implementados neste artigo.

Apresentar um quadro curto das ferramentas:

| Ambiente | Ferramenta | Papel no texto |
|---|---|---|
| Python | `pyglam` | Implementação da GLD utilizada nos exemplos; apresentar a versão e a referência do software |
| Python | `scipy.stats` | Distribuições clássicas e funções estatísticas; `tukeylambda` é a família simétrica e não substitui a GLD-FKML de quatro parâmetros |
| R | [`gld`](https://CRAN.R-project.org/package=gld) | Exemplo de pacote que implementa parametrizações e métodos de ajuste da GLD |
| MATLAB | [`fitdist`](https://www.mathworks.com/help/stats/fitdist.html) | Exemplo de ferramenta para ajuste de distribuições; não atribuir suporte nativo à GLD-FKML sem comprovação |

Os exemplos executáveis do artigo serão em Python. A menção aos demais ambientes situa as ferramentas disponíveis, sem exigir versões dos notebooks em três linguagens.

### 1.5. Objetivo, contribuição didática e organização

Encerrar com o objetivo geral e com os aprendizados oferecidos: interpretar uma função quantil, compreender o ajuste por momentos, avaliar distribuições ajustadas e reconhecer os limites da interpretação física.

**Entrega da seção:** texto de introdução, referências históricas verificadas e quadro de ferramentas, caso ele ajude a leitura. Escrever a versão definitiva depois de conhecer os resultados.

## 2. A distribuição lambda generalizada

**Objetivo da seção:** permitir que o leitor compreenda o que a biblioteca ajusta e como os parâmetros se relacionam com características dos dados.

### 2.1. Da densidade à função quantil

Apresentar, nessa ordem:

1. A função de distribuição acumulada, $F(x)=P(X\leq x)$.
2. A densidade $f(x)$, para o caso contínuo, e sua relação com probabilidades em intervalos.
3. A função quantil $Q(u)=F^{-1}(u)$, entendida como inversa generalizada quando necessário.
4. A leitura de $Q(0{,}5)$ e $Q(0{,}95)$: mediana e percentil 95, nas unidades da variável.

Para distribuições contínuas estritamente crescentes no suporte, derivar $F(Q(u))=u$ e, portanto,

$$
f(Q(u))=\frac{1}{Q'(u)}.
$$

Explicar por que uma expressão explícita para $Q$ facilita a obtenção de quantis. Mostrar também que, para $U\sim\mathcal U(0,1)$, $X=Q(U)$ tem a distribuição desejada. Uma demonstração curta basta. A geração de amostras ilustra uma vantagem da formulação; os dois estudos principais continuam baseados em dados observados.

### 2.2. Formulação FKML e significado dos parâmetros

Adotar apenas a parametrização FKML no desenvolvimento:

$$
Q(u)=\lambda_1+\frac{1}{\lambda_2}
\left[\frac{u^{\lambda_3}-1}{\lambda_3}
-\frac{(1-u)^{\lambda_4}-1}{\lambda_4}\right],
\qquad 0<u<1,\quad \lambda_2>0.
$$

Nos casos de parâmetro de forma igual a zero, usar o limite
$\lim_{a\to0}(u^a-1)/a=\log u$.

| Parâmetro | Interpretação didática | Cuidado |
|---|---|---|
| $\lambda_1$ | Desloca a distribuição | Não é, em geral, a média nem a mediana |
| $\lambda_2$ | Controla inversamente a escala | Aumentá-lo, mantendo os demais fixos, reduz a dispersão; não é o desvio padrão |
| $\lambda_3$ e $\lambda_4$ | Controlam conjuntamente a forma, com efeitos sobre os lados esquerdo e direito | Não identificar um deles exclusivamente com a assimetria e o outro com a curtose |

Se $X$ tem unidade física, $\lambda_1$ tem essa unidade, $\lambda_2$ tem a unidade inversa e os parâmetros de forma são adimensionais.

**Figura 1:** quatro painéis variando um parâmetro por vez, com os outros fixos. Anotar os valores e comentar o que muda. Usar curvas de densidade e, se couber com legibilidade, uma curva quantil ilustrativa. A figura ensina o significado dos parâmetros; não é um mapa de mecanismos físicos.

### 2.3. Validade, suporte e momentos existentes

Derivar

$$
Q'(u)=\frac{u^{\lambda_3-1}+(1-u)^{\lambda_4-1}}{\lambda_2}>0.
$$

Na FKML, $\lambda_2>0$ garante essa monotonicidade no intervalo aberto, para parâmetros de forma reais finitos. Distinguir validade da distribuição e existência de momentos.

Explicar os limites do suporte:

$$
a=\begin{cases}
\lambda_1-\dfrac{1}{\lambda_2\lambda_3},&\lambda_3>0,\\
-\infty,&\lambda_3\leq0,
\end{cases}
\qquad
b=\begin{cases}
\lambda_1+\dfrac{1}{\lambda_2\lambda_4},&\lambda_4>0,\\
+\infty,&\lambda_4\leq0.
\end{cases}
$$

Para momentos absolutos de ordem inteira positiva $k$, a condição é $\lambda_3,\lambda_4>-1/k$. Assim, o ajuste pelos quatro primeiros momentos exige $\lambda_3,\lambda_4>-1/4$.

Relacionar isso ao caso físico: uma distribuição válida matematicamente pode atribuir probabilidade a velocidades negativas. O suporte deve ser examinado depois do ajuste. Não deduzir os extremos exatos do suporte usando `ppf(0)` e `ppf(1)`: na implementação local inspecionada, essas probabilidades são protegidas numericamente por valores próximos dos extremos.

### 2.4. Método dos momentos, passo a passo

Começar pela ideia: escolher parâmetros para aproximar características resumidas da amostra pelas características correspondentes do modelo.

Para observações $x_1,\ldots,x_n$, definir

$$
\bar x=\frac1n\sum_i x_i,
\qquad m_k=\frac1n\sum_i(x_i-\bar x)^k,
\qquad \widehat\gamma_1=\frac{m_3}{m_2^{3/2}},
\qquad \widehat\beta_2=\frac{m_4}{m_2^2}.
$$

Explicar com palavras: média, variância, assimetria e curtose. A curtose mede uma característica relacionada ao peso de observações afastadas do centro; evitar defini-la simplesmente como altura do pico.

Usar aqui variância com divisor $n$ e curtose de Pearson, cuja referência normal é 3. Essa é a convenção encontrada na implementação local da `pyglam`. A tabela descritiva pode usar outra convenção, mas qualquer diferença deve ser declarada.

Apresentar as quatro relações procuradas:

$$
\mu(\boldsymbol\lambda)\approx\bar x,\quad
\sigma^2(\boldsymbol\lambda)\approx m_2,\quad
\gamma_1(\boldsymbol\lambda)\approx\widehat\gamma_1,\quad
\beta_2(\boldsymbol\lambda)\approx\widehat\beta_2.
$$

Mostrar como obter momentos a partir de quantis, quando as integrais existem:

$$
E[X^r]=\int_0^1Q(u)^r\,du,
\qquad
\mu_r=\int_0^1[Q(u)-E(X)]^r\,du.
$$

Desenvolver a média como exemplo de cálculo:

$$
E[X]=\lambda_1+\frac1{\lambda_2}
\left[\frac1{1+\lambda_4}-\frac1{1+\lambda_3}\right].
$$

Para as demais expressões, explicar a construção e remeter à referência ou ao notebook. A seção não precisa reproduzir expansões algébricas longas para transmitir o método.

**Roteiro didático do ajuste:** calcular os quatro resumos amostrais; propor parâmetros iniciais; calcular os momentos do modelo; medir as discrepâncias; atualizar os parâmetros numericamente; verificar domínio, convergência e aderência aos dados.

Na versão local inspecionada, `fit_lambdas` utiliza por padrão mínimos quadrados sobre resíduos escalonados dos quatro momentos, com opção de múltiplos pontos iniciais. Descrever a versão efetivamente utilizada, sem afirmar que o algoritmo é BFGS ou que resolve obrigatoriamente apenas duas equações.

### 2.5. O que o ajuste por momentos garante e o que precisa ser avaliado

Quatro momentos iguais não identificam necessariamente uma única distribuição; mesmo dentro da FKML podem existir soluções diferentes. Portanto, verificar os momentos e verificar a aderência são etapas complementares. Esse ponto está documentado no [método de momentos do pacote `gld`](https://search.r-project.org/CRAN/refmans/gld/html/fit.fkml.moments.html).

Explicar ainda a sensibilidade dos momentos de ordem alta a valores extremos e a distinção entre a flexibilidade da família e a capacidade de um estimador específico de encontrá-la. Um resultado desfavorável com o método dos momentos não demonstra que nenhum ajuste possível da GLD funcionaria.

**Entrega da seção:** equações comentadas, Figura 1 e uma explicação que permita ao leitor acompanhar o ajuste sem conhecer o código interno da biblioteca.

## 3. Dados de grandezas físicas reais e modelagem por distribuições

**Objetivo da seção:** conectar a teoria estatística com as grandezas físicas dos exemplos e com aplicações em engenharia.

### 3.1. O que significa dizer que uma grandeza segue uma distribuição?

Explicar que se trata de um modelo para observações em condições especificadas. Informar sempre qual é a variável: velocidade instantânea ou média diária, temperatura de um período definido, precipitação diária ou máximo anual.

A forma observada depende das condições físicas, do procedimento de medição e da agregação. A semelhança entre curvas não prova, por si, um mecanismo gerador. Um histograma de temperaturas de vários meses, por exemplo, pode combinar populações com médias distintas.

### 3.2. Exemplos de relações entre grandezas e famílias de distribuições

Usar uma tabela acompanhada de explicações curtas:

| Grandeza ou situação | Modelo de referência | Hipóteses ou limites a explicar |
|---|---|---|
| Soma de muitas contribuições | Normal | Aproximações associadas ao teorema central do limite exigem condições; isso não demonstra que toda série de temperatura seja normal |
| Módulo de duas componentes gaussianas | Rayleigh | Componentes independentes, de média zero e mesma variância; a aplicação ao vento real precisa ser avaliada |
| Máximos por blocos | Gumbel como uma possibilidade | A família limite depende das condições e da distribuição de origem; nem todo máximo segue Gumbel |
| Resistência ou tempo de falha de materiais | Weibull como modelo possível | A adequação depende do material, do modo de falha e das condições do ensaio |

A Rayleigh merece uma dedução breve, pois será usada na seção 4: partir de $V=\sqrt{V_x^2+V_y^2}$ sob as hipóteses acima e apresentar

$$
f_V(v)=\frac{v}{\sigma^2}\exp\left(-\frac{v^2}{2\sigma^2}\right),\qquad v\geq0.
$$

Explicar o significado de $\sigma$ nessa construção. Não equiparar automaticamente vento atmosférico e gás ideal. A [implementação da Rayleigh na SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rayleigh.html) também permite conferir a convenção de escala.

### 3.3. Por que isso importa na confiabilidade de materiais?

Usar um exemplo conceitual curto: se uma resistência aleatória $R$ é comparada com uma solicitação fixa $s$, a probabilidade de falha é $P_f=P(R<s)=F_R(s)$ para um modelo contínuo. Isso mostra por que conhecer apenas a resistência média não basta e por que a região inferior da distribuição pode ser relevante.

Explicar que uma solicitação também aleatória exigiria modelar a relação entre resistência e solicitação. Aqui basta a ilustração com limiar fixo, sem acrescentar um estudo computacional de confiabilidade. Como apoio para discutir escolha física e empírica de modelos, consultar o [guia do NIST](https://www.itl.nist.gov/div898/handbook/apr/section2/apr21.htm).

### 3.4. Protocolo comum aos estudos de caso

Apresentar o procedimento que será executado nas duas seções seguintes: documentar os dados, examinar sua qualidade, definir o recorte, descrever a amostra, ajustar as famílias, comparar curvas e quantis e interpretar os resultados nas unidades da grandeza.

**Entrega da seção:** tabela de exemplos, dedução curta da Rayleigh e ilustração da confiabilidade por um limiar. A hipótese física de cada família deve ficar explícita.

## 4. Ajuste de distribuições e interpretação de dados de vento

**Objetivo da seção:** mostrar o procedimento completo, de forma que o leitor consiga reproduzi-lo.

### 4.1. Apresentação breve da biblioteca e do ambiente

Apresentar `pyglam`, versão utilizada, referência de software, instalação reproduzível e funções necessárias: `GlamFKML`, `fit_lambdas`, `pdf`, `cdf` e `ppf`. `rvs` pode aparecer na ilustração da transformação inversa.

Na cópia local inspecionada, o ajuste retorna `sol.x` e não atualiza a instância original; o modelo ajustado é criado com `GlamFKML(*sol.x)`. A versão declarada no `pyproject.toml` é 1.0.0, mas o manuscrito deverá registrar também o commit ou release efetivamente utilizado.

O endereço [Pesquisa-UFCAT/pyGLAM](https://github.com/Pesquisa-UFCAT/pyGLAM) consta dos metadados locais. Confirmar sua disponibilidade pública antes da entrega dos materiais. O repositório dos artefatos de aula será identificado separadamente na seção 6.

### 4.2. Origem, unidade e qualidade dos dados

Arquivo disponível: [`dados/estacao_A001_diario.csv`](dados/estacao_A001_diario.csv). Os materiais anteriores o identificam como uma extração de dados diários do INMET, estação A001. Recuperar a referência original e confirmar os metadados da estação, localização, instrumento, altura da medição e definição da média diária, quando disponíveis.

**Inspeção do CSV local realizada para este plano:**

| Item | Resultado |
|---|---|
| Intervalo de datas registrado | 06/05/2000 a 25/04/2025 |
| Número de linhas | 9.121 |
| Datas repetidas | 0 |
| Valores finitos de vento | 8.840 |
| Células vazias de vento | 281 |
| Zeros ou valores negativos de vento no CSV | 0 |
| Média do vento | Aproximadamente 2,345 m/s |
| Desvio padrão, divisor $n$ | Aproximadamente 0,711 m/s |
| Mínimo e máximo registrados | 0,7 e 5,5 m/s |

Esses números descrevem a cópia local; não certificam a qualidade instrumental nem documentam filtros feitos antes da exportação. Registrar o que foi excluído, os motivos e a contagem antes/depois. Tratar zeros segundo os metadados, se aparecerem em uma nova extração; não removê-los apenas porque dificultam o ajuste.

### 4.3. Caracterização exploratória

Mostrar a evolução temporal e a distribuição dos valores. Examinar lacunas, mudanças de patamar, sazonalidade e dependência entre dias sucessivos. Um gráfico de autocorrelação pode permanecer no notebook como diagnóstico.

Calcular tamanho amostral, média, mediana, desvio padrão, assimetria, curtose e quantis 5%, 50% e 95%. Explicar o significado desses resumos para a velocidade medida.

**Figura 2:** visão temporal e histograma da amostra analisada, com uma indicação simples da variação mensal quando ela for relevante. **Tabela 1:** descrição dos dados e estatísticas dos dois estudos de caso.

Se houver regimes claramente distintos, definir e justificar o recorte antes dos ajustes. Se for mantida a série agregada, apresentar o modelo como descrição da distribuição no período observado, sem inferir estacionariedade ou independência de todas as medições.

### 4.4. Famílias comparadas e estimação

**Comparação principal:** GLD-FKML por momentos e Rayleigh com origem fixada em zero. Para a Rayleigh, declarar o estimador empregado; a máxima verossimilhança fornece $\widehat\sigma=\sqrt{\sum_i v_i^2/(2n)}$. Sua utilização como estimador descritivo não valida automaticamente a hipótese de observações independentes.

O CSV contém médias diárias. Não esperar que tenham a mesma distribuição de velocidades instantâneas sob as hipóteses da dedução. O coeficiente de variação observado é aproximadamente 0,303, enquanto a Rayleigh com origem zero tem $\sqrt{(4-\pi)/\pi}\approx0,523$. Essa diferença já sinaliza uma limitação estrutural da referência, qualquer que seja sua escala.

**Complemento recomendado dentro do mesmo exemplo:** incluir Weibull de dois parâmetros, também com origem zero, como referência adicional de forma ajustável. Isso exige apenas mais uma curva no mesmo procedimento e torna mais informativa a discussão da vantagem da GLD. A Rayleigh é o caso de forma 2 da Weibull, com a conversão correspondente de escala.

Usar os mesmos registros em todas as famílias e declarar seus estimadores. A comparação será entre modelos ajustados por procedimentos especificados, sem atribuir toda diferença exclusivamente à família probabilística.

Para a GLD, registrar parâmetros, configuração do ajuste, domínio dos momentos, resíduos e suporte. Múltiplos inícios ajudam a examinar a estabilidade numérica. Fixar e documentar a regra de escolha da solução; `success=True` sozinho não demonstra um bom ajuste.

### 4.5. Desempenho: o que medir e como ensinar a leitura

Manter um conjunto pequeno de diagnósticos:

| Diagnóstico | Pergunta que responde |
|---|---|
| Histograma e densidades ajustadas | Como o modelo representa a forma geral? |
| Acumulada empírica e acumuladas ajustadas | Em que faixas de valores há diferenças nas probabilidades acumuladas? |
| QQ-plot | Quais quantis são subestimados ou superestimados? |
| Distância KS, $D_n=\sup_x|F_n(x)-\widehat F(x)|$ | Qual é a maior discrepância acumulada na amostra? |
| Diferenças nos quantis 5%, 50% e 95% | Qual é o tamanho da discrepância nas unidades da variável? |

Definir $\Delta_p=Q_{\rm modelo}(p)-Q_{\rm empírico}(p)$ e informar seu sinal. Isso torna o resultado interpretável em m/s e evita percentuais relativos pouco claros quando um quantil é próximo de zero.

**Figura 3:** diagnóstico dos modelos de vento, combinando densidade, acumulada e QQ-plots em painéis legíveis. **Tabela 2:** parâmetros, número de parâmetros livres, $D_n$, diferenças de quantis e indicação de eventual probabilidade atribuída a velocidades negativas.

Avaliar o modelo diretamente por sua CDF e sua função quantil, sem introduzir uma amostra simulada apenas para calcular métricas. O módulo `Performance` da versão local compara duas amostras e calcula algumas métricas com densidades estimadas por KDE; seu uso exigiria explicar esse procedimento e a variabilidade adicional da simulação. Ele não precisa ser o centro da avaliação deste artigo.

Reportar KS como distância descritiva. O p-valor usual não se aplica automaticamente quando os parâmetros são ajustados aos mesmos dados, há dependência temporal ou arredondamento. KL e $R^2$ de curvas de densidade não são necessários ao núcleo do estudo.

### 4.6. Interpretação física e didática

Responder com os resultados obtidos: a GLD descreve diferenças que Rayleigh não consegue representar? A Weibull já oferece representação comparável? Onde aparecem discrepâncias? O suporte é fisicamente aceitável? A diferença nos quantis é relevante na escala da grandeza?

Os resultados sobre os mesmos dados do ajuste são descritivos. Para afirmar desempenho fora da amostra, acrescentar uma avaliação em anos posteriores, com a divisão definida antes de ajustar os modelos. Não embaralhar os dias como se fossem observações independentes nem interpretar mudanças temporais como simples erro do estimador.

O vínculo com energia eólica pode ser mencionado, mas as médias diárias disponíveis não permitem recuperar diretamente $E[V^3]$ de velocidades instantâneas e a potência associada.

**Entrega da seção:** procedimento reproduzível, Figuras 2–3, tabelas de dados e desempenho e interpretação dos resultados, incluindo as limitações encontradas.

## 5. Segundo exemplo: ajuste e interpretação de dados de temperatura

**Escolha recomendada:** temperatura média diária da mesma estação, comparando GLD-FKML e normal. Isso permite reaproveitar a base e concentrar a diferença didática na sazonalidade, no recorte da amostra e na interpretação dos quantis.

### 5.1. Por que esta grandeza?

A coluna `temperatura_c` contém 8.781 valores finitos e 340 células vazias no CSV local. A unidade é grau Celsius. A normal será uma referência a avaliar; a grandeza “temperatura” não determina, por si, uma distribuição normal.

Apresentar o vínculo com o ensino: uma curva simétrica pode ser útil em algumas condições, mas a série observada pode refletir diferentes regimes e apresentar assimetria. O segundo exemplo ensina a formular e examinar essa aproximação.

### 5.2. Definição do recorte temporal

Mostrar primeiro a evolução temporal e a variação por mês. Como recorte inicial concreto, utilizar os registros de **julho**, mantendo uma mesma faixa do calendário anual: há **768 observações válidas**, distribuídas por 25 anos, na cópia local.

Julho é a proposta de recorte deste plano, não uma seleção por melhor aderência à normal ou à GLD. Se a documentação ou a qualidade dos dados exigir outra escolha, registrar a justificativa antes da comparação dos ajustes. Não testar meses até encontrar uma curva que confirme a conclusão desejada.

Restringir o calendário reduz uma fonte de heterogeneidade, mas não elimina dependência entre dias nem mudanças entre anos. Examinar os registros ao longo do tempo e declarar que a distribuição ajustada descreve as temperaturas do recorte observado.

### 5.3. Ajustes, gráficos e métricas

Reutilizar o procedimento da seção 4: mesmas definições de resumos, GLD por momentos, normal com parâmetros estimados e diagnósticos de densidade, acumulada e quantis. Para a normal, declarar a convenção de variância adotada no ajuste.

Evitar repetir a apresentação da biblioteca e das equações da GLD. Dedicar o texto ao que muda com a grandeza e com o recorte.

**Figura 4:** temperatura ao longo do calendário, com distribuição por mês e indicação do recorte escolhido. **Figura 5:** comparação normal × GLD no recorte, com curvas ajustadas e QQ-plots. Acrescentar os resultados às tabelas comuns, usando °C para diferenças de quantis.

### 5.4. Interpretação

Discutir se a flexibilidade acrescenta uma melhoria visível e interpretável e em quais regiões da distribuição. Caso a normal descreva adequadamente o recorte, discutir a utilidade de um modelo mais simples. Caso ambas falhem, discutir a heterogeneidade dos dados e os limites do procedimento.

Os quantis representam a distribuição das observações no recorte; não são intervalos de confiança da média, limites instrumentais nem previsões meteorológicas para um dia específico.

### 5.5. Por que não escolher automaticamente chuva e Gumbel?

Esta orientação serve à escolha do exemplo e não precisa entrar integralmente no manuscrito:

| Alternativa disponível | Situação verificada ou cuidado necessário | Encaminhamento |
|---|---|---|
| Precipitação diária | 8.759 valores finitos, dos quais 5.675 são zero | Um ajuste contínuo único não representa a massa em zero; analisar dias chuvosos mudaria o objeto para uma distribuição condicional |
| Máximas anuais de precipitação | O arquivo tem 312 registros de 29 estações; apenas 24 registros são da A001 | Não empilhar estações automaticamente. A homogeneidade regional, a dependência espacial e a cobertura dos anos exigem tratamento |
| Umidade relativa | 8.959 valores finitos; limites físicos de 0% a 100% | Pode ser alternativa com uma distribuição beta e avaliação de suporte, mas não é necessária ao plano principal |

No arquivo de máximas, a coluna de cobertura varia de 300 a 366 dias por registro. Dias faltantes podem ocultar justamente o extremo. Usar Gumbel demandaria definir blocos comparáveis, critérios de cobertura e uma base adequada; com apenas 24 máximas da A001, o ajuste de quatro momentos também teria incerteza elevada. Por essas razões, a temperatura é a proposta mais direta para este artigo.

**Entrega da seção:** segundo exemplo completo, com a mesma lógica de avaliação e uma discussão específica sobre o recorte temporal.

## 6. Considerações finais

**Objetivo da seção:** responder ao objetivo do artigo e indicar como o material pode ser utilizado no ensino.

### 6.1. Conclusões sustentadas pelos dois exemplos

Retomar o que a GLD conseguiu representar e em quais aspectos os modelos clássicos foram suficientes ou apresentaram limitações. Usar as métricas e figuras efetivamente obtidas. Delimitar as conclusões às variáveis, aos recortes e aos métodos de estimação analisados.

### 6.2. Vantagens da formulação por quantis

Destacar a obtenção direta de percentis, a leitura em unidades físicas e a possibilidade de amostragem pela transformação inversa. Relacionar essas vantagens aos exemplos do artigo e mencionar que CDF e densidade podem exigir procedimentos numéricos.

### 6.3. Desafios de ensino e limites do estudo

Retomar dificuldades concretas trabalhadas no texto: distinguir densidade e probabilidade; interpretar quantis; compreender momentos de ordem alta; reconhecer que convergência numérica não garante aderência; separar adequação estatística e explicação física; perceber efeitos do recorte e da agregação temporal.

Apresentar esses pontos como desafios conceituais abordados pelo material. Sem uma aplicação em turma, não afirmar que houve ganho de aprendizagem, redução de dificuldades ou eficácia pedagógica comprovada.

### 6.4. Artefatos de aula e acesso pelo GitHub

Inserir o endereço real do repositório dos materiais quando ele estiver criado e acessível. Não usar o link da biblioteca como se fosse, automaticamente, o link dos artefatos deste artigo.

Conteúdo sugerido para o repositório:

```text
README.md
requirements.txt
dados/
    README.md
    estacao_A001_diario.csv
notebooks/
    01_funcao_quantil_e_momentos.ipynb
    02_ajuste_vento.ipynb
    03_ajuste_temperatura.ipynb
figuras/
tabelas/
```

O README deve informar público, pré-requisitos, ordem dos notebooks, instalação e execução. O README dos dados deve registrar fonte original, acesso, definição das colunas, unidades, extração, filtros e condições de redistribuição; se necessário, fornecer instruções de obtenção em vez de redistribuir os arquivos.

Adicionar ao fim dos notebooks perguntas breves de interpretação: o que muda ao modificar um parâmetro? Onde os modelos divergem? O percentil 95 é o máximo? O que pode mudar quando se agregam vários meses? As perguntas utilizam as análises já realizadas.

**Entrega da seção:** conclusões proporcionais à evidência, limitações e link funcional dos materiais. Os notebooks deverão executar do início ao fim no ambiente documentado antes da submissão.

## Plano de execução para a aluna

| Etapa | Trabalho | Entrega verificável |
|---|---|---|
| 1 | Recuperar procedência e metadados dos CSVs | Ficha da estação, dicionário de dados e registro dos filtros |
| 2 | Explorar vento e temperatura e fechar os recortes | Tabela descritiva e gráficos temporais; escolhas registradas antes dos ajustes |
| 3 | Construir a explicação da função quantil e dos momentos | Rascunho da seção 2 e Figura 1 |
| 4 | Executar integralmente o exemplo de vento | Notebook, parâmetros, suporte, Figura 3 e métricas |
| 5 | Reaplicar o procedimento à temperatura | Notebook e comparação normal × GLD |
| 6 | Escrever a interpretação física dos resultados | Rascunhos das seções 3–5, sem conclusões antecipadas |
| 7 | Completar a revisão histórica e a contribuição didática | Introdução e referências conferidas |
| 8 | Consolidar os materiais e testar a execução completa | Repositório organizado, ambiente registrado e figuras reproduzidas |
| 9 | Redigir conclusão, resumo e versão em inglês dos elementos exigidos | Manuscrito completo, com links conferidos e formatação da revista |

### Critérios de conclusão do trabalho

- [ ] As seis seções seguem uma pergunta central e os dois exemplos estão completos.
- [ ] As fontes e os recortes dos dados estão documentados.
- [ ] A parametrização e as convenções dos momentos são consistentes entre texto e código.
- [ ] Foram examinados suporte, convergência e aderência dos modelos.
- [ ] A comparação usa os mesmos registros e informa os estimadores e números de parâmetros.
- [ ] Figuras e tabelas podem ser reproduzidas a partir dos notebooks.
- [ ] As conclusões distinguem resultados descritivos, hipóteses físicas e potencial didático.
- [ ] O repositório dos artefatos de aula tem um link real e instruções de execução.

## Leituras de partida

Os links ao longo do plano orientam a busca; conferir os textos integrais e completar os metadados bibliográficos antes de citar no manuscrito.

1. **Enquadramento editorial:** [RBEF — escopo e instruções](https://www.scielo.br/journal/rbef/about/?ilang=pt_BR).
2. **História e construção da família:** Pearson (1895), Johnson (1949), Ramberg e Schmeiser (1974) e Freimer e colaboradores (1988), indicados na seção 1.
3. **Estimação e limites do método:** [documentação de `fit.fkml.moments`](https://search.r-project.org/CRAN/refmans/gld/html/fit.fkml.moments.html) e referências primárias ali indicadas. A documentação apoia a implementação; a fundamentação matemática deve citar a literatura correspondente.
4. **Aplicações contemporâneas:** [Zhu e Sudret (2021)](https://doi.org/10.1137/20M1337302) e [MF-GLaM (2025), página dos autores com acesso à publicação](https://sudret.ibk.ethz.ch/news-events/RSUQ-news/2025/11/new-paper-published-in-computer-methods-in-applied-mechanics-and-engineering.html).
5. **Escolha de modelos físicos:** [NIST — escolha de uma distribuição para dados de vida/falha](https://www.itl.nist.gov/div898/handbook/apr/section2/apr21.htm).
6. **Exemplo de exposição didática de tratamento de dados na RBEF:** [Mínimos Quadrados na Regressão Simples e Múltipla e Propagação de Incertezas sem Derivadas](https://www.scielo.br/j/rbef/a/Q8wQCgsgLYFSSRnbtFFJZHF/). Ler para observar como conceitos e procedimentos são explicados; a estrutura deste plano segue o objetivo próprio do presente artigo.

**Limite da verificação realizada para este plano:** foram inspecionados os dois CSVs locais, a API e trechos da implementação local da `pyglam`, além das fontes bibliográficas e documentais indicadas. Não foram executados os ajustes dos estudos de caso nem comprovada a procedência completa dos CSVs. As contagens informadas são observações da cópia local; vantagens dos modelos, figuras finais e resultados de aprendizagem não foram presumidos.
