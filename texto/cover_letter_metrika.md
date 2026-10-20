Wanderlei Malaquias Pereira Junior
Universidade Federal de Catalão (UFCAT), Catalão, Goiás, Brazil
Department of Civil, Environmental, and Geo-Engineering, University of Minnesota, Minneapolis, MN, USA
wanderlei_junior@ufcat.edu.br

[Date]

The Editors-in-Chief
*Metrika*
Springer

**Re: Submission of the manuscript "pyGLAM: Moment matching and diagnostics for generalized lambda distributions in Python"**

Dear Editors,

On behalf of all authors, I am pleased to submit the manuscript "pyGLAM: Moment matching and diagnostics for generalized lambda distributions in Python" for consideration as an original article in *Metrika*.

The generalized lambda distribution (GLD) is a four-parameter family defined through its quantile function. It is widely used to summarise a simulated or empirical sample by a distribution rather than by a few moments. The family has several parameterizations, notably the Ramberg-Schmeiser (RS) and the Freimer-Kollia-Mudholkar-Lin (FKML) forms. These parameterizations cover different regions of the parameter space, impose different validity conditions, and can be fitted by several competing estimators. In practice, a user of a fitted GLD usually cannot tell whether a poor fit comes from the sample, from the family, or from the estimator. The manuscript addresses this question.

The main contributions are the following.

1. **A decomposition of the fitting error.** The error of a GLD fit is separated into a sampling component, which decays at the rate N^(-1/2), and a structural component, which does not. The decomposition is estimated from the slope of the Kolmogorov-Smirnov distance against sample size, with a bootstrap confidence interval. In a factorial Monte Carlo study (two parameterizations, six target distributions, four sample sizes, 100 replicates per cell), the normal and the uniform show no structural component, the Gumbel and the Beta a small one, and the Gamma a floor under both parameterizations. On a Student-t with four degrees of freedom, the same diagnostic points to the estimator rather than to the family.

2. **Roots of the moment equations that are not distributions.** Four-moment matching admits roots that reproduce the four sample moments, are reported as converged by the optimizer, and have a decreasing quantile function, so they are not valid distributions. We impose the admissible regions of the RS form inside the objective and screen the candidates of both forms for monotonicity. All 4,800 selected fits in the study were valid.

3. **An open-source implementation.** The methods are implemented in the Python package pyGLAM (MIT licence, available from PyPI, version 1.1.0). It offers both parameterizations behind one interface, multi-start moment matching restricted to valid distributions, and a goodness-of-fit module. Its quantile, distribution and density functions and its moments agree with the R packages `gld` and `GLDEX` to numerical precision. Its fits of the same samples are valid, reproduce the sample moments, and run several times faster.

We believe the manuscript suits *Metrika* because its central results concern the statistical behaviour of an estimator, namely how the error of moment matching for a quantile-defined family splits into sampling and structural parts and when the moment equations return invalid solutions. The software is the vehicle that makes these results reproducible and usable. The package, its documentation with executable examples, and the notebooks that generate every table and figure are publicly available, so all results can be reproduced.
Source code: https://github.com/Pesquisa-UFCAT/pyGLAM
Documentation: https://pesquisa-ufcat.github.io/pyGLAM/

The manuscript is original and is not under consideration for publication elsewhere. All authors have read and approved the submitted version and agree to its submission to *Metrika*. We have no conflicts of interest to declare. [Confirm both statements and add funding information, if any.]

[Optional: we suggest the following reviewers, who are familiar with quantile-defined distributions and their estimation: Name 1 (affiliation, e-mail); Name 2 (affiliation, e-mail).]

Thank you for considering our work. I look forward to hearing from you.

Sincerely,

Wanderlei Malaquias Pereira Junior (corresponding author)
on behalf of Marcos Luiz Henrique, Ketson Roberto Maximiano dos Santos, Victor Hugo Moreira do Couto Souza and Renata Maria Pensin
