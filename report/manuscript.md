Neurolytics: Transformation and Scale Sensitivity in Treatment-Effect Ranking
Draft status: Member P manuscript contribution. This document covers estimands, simulation design, methods, model cards, related work, and Appendix A. It is not the complete project manuscript.

1. Estimands and theoretical framework
1.1 Level-scale conditional average treatment effect
Let \(X\) denote pre-treatment covariates, \(W\in\{0,1\}\) randomized treatment assignment, and \(Y(1)\) and \(Y(0)\) the potential outcomes under treatment and control.

The primary estimand is the conditional average treatment effect on the original outcome scale:
\[

\tau_{\mathrm{level}}(x)

=

\mathbb{E}[Y(1)-Y(0)\mid X=x].

\]
This is the target relevant to prioritizing individuals by expected incremental outcome on the original scale.
1.2 Transformed-scale estimand
Let \(m\) be an outcome transformation and \(a>0\) a positive scaling constant. Define
\[

\tau_{m,a}(x)

=

\mathbb{E}[m(aY(1))-m(aY(0))\mid X=x].

\]
In general,
\[

\tau_{m,a}(x)\neq m(a\tau_{\mathrm{level}}(x)).

\]
The transformation is applied to each potential outcome before taking the conditional expectation and treatment contrast. Consequently, nonlinear transformations can change both the magnitude and the ordering of conditional treatment effects.

The central methodological question is whether treatment-effect rankings obtained after transforming and rescaling outcomes agree with rankings based on the original level-scale estimand.
1.3 Zero-inflated lognormal representation
The simulation outcome has a point mass at zero and a continuous positive component. Conditional on treatment \(w\) and covariates \(X=x\), write
\[

Y(w)=B(w)S(w),

\]
where
\[

B(w)\mid X=x\sim\operatorname{Bernoulli}(p_w(x))

\]
and
\[

\log S(w)\mid X=x\sim

\mathcal{N}(\mu_w(x),\sigma^2).

\]
The purchase indicator and positive spending component are generated independently conditional on \(X\) and treatment.

The conditional mean is
\[

\mathbb{E}[Y(w)\mid X=x]

=

p_w(x)\exp\left(\mu_w(x)+\frac{\sigma^2}{2}\right).

\]
Therefore,
\[

\boxed{

\tau_{\mathrm{level}}(x)

=

\exp\left(\frac{\sigma^2}{2}\right)

\left[

p_1(x)e^{\mu_1(x)}

\-

p_0(x)e^{\mu_0(x)}

\right].

}

\]
This expression is the known ground-truth CATE used for evaluating estimated treatment-effect rankings and policies.
1.4 Small-scale behavior
Suppose \(m(0)=0\), \(m'(0)=1\), and the relevant expectations satisfy the integrability conditions required to exchange the limit and expectation. A first-order expansion gives
\[

m(aY)=aY+o(a)

\]
as \(a\to0^+\). It follows that
\[

\boxed{

\lim_{a\to0^+}

\frac{\tau_{m,a}(x)}{a}

=

\tau_{\mathrm{level}}(x).

}

\]
Thus, for transformations satisfying these conditions, the transformed contrast approaches the level-scale contrast up to a positive multiplicative factor at sufficiently small scales.
1.5 Large-scale behavior under the simulated DGP
For the zero-inflated lognormal outcome distribution, the positive-outcome component dominates the large-\(a\) behavior of the log1p and arcsinh transformations.

For log1p,
\[

m(aY)=\log(1+aY).

\]
For positive \(Y\),
\[

\log(1+aY)=\log a+\log Y+o(1).

\]
Since \(Y=0\) contributes zero,
\[

\mathbb{E}[\log(1+aY(w))\mid X=x]

=

p_w(x)(\log a+\mu_w(x))+o(1).

\]
Consequently,
\[

\boxed{

\tau_{\log1p,a}(x)

=

(p_1-p_0)\log a

\+

p_1\mu_1-p_0\mu_0

\+

o(1).

}

\]
For arcsinh,
\[

\operatorname{arcsinh}(aY)

=

\log(2aY)+o(1)

\]
for positive \(Y\) at large \(a\). Therefore,
\[

\boxed{

\tau_{\operatorname{arcsinh},a}(x)

=

(p_1-p_0)\log a

\+

(p_1-p_0)\log 2

\+

p_1\mu_1-p_0\mu_0

\+

o(1).

}

\]
These expressions show why scale can affect rankings: the coefficient of \(\log a\) depends on the treatment effect on purchase probability, which can vary across individuals.
1.6 Transformation-specific scale properties
The four primary transformations are identity, log1p, arcsinh, and square root.

For identity,
\[

\tau_{\mathrm{identity},a}(x)

=

a\tau_{\mathrm{level}}(x).

\]
Since \(a>0\), the population ranking is unchanged across scales.

For square root,
\[

\sqrt{aY}=\sqrt{a}\sqrt{Y},

\]
so
\[

\tau_{\sqrt{\cdot},a}(x)

=

\sqrt{a}\tau_{\sqrt{\cdot},1}(x).

\]
Its population ranking is also invariant to positive rescaling, although it need not agree with the level-scale ranking.

Log1p and arcsinh do not have this simple multiplicative scale relationship. Their rankings may change as \(a\) changes.

The empirical rank transformation is used as a negative control. Positive rescaling does not change ranks, so it is evaluated only at \(a=1\). Its sample-dependent nature distinguishes it from the four population transformations.
2. Simulation design
2.1 Objective
The simulation evaluates how outcome transformation and positive outcome rescaling affect conditional treatment-effect estimation, treatment-effect ranking, and policy evaluation under zero-inflated outcomes.

The design varies the treatment-effect mechanism, the control-group zero rate, the transformation, and the scale.
2.2 Data-generating process
Covariates are generated independently from a standard multivariate normal distribution:
\[

X_i\sim\mathcal{N}(0,I_d),

\]
with \(d=10\) in the main simulation implementation.

Treatment assignment is randomized:
\[

W_i\sim\operatorname{Bernoulli}(e),

\]
where the default propensity is \(e=0.5\).

The baseline purchase log-odds component is
\[

g(X)=0.8X_1-0.5X_2,

\]
and the baseline log-spending component is
\[

h(X)=2.5+0.5X_3.

\]
Here, \(X_1,\ldots,X_5\) denote the first five covariates in one-based mathematical indexing. The implementation uses zero-based columns `X[:, 0]` through `X[:, 4]`.

The purchase probability is
\[

p_w(X)=

\operatorname{expit}

\left[

b_0+g(X)+w\delta_p(X)

\right],

\]
and the positive spending log-mean is
\[

\mu_w(X)=h(X)+w\delta_s(X).

\]
The intercept \(b_0\) is calibrated numerically so that the control-group purchase probability averaged over a standard-normal reference population equals \(1-r\), where \(r\) is the requested control-group zero rate.

The positive spending outcome is generated as
\[

S(w)=\exp(\mu_w(X)+\epsilon),

\qquad

\epsilon\sim\mathcal{N}(0,\sigma^2).

\]
The observed outcome is
\[

Y=B S,

\qquad

B\sim\operatorname{Bernoulli}(p_w(X)).

\]
The default positive-outcome log-scale standard deviation is \(\sigma=0.8\).
2.3 Treatment-effect scenarios
The treatment changes purchase log-odds by \(\delta_p(X)\) and positive log-spending by \(\delta_s(X)\).

| Scenario | Purchase log-odds shift \(\delta_p(X)\) | Log-spending shift \(\delta_s(X)\) |

|---|---:|---:|

| Null | \(0\) | \(0\) |

| Homogeneous | \(0.4\) | \(0.15\) |

| Extensive | \(0.3+0.8X_4\) | \(0\) |

| Intensive | \(0\) | \(0.2+0.4X_5\) |

| Aligned | \(0.3+0.8X_4\) | \(0.2+0.4X_4\) |

| Opposed | \(0.3+0.8X_4\) | \(0.2-0.4X_4\) |

The scenarios distinguish several mechanisms:

- **Null:** no treatment effect on either component.

- **Homogeneous:** constant shifts to purchase log-odds and positive log-spending.

- **Extensive:** heterogeneous effects on purchase probability only.

- **Intensive:** heterogeneous effects on positive spending only.

- **Aligned:** purchase and positive-spending shifts vary in the same direction.

- **Opposed:** purchase and positive-spending shifts move in opposing directions.
2.4 Zero-rate stress grid
The control-group zero-rate targets are
\[

r\in\{0.70,0.90,0.98\}.

\]
The intercept calibration targets the control purchase probability on the standard-normal reference population. The treatment-group zero rate is allowed to differ from the control-group rate because treatment can affect purchase probability.

The initial pilot grid used \(n=5{,}000\), \(\sigma=0.8\), \(e=0.5\), and a single seed across the six scenarios and three zero-rate settings. It produced 180 rows for each zero-rate setting and 540 rows overall. These pilot results are execution checks and should not be interpreted as a multi-seed robustness analysis.

The later oracle analysis and sanity audit are reported separately in their corresponding project reports.
2.5 Transformations and scales
The primary transformations are
\[

m(y)\in

\left\{

y,\log(1+y),\operatorname{arcsinh}(y),\sqrt{y}

\right\}.

\]
The positive scale grid is
\[

a\in

\{0.001,0.01,0.1,1,10,100,1000\}.

\]
The empirical rank transformation is also included:
\[

m_{\mathrm{rank}}(y_i)

=

\frac{\operatorname{rank}(y_i)}{n},

\]
using average ranks for ties. Since positive scaling cannot change ranks, it is evaluated only at \(a=1\).
2.6 Evaluation sample
For each simulation replicate, the training sample contains \(n\) observations. A separate test sample is generated using a different seed. Its size is
\[

n_{\mathrm{test}}=\max(2000,\lfloor n/2\rfloor).

\]
The test sample is used to evaluate estimated treatment effects against the known population CATE. This separates model fitting from evaluation on the same observations.
3. Methods
3.1 Standard meta-learners
The implementation includes four standard meta-learners.

**S-learner.** A single outcome model is trained using the covariates and treatment indicator as inputs. Treatment effects are predicted by evaluating the fitted model once with treatment set to one and once with treatment set to zero:
\[

\widehat{\tau}(x)

=

\widehat{\mu}(x,1)-\widehat{\mu}(x,0).

\]
**T-learner.** Separate outcome models are fitted to the treatment and control groups:
\[

\widehat{\tau}(x)

=

\widehat{\mu}_1(x)-\widehat{\mu}_0(x).

\]
**X-learner.** The X-learner first estimates the two outcome functions, constructs imputed treatment effects in each treatment arm, fits separate models to those imputed effects, and combines their predictions using the treatment propensity.

**DR-learner.** The DR-learner estimates the two outcome functions and constructs the doubly robust pseudo-outcome
\[

\widehat{\phi}_i

=

\widehat{\mu}_1(X_i)-\widehat{\mu}_0(X_i)

\+

\frac{W_i(Y_i-\widehat{\mu}_1(X_i))}{e}

\-

\frac{(1-W_i)(Y_i-\widehat{\mu}_0(X_i))}{1-e}.

\]
A final regression model is fitted to this pseudo-outcome.

The current implementation uses a constant known propensity \(e\). The DR-learner implementation does not use cross-fitting; results should therefore not be described as cross-fitted estimates.
3.2 Transformed-outcome learner
For each transformation \(m\) and scale \(a\), the transformed-outcome learner applies
\[

Y_i^\*=m(aY_i)

\]
and constructs the pseudo-outcome
\[

\widetilde{Y}_i

=

\frac{W_i-e}{e(1-e)}Y_i^\*.

\]
A regression model is fitted to predict the conditional expectation of this pseudo-outcome given \(X\). Under randomized assignment with known propensity, its conditional expectation targets the transformed-scale treatment contrast:
\[

\mathbb{E}[\widetilde{Y}\mid X=x]

=

\tau_{m,a}(x).

\]
Each transformation-scale configuration is fitted separately.
3.3 Outcome models
The default regression model is LightGBM with an L2 regression objective. The standard parameter configuration uses 200 estimators, learning rate 0.05, 15 leaves, and 50 minimum child samples.

The implementation also includes:

- **Tweedie:** a LightGBM Tweedie objective, with the default power set to 1.5.

- **Hurdle:** a classifier for positive outcomes combined with a Gamma regression model for positive outcome magnitude.

- **Negative Binomial:** a Statsmodels GLM implementation for integer-valued count outcomes.

Tweedie and Hurdle are supported as alternative outcome models for the level-scale T-learner. The transformed-outcome experiments use L2. The S-, X-, and DR-learners use L2 in the main runner because their intermediate or pseudo-outcomes can be negative.

The main simulated outcome is continuous and zero-inflated lognormal. The Negative Binomial implementation is not a primary model comparison in this DGP and should not be interpreted as evidence from a dedicated integer-count simulation.
3.4 Hyperparameter tuning
The tuning experiment uses a training, validation, and test split of 60%, 20%, and 20%, respectively. Candidate configurations are evaluated on the validation split, and the test split is reserved for final evaluation.

The candidate search includes:

- Number of estimators: 100, 200, 400.

- Learning rate: 0.03, 0.05, 0.10.

- Number of leaves: 7, 15, 31.

- Maximum depth: 3, 5, 7, or unrestricted.

- Minimum child samples: 20, 50, 100.

Tweedie additionally searches over powers 1.1, 1.3, 1.5, and 1.7.

Each model receives the same candidate budget. L2 and Hurdle are selected using validation RMSE, whereas Tweedie is selected using mean Tweedie deviance. These model-specific validation metrics are not directly comparable.

The documented validation run used the opposed scenario, \(n=5{,}000\), control zero-rate target 0.90, \(\sigma=0.8\), \(e=0.5\), seed 42, and 12 candidates per model. All 36 candidate evaluations completed successfully. This is a focused validation run, not a comprehensive comparison over all scenarios and seeds.
3.5 Evaluation metrics
All ranking and policy metrics are calculated on the independent test sample.

**Spearman rank correlation**
\[

\rho

=

\operatorname{Spearman}

(\widehat{\tau}(X),\tau_{\mathrm{truth}}(X)).

\]
This measures agreement between estimated and true treatment-effect rankings.

**Top-20% overlap**

Let \(A\) and \(B\) denote the estimated and true top-20% selections. The overlap is the intersection-over-union:
\[

\operatorname{IoU}(A,B)

=

\frac{|A\cap B|}{|A\cup B|}.

\]
The selected group size is determined by rounding the requested fraction of the sample size, with at least one observation selected.

**Policy value**

The estimated policy selects the top 20% of individuals according to predicted treatment effects. Its simulated policy value is
\[

V(\widehat{\pi})

=

\frac{1}{|\widehat{S}|}

\sum_{i\in\widehat{S}}

\tau_{\mathrm{level}}(X_i),

\]
where \(\widehat{S}\) is the selected group. This is an oracle evaluation using known simulated level-scale CATEs, not an outcome-based estimate from observed data.

**Regret**
\[

\operatorname{Regret}(\widehat{\pi})

=

V(\pi^\*)-V(\widehat{\pi}),

\]
where \(\pi^\*\) selects the true top 20% according to level-scale CATE.

**Scale-ranking stability**

For a fixed transformation \(m\), the stability metric compares the estimated rankings at scale \(a\) and the reference scale \(a=1\):
\[

\rho_{a,a=1}

=

\operatorname{Spearman}

\left(

\widehat{\tau}_{m,a},

\widehat{\tau}_{m,1}

\right).

\]
This measures stability of the estimated rankings across scales, rather than agreement with the true CATE.

For the null scenario, the true CATE is constant, so rank correlation and top-20% overlap against a unique true ranking are undefined. These metrics are reported as missing rather than assigned arbitrary values.
3.6 Ranking truth for transformed outcomes
For identity, log1p, arcsinh, and square-root transformations, the population transformed CATE is calculated using the known zero-inflated lognormal DGP. The implementation uses 40-point Gauss-Hermite quadrature for the positive lognormal component.

For the empirical rank transformation, no exact population transformed-CATE implementation is used. Its estimated ranking is therefore compared with the original level-scale CATE ranking as a negative-control comparison. This distinction should be retained when interpreting its ranking metrics.
3.7 Back-transformation baseline
The back-transformation baseline fits separate models to log1p-transformed outcomes in the two treatment groups and predicts
\[

\widehat{\tau}_{\mathrm{back}}(x)

=

\operatorname{expm1}(\widehat{\mu}_{1,\log}(x))

\-

\operatorname{expm1}(\widehat{\mu}_{0,\log}(x)).

\]
This is a naive inverse-transformation baseline. In general, exponentiating a predicted conditional mean on the log scale does not recover the conditional mean on the original scale because of the nonlinear inverse transformation and Jensen's inequality. Its results should be interpreted as a practical baseline rather than an exact conditional-mean correction.
4. Table 1. Model cards
| Model | Role | Target or construction | Outcome model | Important implementation note |

|---|---|---|---|---|

| S-learner | Standard meta-learner | Treatment indicator included as a feature; contrast predictions at \(W=1\) and \(W=0\) | L2 | One shared outcome model |

| T-learner | Standard meta-learner | Difference between treatment- and control-specific predictions | L2 by default; Tweedie and Hurdle optional | Separate models by arm |

| X-learner | Standard meta-learner | Imputed treatment effects fitted in each arm and propensity-weighted combination | L2 | Uses constant known propensity |

| DR-learner | Standard meta-learner | Doubly robust pseudo-outcome followed by final regression | L2 | No cross-fitting in current implementation |

| Transformed-outcome learner | Main transformation comparison | Regression on \((W-e)m(aY)/(e(1-e))\) | L2 | Separate fit for each transformation and scale |

| BackTransform | Ablation baseline | Difference between inverse-transformed log1p arm predictions | L2 | Naive inverse transformation; conditional-mean bias is possible |

| Tweedie T-learner | Alternative outcome model | Separate arm-specific outcome regressions | LightGBM Tweedie | Compared using Tweedie deviance during tuning |

| Hurdle T-learner | Alternative outcome model | Positive-outcome probability multiplied by predicted positive mean | Classifier + Gamma model | Explicitly models zero occurrence and positive magnitude |

| Negative Binomial | Implemented model option | Count-outcome regression | Statsmodels GLM | Requires integer-valued outcomes; not a primary comparison in the continuous DGP |
5. Related work
5.1 Causal inference with nonstandard and zero-inflated outcomes
Treatment-effect estimation becomes more complicated when outcomes contain a substantial mass at zero or have strongly skewed positive values. In such settings, a single conditional mean can combine distinct mechanisms, such as the probability of a positive outcome and the magnitude of the outcome conditional on being positive.

Work on causal effects with nonstandard outcomes motivates careful definition of the target estimand and attention to how transformations affect interpretation. In particular, transformed-outcome contrasts need not represent the same decision target as contrasts on the original outcome scale.

Chen and Roth (2024) and Mullahy and Norton (2024) provide relevant context for the interpretation and analysis of outcomes with substantial zero mass and nonlinear outcome structures.

Related work directly addressing zero-inflated outcomes includes Oganisian et al. (2021), who develop a Bayesian nonparametric model for prediction, clustering, and causal estimation with zero-inflated outcomes, and Kim et al. (2024), who study heterogeneous treatment effects with zero-inflated data using a Bayesian nonparametric model. These studies provide methodological context for modelling outcomes with a point mass at zero, although their modelling frameworks and objectives are not identical to the transformation-and-scale comparison considered here.
5.2 Meta-learners for heterogeneous treatment effects
Meta-learners provide general frameworks for estimating conditional treatment effects using supervised learning algorithms. Künzel et al. (2019) introduce and compare S-, T-, and X-learner approaches. Nie and Wager (2021) develop the R-learner framework, while doubly robust approaches combine outcome modelling and treatment assignment information.

These methods motivate the use of multiple learner architectures as benchmarks. Their performance can depend on the outcome structure, the quality of nuisance-function estimates, and the relationship between the chosen prediction loss and the target estimand.

Zhang et al. (2022) provide a broader survey of treatment-effect heterogeneity and uplift modelling. This survey helps situate the project among methods that estimate heterogeneous effects and use those estimates to prioritize individuals.
5.3 Uplift modelling and policy learning
Uplift modelling and treatment-effect ranking focus on identifying individuals for whom treatment is expected to produce the greatest incremental benefit. Gutierrez and Gérardy (2017) survey uplift modelling approaches, while policy-learning work such as Athey and Wager (2021) studies the construction and evaluation of treatment policies.

Revenue- and profit-oriented uplift applications also motivate the distinction between effects on participation or conversion and effects on the value of a positive outcome. Relevant application-oriented work includes Gubela et al. (2020), Baier and Stöcker (2022), and He et al. (2024).
5.4 Research question and scope
The present project studies a focused question: when outcomes contain a large zero mass and a skewed positive component, how do nonlinear outcome transformations and positive rescaling affect treatment-effect rankings relative to the original level-scale CATE?

The simulation separates changes in purchase probability from changes in positive spending. It then evaluates transformation-scale-specific population CATE rankings, estimated ranking performance, and policy value on the original outcome scale.

This is a targeted investigation of transformation and scale sensitivity. The results should not be framed as establishing broad novelty over all work on zero-inflated causal outcomes, uplift modelling, or policy learning.
Appendix A. Full data-generating process
A.1 Covariates and treatment assignment
For each unit \(i\),
\[

X_i\sim\mathcal{N}(0,I_{10}),

\qquad

W_i\sim\operatorname{Bernoulli}(e).

\]
The default treatment propensity is \(e=0.5\).
A.2 Baseline functions
Define
\[

g(X)=0.8X_1-0.5X_2,

\]
\[

h(X)=2.5+0.5X_3.

\]
The control purchase probability is
\[

p_0(X)=\operatorname{expit}(b_0+g(X)).

\]
The intercept \(b_0\) is calibrated numerically so that
\[

\mathbb{E}_{X\sim\mathcal{N}(0,I_{10})}

[p_0(X)]

=

1-r,

\]
where \(r\) is the target control zero rate.
A.3 Scenario-specific treatment shifts
Let \(Z_3=X_4\) and \(Z_4=X_5\) in one-based mathematical indexing. The scenario-specific shifts are:
\[

(\delta_p(X),\delta_s(X))

=

\begin{cases}

(0,0), & \text{null},\\\\

(0.4,0.15), & \text{homogeneous},\\\\

(0.3+0.8Z_3,0), & \text{extensive},\\\\

(0,0.2+0.4Z_4), & \text{intensive},\\\\

(0.3+0.8Z_3,0.2+0.4Z_3), & \text{aligned},\\\\

(0.3+0.8Z_3,0.2-0.4Z_3), & \text{opposed}.

\end{cases}

\]
A.4 Potential-outcome components
For treatment \(w\in\{0,1\}\),
\[

p_w(X)

=

\operatorname{expit}

\left[

b_0+g(X)+w\delta_p(X)

\right],

\]
\[

\mu_w(X)=h(X)+w\delta_s(X).

\]
Generate the purchase indicator
\[

B(w)\mid X

\sim

\operatorname{Bernoulli}(p_w(X))

\]
and positive spending
\[

S(w)\mid X

=

\exp(\mu_w(X)+\epsilon),

\qquad

\epsilon\sim\mathcal{N}(0,\sigma^2).

\]
The potential outcome is
\[

Y(w)=B(w)S(w).

\]
The observed outcome is
\[

Y=WY(1)+(1-W)Y(0).

\]
A.5 Ground-truth level-scale CATE
Since
\[

\mathbb{E}[S(w)\mid X]

=

\exp\left(\mu_w(X)+\frac{\sigma^2}{2}\right),

\]
the conditional mean potential outcome is
\[

\mathbb{E}[Y(w)\mid X]

=

p_w(X)

\exp\left(\mu_w(X)+\frac{\sigma^2}{2}\right).

\]
Therefore,
\[

\boxed{

\tau_{\mathrm{level}}(X)

=

\exp\left(\frac{\sigma^2}{2}\right)

\left[

p_1(X)e^{\mu_1(X)}

\-

p_0(X)e^{\mu_0(X)}

\right].

}

\]
A.6 Ground-truth transformed CATE
For a transformation \(m\) and scale \(a>0\),
\[

\tau_{m,a}(X)

=

p_1(X)

\mathbb{E}

\left[

m\left(a\exp(\mu_1(X)+\epsilon)\right)

\mid X

\right]

\-

p_0(X)

\mathbb{E}

\left[

m\left(a\exp(\mu_0(X)+\epsilon)\right)

\mid X

\right].

\]
The implementation evaluates these expectations numerically using 40-point Gauss-Hermite quadrature.

For the identity transformation, this expression reduces to the level-scale CATE multiplied by \(a\).
A.7 Implementation and reproducibility notes
The simulation implementation uses fixed random seeds for reproducibility. The calibration of \(b_0\) uses a standard-normal reference population. When the same DGP is applied to an externally supplied covariate matrix, such as the semi-synthetic Criteo covariates, the calibrated target zero rate is not guaranteed to equal the realized zero rate in that particular covariate population.

The simulation runner generates an independent test sample for each replicate and evaluates learned treatment-effect functions against the known CATE. The reported policy value and regret therefore use simulation ground truth.

The complete implementation and experiment-specific results are maintained in the project source files and accompanying reports.