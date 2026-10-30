# Estimands and Outcome-Scale Limits

## Estimands

Let \(Y(1),Y(0)\geq 0\) be potential outcomes and \(X=x\) the covariates. The level-scale conditional average treatment effect (CATE) is

\[
\tau_{\mathrm{level}}(x)
=
\mathbb{E}[Y(1)-Y(0)\mid X=x].
\]

For a transformation \(m\) satisfying \(m(0)=0\) and a positive scale \(a\), define

\[
\tau_{m,a}(x)
=
\mathbb{E}[m(aY(1))-m(aY(0))\mid X=x].
\]

In general,

\[
\tau_{m,a}(x)\neq m(a\tau_{\mathrm{level}}(x)):
\]

transformation and expectation do not commute. Consequently, transformed-scale treatment effects may rank individuals differently from level-scale effects.

## Small-scale limit

For differentiable transformations with \(m(0)=0\), \(m'(0)=1\), and suitable integrability conditions,

\[
\lim_{a\to 0^+}\frac{\tau_{m,a}(x)}{a}
=
\tau_{\mathrm{level}}(x).
\]

Thus, log1p and arcsinh approach a scaled version of the level estimand as \(a\) becomes small. Identity satisfies this relationship exactly at every positive scale.

## Large-scale behavior under the project DGP

The project uses zero-inflated lognormal spend: purchase probability \(p_w(x)\), log-spend mean \(\mu_w(x)\), and log-spend noise standard deviation \(\sigma\). For log1p and arcsinh, under this DGP,

\[
\tau_{m,a}(x)
=
(p_1-p_0)\log a+C_m(x)+o(1),
\qquad a\to\infty,
\]

where

\[
C_{\log1p}=p_1\mu_1-p_0\mu_0,
\]

and

\[
C_{\operatorname{arcsinh}}
=
(p_1-p_0)\log 2+p_1\mu_1-p_0\mu_0.
\]

These limits depend on the DGP and are not universal claims for arbitrary outcome distributions.

### Distributional assumptions

For the large-scale limits, conditional on a positive outcome in treatment arm \(w\), assume

\[
\log Y(w)\mid Y(w)>0,X=x
\sim \mathcal{N}(\mu_w(x),\sigma^2).
\]

The positive-outcome distribution is therefore lognormal, with finite conditional mean

\[
s_w(x)
=
\exp\left(\mu_w(x)+\frac{\sigma^2}{2}\right).
\]

For positive outcomes, as \(a\to\infty\),

\[
\log(1+aY)=\log a+\log Y+o(1),
\]

and

\[
\operatorname{arcsinh}(aY)
=
\log a+\log 2+\log Y+o(1).
\]

These expansions explain the leading logarithmic term and the transformation-specific constants in the large-scale CATE expressions.

### Conditions for the small-scale limit

The small-scale result additionally requires finite conditional first moments and an appropriate domination condition that permits the limit to pass through the conditional expectations. It applies to differentiable transformations with \(m(0)=0\) and \(m'(0)=1\), such as log1p and arcsinh. It is not directly applicable to the square-root or rank transformations.

## Decomposition into purchase probability and conditional spend size

Define the purchase probability and conditional positive spend for treatment arm \(w\) as

\[
p_w(x)=P(Y(w)>0\mid X=x),
\]

\[
s_w(x)=E[Y(w)\mid Y(w)>0,X=x].
\]

Because the outcome is nonnegative and has a point mass at zero,

\[
E[Y(w)\mid X=x]=p_w(x)s_w(x).
\]

The level-scale CATE can therefore be expressed as

\[
\tau_{\mathrm{level}}(x)
=
p_1(x)s_1(x)-p_0(x)s_0(x).
\]

Adding and subtracting \(p_1(x)s_0(x)\) gives

\[
\boxed{
\begin{aligned}
\tau_{\mathrm{level}}(x)
={}&[p_1(x)-p_0(x)]s_0(x)\\
&+p_1(x)[s_1(x)-s_0(x)].
\end{aligned}
}
\]

This decomposition separates the effect associated with changes in purchase probability from the effect associated with changes in spending conditional on purchase. It uses the control-arm conditional spend as the reference for the first component and the treated-arm purchase probability for the second component.

## Scale invariance and ranking

- **Identity:** \(\tau_{m,a}=a\tau_{\mathrm{level}}\); rankings are invariant to positive \(a\).
- **Square root:** \(\tau_{m,a}=\sqrt{a}\,\tau_{m,1}\); rankings are invariant to positive \(a\), but need not match level-scale rankings.
- **Log1p and arcsinh:** rankings can change with \(a\), motivating the oracle sensitivity analysis.

Estimand limits alone do not establish ranking convergence. The null scenario has \(\tau_{\mathrm{level}}(x)=0\) for all \(x\), so rank correlation against the level-scale truth is undefined and must be handled separately.