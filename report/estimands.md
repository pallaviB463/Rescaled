# P1 — Estimands and Outcome-Scale Limits

## Estimands

Let \(Y(1),Y(0)\geq0\) be potential outcomes and \(X=x\) the covariates. The level-scale conditional average treatment effect (CATE) is

\[
\tau_{\mathrm{level}}(x)
=\mathbb E[Y(1)-Y(0)\mid X=x].
\]

For a transformation \(m\) satisfying \(m(0)=0\) and a positive scale \(a\), define

\[
\tau_{m,a}(x)
=\mathbb E[m(aY(1))-m(aY(0))\mid X=x].
\]

In general, \(\tau_{m,a}(x)\ne m(a\tau_{\mathrm{level}}(x))\): transformation and expectation do not commute. Consequently, transformed-scale treatment effects may rank individuals differently from level-scale effects.

## Small-scale limit

For differentiable transformations with \(m(0)=0\), \(m'(0)=1\), and suitable integrability conditions,

\[
\lim_{a\to0^+}\frac{\tau_{m,a}(x)}{a}
=\tau_{\mathrm{level}}(x).
\]

Thus log1p and arcsinh approach a scaled version of the level estimand as \(a\) becomes small. Identity satisfies this relationship exactly at every positive scale.

## Large-scale behavior under the project DGP

The project uses zero-inflated lognormal spend: purchase probability \(p_w(x)\), log-spend mean \(\mu_w(x)\), and log-spend noise standard deviation \(\sigma\). For log1p and arcsinh, under this DGP,

\[
\tau_{m,a}(x)
=(p_1-p_0)\log a+C_m(x)+o(1),
\qquad a\to\infty,
\]

where

\[
C_{\log1p}=p_1\mu_1-p_0\mu_0,
\]

and

\[
C_{\operatorname{arcsinh}}
=(p_1-p_0)\log 2+p_1\mu_1-p_0\mu_0.
\]

These limits depend on the DGP and are not universal claims for arbitrary outcome distributions.

## Scale invariance and ranking

- **Identity:** \(\tau_{m,a}=a\tau_{\mathrm{level}}\); rankings are invariant to positive \(a\).
- **Square root:** \(\tau_{m,a}=\sqrt a\,\tau_{m,1}\); rankings are invariant to positive \(a\), but need not match level-scale rankings.
- **Log1p and arcsinh:** rankings can change with \(a\), motivating the oracle sensitivity analysis.

Estimand limits alone do not establish ranking convergence. The null scenario has \(\tau_{\mathrm{level}}(x)=0\) for all \(x\), so rank correlation against the level-scale truth is undefined and must be handled separately.
