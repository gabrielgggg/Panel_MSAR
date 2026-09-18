# Plan: common-λ catch-up around a global trend

Status: implemented in `panel_msar.py` (`convergence=True`). First empirical run is the third spec in `data_2026.09.17/run_msar_re.py`. This is the **common λ** case only (no country-specific λ_i). Do not start the heterogeneous-λ (2D GH) extension until this version is estimated and plotted.

## Target specification

\[
\begin{aligned}
y_{it} &= \bar a + g t + b_i\,\lambda^{t-T_{i0}} + z_{it}, \\
z_{i,t+1} &= \bigl(1-\rho(s_{it})\bigr)\mu(s_{it})+\rho(s_{it})z_{it}+\sigma(s_{it})\varepsilon_{it}, \\
s_{i,t+1} &\sim \Pi(\cdot\mid s_{it}), \\
b_i &\sim N(0,\omega_b^2),\qquad 0<\lambda<1.
\end{aligned}
\]

- \(\bar a + g t\): common long-run path (calendar time).
- \(b_i\lambda^{t-T_{i0}}\): deterministic catch-up. \(T_{i0}\) is country \(i\)'s **entry date** (first observation). \(b_i\) is the gap at entry.
- \(z_{it}\): MS-AR(1) cycle, unchanged (3 regimes, switching ρ and σ, median μ pinned at 0).
- \(E[b_i]=0\) is a hard pin so \(\bar a\) is the long-run intercept.

**Entry date.** After `_prepare`, panel time is already shifted so the earliest date in the *whole* panel is 0. \(T_{i0}\) is that country's first internal time (0 only for countries present at the global start). The exponential uses \(t-T_{i0}\), not global \(t\). Late joiners would otherwise be evaluated at \(t\approx 50\) and \(b_i\) would be a fictional pre-sample gap. The slope \(g t\) stays global calendar time (internal \(t\)).

**Nest.** \(\lambda=1\) recovers the current RE intercept model: \(y_{it}=\bar a + b_i + g t + z_{it}\) with \(a_i=\bar a+b_i\). The nest sits on the boundary of \((0,1)\). Compare estimates and cycle plots; do not treat a LR statistic as \(\chi^2_1\).

## Flag and compatibility

New constructor flag: `convergence=False`.

When `convergence=True`:

- Force `random_intercepts=True`.
- Common \(g\); no extra seasonal terms.

Default `convergence=False` leaves every existing report unchanged.

## Unconstrained parameters

Add to the outer θ (after `g`, alongside current RE names):

| Name | Map | Role |
|---|---|---|
| `a_bar` | identity | long-run common intercept \(\bar a\) |
| `log_omega` | \(\omega_b=\exp(\cdot)\) | sd of entry gaps (same slot as current `omega`) |
| `logit_lambda` | \(\lambda=1/(1+e^{-u})\) | common catch-up factor |

Drop the current RE mean `alpha` as a *mean of intercepts*. Under this spec the GH mean of \(b_i\) is **zero**; the location is entirely \(\bar a\).

`param_names` when `convergence`:

`[P slots, rho, mu, sigma, g, a_bar, log_omega, logit_lambda]`

Mu-pin shift (ordering regimes) is absorbed into \(\bar a\), as it is now absorbed into `a` / `alpha`.

## Likelihood (still 1D GH)

Packed data already has per-country slices. Entry time is `tcat[offsets[i]]` (first observation of that country, time already shifted).

For GH node \(\ell\), \(b^{(\ell)}=\omega_b z_\ell^{\mathrm{GH}}\) (mean 0):

\[
z_{it}^{(\ell)} = y_{it} - \bar a - g t_{it} - b^{(\ell)}\,\lambda^{t-T_{i0}}.
\]

Then the usual Hamilton-filter country ll, weighted by GH weights, `logsumexp` as in `_country_ll_re`.

Implement by generalizing `_trend` (or a sibling `_mean_path`) to

```text
abar + g*t + b * lambda**(t - T_i0)
```

with `b=0` when `convergence` is off (current intercept-in-`a` path). When `convergence` is off, keep today’s `a + g t` so old results do not change.

Do **not** put \(\lambda^{t-T_{i0}}\) inside Numba unless it is an easy extra argument; computing the mean path in Python and passing `z` into `_country_ll_nb` is enough (same as now).

## Posterior \(b_i\) and cycles

Reuse the GH softmax in `_re_posterior_a`, but the node values are \(b^{(k)}\) not intercepts. Store:

- `params["a_bar"]`, `params["lambda"]`, `params["omega"]`
- `a_out[i] = a_bar + b_hat[i]` only as a convenience “entry intercept”; also store `b_hat[i]`
- cycles: \(z_{it}=y_{it}-\bar a-g t-b_i\lambda^{t-T_{i0}}\)

Plot the **cycle** as now, and optionally the deterministic path \(\bar a+gt+b_i\lambda^{t-T_{i0}}\) (not required for the first report).

## Starting values

From current OLS intercepts \(a_i^{\mathrm{OLS}}\) (or country means of \(y-g_0 t\)):

- \(\bar a_0 = \mathrm{mean}(a_i^{\mathrm{OLS}})\)
- \(\omega_{b0} = \mathrm{std}(a_i^{\mathrm{OLS}})\)
- \(\lambda_0 = 0.98\) (logit \(\approx 3.89\)), plus a start at \(0.95\) and one near \(0.999\) (almost the old RE)

Keep the existing MS-AR multi-start skeleton (`n_starts=7` in empirical runners, threaded). Reset any inner cache each start, as now.

If a start wants \(\lambda\to 1\), logit \(\to+\infty\); clip logit to something like \([-20, 20]\) (\(\lambda\in(2\times10^{-9},1-2\times10^{-9})\)) so the optimizer does not overflow \(\lambda^{t-T_{i0}}\).

## SEs, summary, TeX

- Hessian on unconstrained θ, including `logit_lambda`.
- Report \(\lambda\) with delta-method SE: \(\mathrm{se}(\lambda)=\mathrm{se}(u)\cdot\lambda(1-\lambda)\).
- `summary` / `_trend_note`: \(\bar a\), \(g\), \(\lambda\), \(\omega_b\).
- Warn if \(\hat\lambda>0.995\) (“indistinguishable from RE intercepts”) or if \(\hat\lambda<0.5\) (“near-immediate jump to the common path”).

## First empirical run

After the code path is in `panel_msar.py` and `_trend_note`:

- Data: `data_2026.09.17` (SA, `realGDPsa_usd_pa_empl`), full sample, `rho_max=0.99`, 3 starts.
- Two or three columns in one PDF:
  1. Current Normal RE intercepts (baseline, already running / already in that folder).
  2. This spec (common λ).
- Output only under `data_2026.09.17/` (runner + tex/pdf + cycle figs). Do not overwrite `msar_re.pdf` if that file is the no-catch-up baseline; use e.g. `msar_re_lambda.pdf`.

Compare: log-likelihood vs RE intercepts; \(\hat\lambda\); whether quiet-regime \(\rho\) falls; whether cycle plots lose the slow one-sided drifts.

## Out of scope for this pass

- Random \(\lambda_i\) (2D GH).
- Permanent \(a_i\) *plus* \(b_i\lambda^{t-T_{i0}}\).
- Permanent FE intercepts or extra seasonal terms.
- Treating \(\lambda=1\) as an interior H0 for a standard LR test.
- Changing `_prepare` time origin or `min_t`.

## Implementation order

1. Flag, pack/unpack, `param_names`, compatibility errors.
2. Per-country \(T_{i0}\) from packed offsets; mean-path helper.
3. Point `_country_ll_re` / `_re_posterior_a` / cycle residual at the new mean path when `convergence`.
4. Starts, mu-shift into \(\bar a\), SEs, summary, `_trend_note`.
5. Tiny smoke: 2 countries, `n_starts=1`, `λ` recovers near 1 on data generated from the old RE DGP (`simulate_panel` if usable).
6. Empirical runner on 2026-09-17, then PDF.
