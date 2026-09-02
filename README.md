# Panel MS-AR(1) around a common log-linear trend

Joint MLE of one model on an unbalanced country panel. Shared Markov parameters; country-specific latent regimes. Optional country intercepts $a_i$ and/or slopes $g_i$. `y` can be any series; logs are the intended scale.

## Model

$$
\begin{aligned}
y_{it} &= a_i + g_i\, t + z_{it}, \\
z_{i,t+1} &= \bigl(1-\rho(s_{it})\bigr)\mu(s_{it}) + \rho(s_{it})\, z_{it} + \sigma(s_{it})\,\varepsilon_{it}, \\
s_{i,t+1} &\sim \Pi(\,\cdot\mid s_{it}).
\end{aligned}
$$

$a_i$ and $g_i$ may be common ($a$, $g$) or country-specific, independently.

Countries are independent given shared $\theta$. Likelihood: Hamilton filter per country, sum log-likelihoods.

| Piece | Specification |
|---|---|
| Outcome $y$ | Any series; ideally in logs. |
| Trend | Common $g$ (`country_trends=False`) or country-specific $g_i$ (`True`). $g$ is per unit of calendar time; $\rho$ is per observation. |
| Time $t$ | Calendar time, **common origin** for every country. Any regular frequency. |
| Country intercepts | Off by default (`country_intercepts=False`). If off, permanent level differences load on the cycle / regimes. |
| Regimes $k$ | Odd ($1, 3, 5, \ldots$) so a unique median regime exists. $k$ is specified, not selected. |
| Mean restriction | After estimation, regimes are ordered by $\mu$ and shifted so $\mu_{\lfloor k/2\rfloor}=0$. The shift is absorbed into $a$ or the $a_i$. Set `zero_mu=True` to restrict **every** $\mu(s)=0$; regimes are then ordered by $\sigma$ (or $\rho$ if $\sigma$ is common). |
| Persistence | One $\rho$ for all regimes (`common_rho=True`). |
| Variance | $\sigma(s)$ switches with the regime (`common_sigma=False`). |
| Transitions | Common $\Pi$. Latent path $s_{it}$ is country-specific. |
| Timing | Regime dated $t$ governs the transition from $z_t$ to $z_{t+1}$; then a new regime is drawn. |
| Initial condition | $z_1 \mid s_1 \sim N\bigl(\mu_s,\, \sigma_s^2/(1-\rho_s^2)\bigr)$, with $s_1$ from the ergodic distribution of $\Pi$. |
| Sample | Unbalanced panel. Drop countries with fewer than `min_t` **observations**. Calendar gaps: keep the longest contiguous spell (no interpolation). |

$a$ (or the $a_i$) and the regime means are collinear without the median-mean pin. By default, country intercepts and slopes are profiled out of the joint likelihood so they are not in the outer parameter vector. Set `two_step=True` or `two_step="quadratic"` to OLS-detrend first with a quadratic in calendar time, $a + g t + h t^2$ (country-specific $a_i$, $g_i$, $h_i$ follow `country_intercepts` / `country_trends`; $h$ is country-specific iff slopes are) and then run the joint MS-AR on residuals. $a$, $g$, and $h$ are not re-estimated in the second step. Set `two_step="cf"` for a country-specific Christiano–Fitzgerald **low-pass** trend: the cycle keeps oscillations from 2 observations up to `cf_cutoff` years (default **15 years**, 60 quarters on quarterly data — a smooth growth-trend cutoff, not the 6–32 business-cycle band); `drift=True` as in CF for I(1) logs. Two-step is not a single joint MLE of the trend and the cycle. After the median-$\mu$ pin, the shift is added to $a$ (quadratic) or subtracted from the CF cycle.

## Estimator

`panel_msar.py` (`PanelMSAR`). Multi-start L-BFGS-B on unconstrained parameters: row-wise softmax logits for `Pi` (`k(k-1)` free); `rho = tanh`; `sigma = exp`; free means are every `mu[s]` except `s = k//2`, or none if `zero_mu=True`. After each successful fit, regimes are ordered by `mu` and the median mean is shifted into `a` (or ordered by `sigma` if all means are zero). Numba Hamilton filter (install `numba`). Hessian SEs are on the unconstrained vector, then delta-method to the table; for a paper, bootstrap countries.

`demo_panel_msar.py` simulates a 3-regime DGP and recovers parameters.

`demo_oecd_msar.py` fits the same model on a demo OECD quarterly panel of log real GDP per worker (`data/demo_oecd_gdp_per_worker_q.csv`). Refresh the panel with `python data/fetch_demo_oecd.py`.

## Data and API

Pass `country`, `time`, `y` (ideally in logs). `time` must share one calendar origin, not periods-since-entry.

- Year-fraction, e.g. quarterly `1970.0, 1970.25, …`: `g` per year, `rho` per quarter.
- Datetimes / pandas `Period`s: converted to `year + (month-1)/12`.
- Integer period index (Stata `%tq`): left as-is; `g` per period.
- Do not pass year.quarter codes (`1970.1, 1970.2, 1970.3, 1970.4`).

Internally `t` is shifted so the earliest sample date is 0 (`res.time_base`, `res.time_step`).

```python
from panel_msar import PanelMSAR

mod = PanelMSAR(
    n_regimes=3, common_rho=True, common_sigma=False,
    country_intercepts=False, country_trends=False,
    two_step=False, cf_cutoff=15, zero_mu=False, min_t=12,
)
res = mod.fit(
    df["country"], df["time"], df["y"],
    n_starts=8, maxiter=400, detrend_pdf="cycle.pdf",
)
print(res)                 # g, rho, regime table, Pi; SEs in parentheses underneath
res.params                 # a, g (scalar or dict), h if quadratic two-step, rho, mu, sigma, P
res.filtered_probs[cid]    # time, cycle, p_regime0/1/2
res.plot_detrended("cycle.pdf")
```

Dependencies: `numpy`, `pandas`, `scipy`, `numba`, `matplotlib`. `statsmodels` is required only for `two_step="cf"`.
