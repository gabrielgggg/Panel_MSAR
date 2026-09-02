# Panel MS-AR(1) around a common log-linear trend

Joint MLE of one model on an unbalanced country panel of log labor productivity. Shared parameters; country-specific latent regimes. No country intercepts.

## Model

```
y_it = a + g * t + z_it

z_{i,t+1} = (1 - rho(s_it)) * mu(s_it) + rho(s_it) * z_it
            + sigma(s_it) * eps_it

s_{i,t+1} ~ Pi( . | s_it )
```

Countries are independent given shared `theta`. Likelihood: Hamilton filter per country, sum log-likelihoods.

| Piece | Specification |
|---|---|
| Outcome `y` | Log labor productivity. |
| Trend | Common intercept `a` and common slope `g`. |
| Time `t` | Calendar time, **common origin** for every country. Any regular frequency. `g` is per unit of this scale; `rho` is per observation. |
| Country intercepts | None. Permanent level differences load on the cycle / regimes. |
| Regimes `k` | Odd (`1, 3, 5, …`) so a unique median regime exists. `k` is specified, not selected. |
| Mean restriction | After estimation, regimes are ordered by `mu` and shifted so `mu[k//2] = 0`. The shift is absorbed into `a`. |
| Persistence | One `rho` for all regimes (`common_rho=True`). |
| Variance | `sigma(s)` switches with the regime (`switch_sigma=True`). |
| Transitions | Common `Pi`. Latent path `s_it` is country-specific. |
| Timing | Regime dated `t` governs the transition from `z_t` to `z_{t+1}`; then a new regime is drawn. |
| Initial condition | `z_1 \| s_1 ~ N(mu_s, sigma_s^2 / (1-rho_s^2))`, `s_1` from the ergodic distribution of `Pi`. |
| Sample | Unbalanced panel. Drop countries with fewer than `min_t` **observations**. Calendar gaps: keep the longest contiguous spell (no interpolation). |

`a` and the regime means are collinear without the median-mean pin. Without country FE, a country that stays below the common path looks like the low-`mu` regime; that is the model.

## Estimator

`panel_msar.py` (`PanelMSAR`). Multi-start L-BFGS-B on unconstrained parameters: row-wise softmax logits for `Pi` (`k(k-1)` free); `rho = tanh`; `sigma = exp`; free means are every `mu[s]` except `s = k//2`. After each successful fit, regimes are ordered by `mu` and the median mean is shifted into `a`. Numba Hamilton filter (install `numba`). Hessian SEs are on the unconstrained vector, then delta-method to the table; for a paper, bootstrap countries.

`demo_panel_msar.py` simulates a 3-regime DGP and recovers parameters.

## Data and API

Pass `country`, `time`, `y` (log productivity). `time` must share one calendar origin, not periods-since-entry.

- Year-fraction, e.g. quarterly `1970.0, 1970.25, …`: `g` per year, `rho` per quarter.
- Datetimes / pandas `Period`s: converted to `year + (month-1)/12`.
- Integer period index (Stata `%tq`): left as-is; `g` per period.
- Do not pass year.quarter codes (`1970.1, 1970.2, 1970.3, 1970.4`).

Internally `t` is shifted so the earliest sample date is 0 (`res.time_base`, `res.time_step`).

```python
from panel_msar import PanelMSAR

mod = PanelMSAR(n_regimes=3, common_rho=True, switch_sigma=True, min_t=12)
res = mod.fit(df["country"], df["time"], df["y"], n_starts=8, maxiter=400)
print(res)                 # regime table + Pi; SEs in parentheses underneath
res.params                 # rho, mu, sigma, P; a and g are estimated but nuisance
res.filtered_probs[cid]    # time, cycle, p_regime0/1/2
```

Dependencies: `numpy`, `pandas`, `scipy`, `numba`.
