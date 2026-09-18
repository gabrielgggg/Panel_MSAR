# Status (resume here)

Last updated: 2026-09-17. Repo: `https://github.com/gabrielgggg/Panel_MSAR` (`main`).

Joint panel Markov-switching AR(1) around a **common** log-linear trend for an unbalanced country panel. Outcome is intended in logs. One joint MLE, not country-by-country. Code lives in `panel_msar.py` (`PanelMSAR`). Numba Hamilton filter. Reports are LaTeX (Palatino / `mathpazo`), compiled with MiKTeX `pdflatex` from the report’s own folder (`msar_report.compile_tex`).

## Locked modeling choices

- Odd `k` (usually 3). Median \(\mu\) pinned at 0 unless `zero_mu=True`.
- Default: switching \(\sigma\). Recent empirical runs use switching \(\rho\) (`common_rho=False`, `common_sigma=False`).
- Calendar time, common origin. Internally `t` is shifted so the earliest date is 0.
- \(|\rho|<\texttt{rho\_max}\). Default `RHO_MAX=0.995`. Empirical runs on the 2026-09-17 file use `rho_max=0.99`.
- `min_t=12` observations (not years). Longest contiguous spell kept.
- Intercept: common \(a\), or Normal RE \(a_i\sim N(\alpha,\omega^2)\) via 11-point Gauss–Hermite. Slope \(g\) is always common.
- `output*.txt` estimate dumps were removed from git; they are **not** gitignored (do not add `output*.txt` to `.gitignore`).

Removed from the estimator (do not restore unless asked): country-specific trends, FE country intercepts, Pareto RE, quarter dummies, seasonal REs, two-step OLS/CF, quadratic common trend, OECD demo, SOE extract, `data_pull/`, `with_pulled_data/`, NSA vintages `data_2026.09.11/` and `data_2026.09.14/`.

## Estimator flags (current `panel_msar.py`)

| Flag | Meaning |
|---|---|
| `n_regimes` | Odd integer. Median \(\mu\) pinned at 0. |
| `common_rho` | One \(\rho\) vs \(\rho(s)\). |
| `common_sigma` | One \(\sigma\) vs \(\sigma(s)\). |
| `random_intercepts` | \(a_i\sim N(\alpha,\omega^2)\), 11-point Gauss–Hermite. |
| `zero_mu` | Every \(\mu(s)=0\). |
| `min_t` | Minimum observations per country. |
| `rho_max` | Strict cap on \(\lvert\rho\rvert\). |

Typical fit call for recent reports:

```python
PanelMSAR(
    n_regimes=3, common_rho=False, common_sigma=False,
    random_intercepts=..., zero_mu=False, min_t=12, rho_max=0.99,
).fit(..., n_starts=7, maxiter=400, seed=1, compute_se=True, store_filtered=True)
```

Compile TeX from the `.tex` parent directory (`compile_tex` in `msar_report.py`).

## Data vintage

**`data_2026.09.17/realGDP_sa_empl.csv`** — seasonally adjusted quarterly real GDP per worker. Outcome `realGDPsa_usd_pa_empl`. Calendar time from the `t` stamp (`2009q1`), not the `year`/`quarter` columns. Load: `y = log(realGDPsa_usd_pa_empl)` on positive rows; `country = iso3`; `time = year + (quarter-1)/4` from parsed `t`.

Report: `msar_re.pdf` from `run_msar_re.py` (three specs). 74 countries, 8,670 obs, 1950Q1–2026Q2, `rho_max=0.99`. Specs 1–2 used 3 starts; spec 3 (catch-up) used 3 starts, all hitting `maxiter=400`.

| Spec | ll | Trend | Notes |
|---|---|---|---|
| Common \(a+g\) | 22,607.90 | \(a=-6.01\) (0.06), \(g=0.0079\) (0.0005) | \(\rho\) all 0.99 |
| Normal RE intercepts | 23,465.89 | \(\alpha=-5.94\) (0.04), \(\omega=0.51\), \(g=0.0086\) | \(\rho\approx 0.99\), \(E[z]=0.056\) |
| Common \(\lambda\) catch-up | 23,506.07 | \(\bar a=-5.88\) (0.04), \(\lambda=0.9997\) (0.0003), \(\omega_b=0.67\), \(g=0.0085\) | \(\lambda\) at 1; warning: indistinguishable from RE intercepts. \(\rho\) still on the cap. \(\sigma\) 10× warning. |

Catch-up time is in **years**, so \(\lambda=0.9997\) per year is a millennial half-life. The extra ~40 ll vs RE is a different local mode of essentially the same model, not evidence of mean-reverting gaps.

## How to rerun

From repo root, with `numpy`, `pandas`, `scipy`, `numba`, `matplotlib`, and MiKTeX on PATH:

```text
python data_2026.09.17/run_msar_re.py
python demo_panel_msar.py
```

A full RE fit is on the order of 15–40 minutes. Empirical runners use `n_starts=7`; the starts run in a process pool (not threads: the GIL serializes the Python quadrature loop). The final polish from the best start is serial.

## Open issues / next questions

- Pooled and RE \(\rho\) still at the 0.99 cap on the SA extract.
- Many fits hit `maxiter=400`.
- Catch-up is now \(a_i + b_i\lambda^{t-T_{i0}}\) (2-D GH). Spec 3 fixes \(\lambda=0.98\) (Barro); spec 4 frees \(\lambda\in(0,0.985)\). The earlier spec that dropped permanent \(a_i\) is retired.
- `fortran/NL.f90` `discretizeMSAR` is written, not compiled on this machine (`gfortran` missing).
- Do not commit LaTeX aux (`.aux`, `.log`, `.out`); they are gitignored.

## Layout

```text
panel_msar.py                 # estimator
msar_report.py                # LaTeX / cycle-plot helpers
demo_panel_msar.py            # simulated DGP recovery
data_2026.09.17/              # SA GDP per worker + three-spec report
fortran/NL.f90                # MS-AR discretization
plan_common_lambda.md         # catch-up plan (not implemented)
```
