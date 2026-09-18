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

Report: `msar_re.pdf` from `run_msar_re.py`. Common \(a+g\) vs Normal RE intercepts, full sample, 3 starts, `rho_max=0.99`. 74 countries, 8,670 obs, 1950Q1–2026Q2. Common ll 22,607.90, \(a=-6.01\) (0.06), \(g=0.0079\) (0.0005), \(\rho\) all 0.99. RE ll 23,465.89, \(\alpha=-5.94\) (0.04), \(\omega=0.51\), \(g=0.0086\), \(\rho\) all \(\approx 0.99\), \(E[z]=0.056\). \(\sigma\) 10× warning on RE.

## How to rerun

From repo root, with `numpy`, `pandas`, `scipy`, `numba`, `matplotlib`, and MiKTeX on PATH:

```text
python data_2026.09.17/run_msar_re.py
python demo_panel_msar.py
```

A full RE fit is on the order of 15–40 minutes. Empirical runners use `n_starts=7`; the starts run in parallel threads. The final polish from the best start is serial.

## Open issues / next questions

- Pooled and RE \(\rho\) still at the 0.99 cap on the SA extract.
- Many fits hit `maxiter=400`.
- Common-\(\lambda\) catch-up is planned in `plan_common_lambda.md` (not implemented).
- `fortran/NL.f90` `discretizeMSAR` is written, not compiled on this machine (`gfortran` missing).
- Do not commit LaTeX aux (`.aux`, `.log`, `.out`); they are gitignored.

## Layout

```text
panel_msar.py                 # estimator
msar_report.py                # LaTeX / cycle-plot helpers
demo_panel_msar.py            # simulated DGP recovery
data_2026.09.17/              # SA GDP per worker + two-spec report
fortran/NL.f90                # MS-AR discretization
plan_common_lambda.md         # catch-up plan (not implemented)
```
