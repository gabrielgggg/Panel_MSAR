# Resume here

Last updated: 2026-09-17. Repo: **https://github.com/gabrielgggg/Panel_MSAR** (`main`).

Clone, `git pull`, then read this file and `STATUS.md`. Do not restore deleted OECD/SOE/`data_pull` trees. Do not add `output*.txt` to `.gitignore`.

**Latest pushed commit (code):** `c611e38` — *Estimate catch-up with permanent RE intercepts; run starts in processes.*

If this file is missing on a new clone, you are behind that commit.

## What was in flight when this was written

A long empirical job **may still be running on the original laptop** (not on GitHub):

```text
python data_2026.09.17/run_msar_re.py
```

That command fits **spec 3 then spec 4** (7 process-pool starts each, `maxiter=400`, Hessian SEs) and overwrites `data_2026.09.17/msar_re.pdf`. Specs 1–2 are **not** re-estimated; they are spliced from `data_2026.09.17/msar_re_specs12.tex`.

On another machine: if the PDF does not yet have sections titled *RE intercepts plus catch-up, lambda fixed at Barro 0.98* and *… free lambda in (0, 0.985)*, the job did not finish. Re-run the command from the repo root. Expect **seven child `python.exe` processes**. Threads were tried and idle at ~20% CPU because the 2-D quadrature is a Python loop (GIL). Do not switch back to threads.

Wall time is long (2-D 7×7 GH × 74 countries × 7 starts × 2 specs, then a serial polish and Hessian). Keep the machine plugged in.

## Model (current estimator)

`panel_msar.py`, class `PanelMSAR`. Joint panel MS-AR(1), one MLE, Numba Hamilton filter.

\[
y_{it}=a+gt+z_{it}
\quad\text{or}\quad
a_i\sim N(\alpha,\omega^2)
\quad\text{or catch-up}\quad
y_{it}=a_i+gt+b_i\lambda^{t-T_{i0}}+z_{it}.
\]

Catch-up (`convergence=True`): \(a_i\) and \(b_i\sim N(0,\omega_b^2)\) **independent**, 7×7 Gauss–Hermite. \(T_{i0}\) is country \(i\)’s first observation after `_prepare` (internal time). \(\lambda\) is **per year**. Barro 2%/year is \(\lambda=e^{-0.02}\approx 0.980\); we use **0.98**.

| Flag | Role |
|---|---|
| `n_regimes` | Odd. Median \(\mu\) pinned at 0. |
| `common_rho` / `common_sigma` | Empirical reports: both `False`. |
| `random_intercepts` | \(a_i\sim N(\alpha,\omega^2)\), 11-point GH. |
| `convergence` | Adds \(b_i\lambda^{t-T_{i0}}\); forces RE intercepts. |
| `lambda_value` | If set (e.g. `0.98`), \(\lambda\) is fixed. If `None`, \(\lambda\) is estimated. |
| `lambda_max` | Cap on **free** \(\lambda\). Mapped \(\lambda=\texttt{lambda\_max}/(1+e^{-u})\). Default **0.985**. |
| `rho_max` | Empirical: **0.99**. Default in code 0.995. |
| `min_t` | Empirical: **12** observations. |
| `zero_mu` | Off in reports. |

`fit(..., n_starts=7, maxiter=400, seed=1)`. Starts run in `ProcessPoolExecutor`; polish from the best \(\theta\) is serial.

**Removed (do not restore):** country-specific trends, FE intercepts, Pareto RE, quarter dummies, seasonal REs, two-step OLS/CF, quadratic trend, OECD demo, SOE extract, `data_pull/`, `with_pulled_data/`, `data_2026.09.11/`, `data_2026.09.14/`.

## Data

Only vintage in the repo: **`data_2026.09.17/realGDP_sa_empl.csv`**.

- \(y=\log(\texttt{realGDPsa\_usd\_pa\_empl})\) on positive rows.
- `country = iso3`.
- Time from stamp `t` (`2009q1`), **not** the `year`/`quarter` columns: `year + (quarter-1)/4`.
- 74 countries, 8,670 obs, 1950Q1–2026Q2 after filters.

## Report specs

PDF: `data_2026.09.17/msar_re.pdf`. Runner: `data_2026.09.17/run_msar_re.py`. Helpers: `msar_report.py`. Compile TeX **from the `.tex` parent** (MiKTeX `pdflatex`).

| Spec | Status | Notes |
|---|---|---|
| 1 Common \(a+g\) | **Done, frozen** | ll **22,607.90**. \(a=-6.01\) (0.06), \(g=0.0079\) (0.0005), \(\rho\) all 0.99. Cycle: `figs/cycle_common_ag.pdf`. |
| 2 Normal RE intercepts | **Done, frozen** | ll **23,465.89**. \(\alpha=-5.94\) (0.04), \(\omega=0.51\), \(g=0.0086\), \(E[z]=0.056\). Cycle: `figs/cycle_re_ai.pdf`. \(\sigma\) 10× warning. |
| 3 RE \(a_i\) + \(b_i\), \(\lambda=0.98\) fixed | **In flight** | `lambda_value=0.98`. Cycle key `lambda_barro`. |
| 4 Same, \(\lambda\) free in (0, 0.985) | **In flight** | `lambda_value=None`, `lambda_max=0.985`. Cycle key `lambda_free`. |

Frozen TeX for 1–2: `data_2026.09.17/msar_re_specs12.tex`. Do not re-fit 1–2 unless asked.

**Retired:** an earlier spec 3 that **dropped** permanent \(a_i\) and estimated \(\hat\lambda=0.9997\) (indistinguishable from RE intercepts, ll 23,506). Ignore `figs/cycle_lambda.pdf` if present; that plot is the retired spec.

## How to run on a new machine

Python: `numpy`, `pandas`, `scipy`, `numba`, `matplotlib`. TeX: MiKTeX, `pdflatex` on PATH.

```text
git clone https://github.com/gabrielgggg/Panel_MSAR.git
cd Panel_MSAR
git pull origin main
pip install -r requirements.txt
python demo_panel_msar.py
python data_2026.09.17/run_msar_re.py
```

The last command is the report. It needs CPU and time. After it finishes, commit `msar_re.pdf`, `msar_re.tex`, `figs/cycle_lambda_barro.pdf`, `figs/cycle_lambda_free.pdf` if they look right. Do not commit `.aux`/`.log`/`.out` (gitignored).

## Other open items (not started)

- Pooled and RE \(\rho\) still sit on the 0.99 cap.
- Many fits hit `maxiter=400`.
- `fortran/NL.f90` `discretizeMSAR` is written, never compiled here (`gfortran` missing).
- Heterogeneous \(\lambda_i\) is out of scope until specs 3–4 are estimated.
- `plan_common_lambda.md` is stale in places (it described catch-up **instead of** \(a_i\); the code now has **both**).

## Layout

```text
panel_msar.py                 estimator
msar_report.py                LaTeX / cycle-plot helpers
demo_panel_msar.py            simulated DGP
data_2026.09.17/              SA GDP per worker + report
  realGDP_sa_empl.csv
  run_msar_re.py              fits specs 3–4, splices 1–2
  msar_re_specs12.tex         frozen spec 1–2 TeX
  msar_re.pdf / .tex
  figs/cycle_*.pdf
fortran/NL.f90                MS-AR discretization
STATUS.md                     longer lab notebook
RESUME.md                     this file
```
