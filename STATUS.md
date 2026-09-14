# Status (resume here)

Last updated: 2026-09-14. Repo: `https://github.com/gabrielgggg/Panel_MSAR` (`main`).

Joint panel Markov-switching AR(1) around a (possibly country-specific) trend for an unbalanced country panel. Outcome is intended in logs. One joint MLE, not country-by-country. Code lives in `panel_msar.py` (`PanelMSAR`). Numba Hamilton filter. Reports are LaTeX (Palatino / `mathpazo`), compiled with MiKTeX `pdflatex` from the report’s own folder.

## Locked modeling choices

- Odd `k` (usually 3). Median \(\mu\) pinned at 0 unless `zero_mu=True`.
- Default: switching \(\sigma\), switching \(\rho\) in the recent empirical runs (`common_rho=False`, `common_sigma=False`).
- Calendar time, common origin. Internally `t` is shifted so the earliest date is 0.
- \(|\rho|<\texttt{rho\_max}\). Default `RHO_MAX=0.995`. Empirical runs on the 2026-09-14 file use `rho_max=0.99`.
- `min_t=12` observations (not years). Longest contiguous spell kept.
- Country intercepts / trends can be profiled (`country_intercepts`, `country_trends`). Recent work uses **random intercepts** instead of FE intercepts.
- Do **not** overwrite `soe_sample/soe_msar_specs.pdf`.
- `output*.txt` estimate dumps were removed from git; they are **not** gitignored (do not add `output*.txt` to `.gitignore`).
- Do not add new files under `data_pull/` or `soe_sample/` unless asked. New empirical work goes in dated folders (`data_2026.09.11/`, `data_2026.09.14/`) or `with_pulled_data/`.

## Estimator flags (current `panel_msar.py`)

| Flag | Meaning |
|---|---|
| `quarter_dummies` | Common Q2, Q3, Q4 in the trend (Q1 omitted). Incompatible with `two_step`. |
| `random_intercepts` | \(a_i\sim N(\alpha,\omega^2)\), 11-point Gauss–Hermite. Incompatible with `two_step` and `country_trends`. |
| `random_seasonals` | Also \(d_{ji}\sim N(\delta_j,\omega_j^2)\). 4D integral by **Laplace**. Implies RE intercepts + quarter dummies. |
| `re_family` | `'normal'` (GH) or `'pareto'` (shifted Type II / Lomax on \([m,\infty)\), location/scale/shape, 15-point Gauss–Legendre on the PIT). Pareto cannot combine with `random_seasonals`. |
| `rho_max` | Strict cap on \(\lvert\rho\rvert\). |
| `two_step` | `False`, `'quadratic'`, or `'cf'` (Christiano–Fitzgerald; `cf_cutoff` years, default 15). |

Hessian SEs are delta-method on shared \(\theta\) (and on \(\pi\), \(E[z]\)). Country-cluster bootstrap exists but is slow (user stopped a B=40 FE-intercept bootstrap).

Typical fit call for recent reports:

```python
PanelMSAR(
    n_regimes=3, common_rho=False, common_sigma=False,
    country_intercepts=False, country_trends=False,
    random_intercepts=..., quarter_dummies=...,
    random_seasonals=..., re_family=...,
    two_step=False, zero_mu=False, min_t=12, rho_max=0.99,
).fit(..., n_starts=3, maxiter=400, seed=1, compute_se=True, store_filtered=True)
```

Compile TeX from the `.tex` parent directory (`compile_tex` in `soe_sample/run_soe_spec_report.py` already does that).

## Data vintages

**`soe_sample/`** — older SOE extract (`realGDP_empl.csv` / `.dta`). `y=log(realGDP_empl)`. SVK 2000–01 and ISR 2017 employment were patched (4× too small). Reports: `soe_msar_specs.pdf` (six specs; do not overwrite), `soe_msar_linear_1970.pdf` (1970+, common \(a,g\) vs FE \(a_i\)), `soe_msar_re_1970.pdf` (common vs Normal RE).

**`data_pull/output/gdp_per_worker_q.csv`** — constructed GDP per worker (2015 USD). Frozen for `with_pulled_data/`. Time column `time` is already year-fraction. Do not write new files into `data_pull/`.

**`data_2026.09.11/realGDP_nsa_empl.csv`** — NSA quarterly GDP per worker. Outcome `realGDP_usd_pa_empl`. Calendar time from the `t` stamp (`2009q1`), **not** the `year`/`quarter` columns (those disagree on many rows). Jumps in \(z_{it}\) are jumps in \(y_{it}\).

**`data_2026.09.14/realGDP_nsa_empl_s3_both.csv`** — current working extract. Same outcome and `t` convention. Fixes some of the 09.11 unit/employment breaks. 90 countries with positive y, 8,489 quarters, 1963Q1–2026Q2 (7,975 from 1995Q1). No `iso3`–`t` duplicates. `year`/`quarter` can still disagree with `t`; runners parse `t`.

Load pattern used in dated runners: `y = log(realGDP_usd_pa_empl)` on positive rows; `country = iso3`; `time = year + (quarter-1)/4` from parsed `t`.

## Reports and headline results

Paths are relative to the repo root. Empirical runs often hit `maxiter=400`; Hessian SEs may still exist. \(\rho\) often sits on the cap in pooled specs.

### SOE (`soe_sample/`)

- `soe_msar_specs.pdf` — six-spec survey (common / FE intercepts / FE trends / CF / CF+zero_mu / CF+common \(\rho\)).
- `soe_msar_linear_1970.pdf` — 1970+: common \(a,g\) (ll 16,652) vs FE \(a_i\)+common \(g\) (ll 17,887; Hessian failed).
- `soe_msar_re_1970.pdf` — 1970+: common vs Normal RE. RE ll 17,673; \(\alpha=-3.98\), \(\omega=1.19\). Hessian OK. \(\rho\) on 0.995 cap in the RE spec.

### Pulled GDP per worker (`with_pulled_data/`)

Runner: `run_pulled_re_1970.py`. Report `msar_re_1970.pdf` was last redone on the **full** pulled sample with `rho_max=0.99` (filename still says 1970). Pooled ll 14,718 vs RE 15,231. Pooled \(\rho\) all at 0.99; pooled \(g\approx 0\).

### 2026-09-11 (`data_2026.09.11/`)

- `run_msar_re.py` → `msar_re.pdf`. Common vs Normal RE, **no** seasonal dummies, full sample, `rho_max=0.99`, 4 starts. Pooled ll 11,694 vs RE 11,980. Middle regime: rare, high \(\sigma\), low \(\rho\).
- Cycle plots showed cliffs. Investigation: `z_jumps_report.pdf`, `jump_events.csv`, `z_jumps.csv`. **Not currency redenomination** (`NER_pa_base` is a country constant). Main causes: employment 4× unit bugs (SVK, ISR, MNE), GDP units break (SLV 2005), source toggles (DOM sawtooth), NSA winter drops (ROM), Macao COVID. Same SVK/ISR 4× bug that was patched in the SOE file.

### 2026-09-14 (`data_2026.09.14/`) — current

| Report | Runner | Specs | Sample | Notes |
|---|---|---|---|---|
| `msar_re.pdf` | `run_msar_re.py` | common vs Normal RE, no dummies | full | ll 11,645 vs **11,896**. RE \(\rho=(0.50,0.93,0.99)\). |
| `msar_re_seasonal.pdf` | `run_msar_seasonal.py` | (1) common + Q dummies (2) Normal RE + common Q dummies (3) Laplace seasonal REs | full, 3 starts | **(2) is the winner**, ll **13,469**. Q4 dummy \(\approx +4.3\%\) vs Q1. (3) Laplace worse (ll 13,005), not comparable to GH, treat as failed extra. |
| `msar_pareto.pdf` | `run_msar_pareto.py` | common + Q dummies vs Pareto Type II RE + Q dummies | full, 3 starts | Pareto ll 13,440 < Normal RE 13,469. Shape \(\approx 20\), scale poorly identified; not a heavy tail. |
| `msar_1995.pdf` | `run_msar_1995.py` | same as seasonal (1)–(2) | 1995Q1+ | ll 12,437 vs **12,732**. Q dummies almost identical to full sample. RE \(\rho=(0.67,0.97,0.99)\). |

Working recommendation if resuming empirical work: **Normal RE intercepts + common Q2–Q4 dummies + common \(g\)**, `rho_max=0.99`, on `data_2026.09.14`. That is seasonal spec (2). Pooled specs still pin \(\rho\) at the cap. Spec (3) seasonal REs via Laplace did not improve the fit; most NSA mean is common, not country-specific (\(\omega_j\sim 0.05\)–\(0.08\)).

## Seasonality (NSA)

The GDP is not seasonally adjusted. Options discussed; implemented so far are **common quarterly dummies** and (failed) **RE quarterly dummies**. Do not X-13 the main spec unless the paper’s claim changes to SA GDP per worker. YoY \(\Delta_4 y\) would drop RE intercepts and change the estimand.

Country-specific seasonal FE on `min_t=12` overfits. 4D GH for seasonal REs is impractical (\(11^4\) nodes); Laplace was the workaround and underperformed.

## How to rerun

From repo root, with `numpy`, `pandas`, `scipy`, `numba`, `matplotlib`, and MiKTeX on PATH:

```text
python data_2026.09.14/run_msar_re.py          # no dummies, full sample
python data_2026.09.14/run_msar_seasonal.py    # three seasonal specs
python data_2026.09.14/run_msar_pareto.py      # Pareto RE
python data_2026.09.14/run_msar_1995.py        # 1995+ dummy specs
```

A full RE fit is on the order of 15–40 minutes per spec (Laplace and Pareto slower). `n_starts=3` in the latest runners.

## Open issues / next questions

- Pooled \(\rho\) still at the 0.99 cap even with dummies; RE takes one regime off the cap.
- Many fits hit `maxiter=400`. Raising `maxiter` or polishing more would be a mechanical next step, not a spec change.
- Pareto Type II is one-sided (\([m,\infty)\)). A two-sided heavy tail (Student-\(t\) RE) was not tried.
- Leftover country sawtooth after **common** dummies is the reason seasonal REs were tried; they did not win on likelihood.
- `year`/`quarter` vs `t` still messy in the CSV; always parse `t`.
- Do not commit LaTeX aux (`.aux`, `.log`, `.out`); they are gitignored.

## Layout

```text
panel_msar.py                 # estimator
soe_sample/                   # SOE extract + older reports
data_pull/                    # construction pipeline (read-only for empirical reruns)
with_pulled_data/             # report on data_pull GDP per worker
data_2026.09.11/              # first NSA extract + jump investigation
data_2026.09.14/              # current NSA extract + latest reports
data/                         # OECD demo
```
