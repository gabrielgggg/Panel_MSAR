# Quarterly real GDP per worker (constant 2015 USD, market FX)

This folder builds, **from scratch**, an unbalanced country–quarter panel of
**real GDP per worker** (employed persons, not population). Nothing in
`../data/` or `../soe_sample/` is used.

| Deliverable | Path |
|---|---|
| Panel | `output/gdp_per_worker_q.csv` |
| Coverage | `output/coverage.csv` |
| Build log | `output/construction_log.txt` |
| Construction note (PDF) | `docs/gdp_per_worker_construction.pdf` |
| Cached extracts | `raw/` |

Required columns: `country` (ISO3), `period` (`YYYY-Qn`), `gdp_per_worker`,
`metadata`. Other columns are audit extras.

Current vintage (this folder’s `raw/` timestamps): **101 countries**,
**7,450** quarterly observations, **1950-Q1–2026-Q2** (unbalanced). Binding
constraint is IMF quarterly real GDP, not employment. 86 countries use
official quarterly ILO employment; 15 use interpolated official annual.

```text
python download_sources.py          # one-time extract cache
python build_gdp_per_worker.py      # construct the panel (downloads if needed)
python -m unittest tests.test_construct
```

Construction note (LaTeX → PDF): from `docs/`, `python make_coverage_table.py` then `pdflatex gdp_per_worker_construction.tex` (twice).

---

## What the series is (and is not)

`gdp_per_worker` is

> annualized real GDP per employed person, in **constant 2015 US dollars
> at 2015 period-average market exchange rates**.

It is **not** PPP, **not** per capita, and **not** converted with the
exchange rate of each quarter. The only FX used for a country is that
country’s 2015 annual average (domestic currency per USD), so movements
in the constructed series are real-LCU volume movements, not FX noise.

---

## Formula

For country *i* and quarter *t*:

\[
Y^{USD}_{i,t}
=
\left(\frac{Y^{r,q}_{i,t}\times 4}{Y^{r,A}_{i,2015}}\right)
\times
\left(\frac{Y^{n,A}_{i,2015}}{E_{i,2015}}\right),
\qquad
\text{gdp per worker}_{i,t}
=
\frac{Y^{USD}_{i,t}}{L_{i,t}}.
\]

| Symbol | Meaning | Source |
|---|---|---|
| \(Y^{r,q}_{i,t}\) | Real GDP, domestic currency, **quarterly flow** (not SAAR) | IMF QNEA `B1GQ`, constant prices, `XDC` |
| \(Y^{r,A}_{i,2015}\) | Real GDP, domestic currency, **2015 annual** | IMF ANEA `B1GQ` constant `XDC`; fallback = sum of four 2015 QNEA quarters |
| \(Y^{n,A}_{i,2015}\) | Current-price GDP, domestic currency, **2015 annual** | IMF ANEA `B1GQ` current `XDC` |
| \(E_{i,2015}\) | 2015 period-average **domestic currency per USD** | IMF ER `XDC_USD` / `PA_RT` / annual |
| \(L_{i,t}\) | Employed persons | ILOSTAT (see below) |
| \(\times 4\) | Convert a quarterly flow to an annualized flow | Unit conversion, not seasonal adjustment |

**Identity (used in tests):** if the four 2015 real quarters sum to the 2015
annual real total, then 2015 annualized \(Y^{USD}\) equals 2015 current-price
GDP divided by \(E_{i,2015}\). Doubling real LCU doubles \(Y^{USD}\). Doubling
the *base-year* FX halves \(Y^{USD}\). No quarterly FX argument exists.

### Why 2015, and why this conversion

Chain-linked national-accounts volumes have a country-specific reference
year (the US QNEA series is referenced to 2017). Dividing each quarter by
that country’s own 2015 real annual total puts every country on a
**2015 = 1 volume index**. Multiplying by 2015 current-price GDP converts
the index back to 2015-price local currency. Dividing by the **2015**
average FX then puts every country in the same international unit:
2015 USD at 2015 market rates.

Using \(E_{i,t}\) every quarter would mix volume with bilateral-dollar
swings. That is what this construction avoids.

### Why multiply by 4

US QNEA 2015-Q1 constant-price GDP is about \(4.67\times 10^{12}\); the four
2015 quarters **sum** to the ANEA 2015 annual total (\(\approx 18.80\times 10^{12}\)).
These are quarterly flows, not BEA-style SAAR. Annualizing by 4 makes
GDP/employment “output per worker *per year*”.

---

## Seasonal adjustment of GDP

IMF QNEA publishes both `SA` and `NSA`.

* If a country has **any** SA constant-price GDP, **all** of that country’s
  GDP observations come from SA (NSA is ignored for that country).
* Otherwise the country is filled from NSA **GDP**, and `gdp_per_worker`
  is then seasonally adjusted with statsmodels `SARIMAX` (see below).

IMF SA and NSA GDP are never spliced inside one country. `gdp_sa` is `SA`
(IMF), `SARIMAX` (NSA GDP, then SARIMAX-adjusted), or `NSA` (too short to
adjust).

### SARIMAX adjustment of NSA countries

NSA GDP (and typically NSA employment) leaves seasonality in
`gdp_per_worker`. For those countries, if there are at least 12 observations
covering all four calendar quarters, we fit

```
log(gdp_per_worker)_t = a + g t + γ_{q(t)} + u_t,
u_t ~ ARMA(p,q)     # SARIMAX, d=0, trend=ct, exog = Q2–Q4 dummies
```

`(p,q)` is chosen by AIC among a small set. Dummy coefficients are *level*
seasonal factors (`d=0` is required: differencing would turn dummies into
pulses). The four `γ`s are recentered to sum to zero so the geometric mean
is preserved. Dummies use the **calendar quarter**, not the row position.

`gdp_real_2015usd` is rebuilt as `gdp_per_worker * emp_persons` after
adjustment. Spells shorter than 12 quarters, or missing a calendar quarter,
stay `NSA` and the metadata records why.

### Power-of-ten unit breaks in QNEA

Some QNEA LCU series change unit (1 → thousands → millions) without a
matching `SCALE`. Jamaica in 2018 is a million-fold drop in the raw
extract while ANEA annual real GDP does not. For each country-year we
compare `4 × mean(quarterly LCU)` to ANEA annual real LCU. If they
differ by \(10^{k}\) with \(|k|\ge 3\), that year’s quarterly LCU is
multiplied by \(10^{-k}\). A SAAR-vs-flow gap of 4 (\(\log_{10}4\approx 0.6\))
is ignored. Years without ANEA inherit the neighbouring year’s factor.
The factor is recorded in `metadata` when it is not 1.

After that rescale, country-years whose `4 × mean(quarterly)` still differs
from ANEA annual real by more than a factor of 5 are **dropped**. Residual
breaks (El Salvador’s colon/USD overlap around dollarization) cannot be
fixed with a power of ten without putting FX back into the volume path.

A remaining adjacent-quarter real-GDP move larger than a factor of 20 is
treated as a unit/currency restatement. Only the contiguous spell that
contains 2015 (the rebase year) is kept, so the volume index is never a
ratio of two different currency units.

Staff projections / imputations (`DERIVATION_TYPE` in `SP`, `SE`, `SEME`,
`SEHI`) are dropped.

---

## Exchange rates, including the euro

IMF ER `XDC_USD` is domestic currency per USD, period average, **annual**.
For the United States the 2015 value is 1. For the United Kingdom it is
the 2015 average pound per dollar.

**Euro-area members as of 1 January 2015** (AT, BE, CY, DE, EE, ES, FI,
FR, GR, IE, IT, LT, LU, LV, MT, NL, PT, SK, SI) have national `XDC_USD`
series that **stop at euro adoption** (Germany: 1998). After adoption the
domestic currency *is* the euro. Those countries, and euro users without a
2015 national series (AD, XK/KOS, ME, SM, VA), inherit the 2015 Euro Area
aggregate rate, IMF country code **`G163`** (`XDC_USD` ≈ 0.901 EUR per USD
in 2015).

That inherited rate is still a **single 2015** converter. Croatia is *not*
in this set: in 2015 it used the kuna.

---

## Employment (per worker, not per capita)

| Priority (country-level, not spliced) | Series | When |
|---|---|---|
| 1 | ILOSTAT official **quarterly** `EMP_TEMP_SEX_AGE_NB_Q` | ≥ 8 quarterly observations after filters |
| 2 | Official **annual** `EMP_TEMP_SEX_AGE_NB_A`, linearly interpolated | otherwise, if official annual exists |
| 3 | ILO **modelled** annual `EMP_2EMP_SEX_AGE_NB_A`, interpolated | last resort |

Filters applied to every ILO extract:

* `sex = SEX_T` (both sexes). Male/female breakdowns are discarded.
* Age preference (one definition per country, never mixed):
  1. `AGE_YTHADULT_YGE15` (15+)
  2. `AGE_AGGREGATE_YGE15` (15+, aggregate classification)
  3. `AGE_AGGREGATE_TOTAL`
  4. `AGE_YTHADULT_Y15-64`
* One ILO `source` code per country (the longest series), so LFS vintages
  are not spliced.
* If official annual employment exists, a quarterly observation whose
  ratio to the interpolated annual path lies outside `[0.4, 2.5]` is
  replaced by that interpolant. This is aimed at known 4× under-counts
  (Israel 2017). The replacement is noted in `metadata`.
* ILO `*_NB` values are **thousands of persons**; they are multiplied by
  1,000.

### Annual → quarterly interpolation

The annual figure for year \(y\) is the annual **average** stock, located
at mid-year \(t = y + 0.5\). Quarter \(q\) of year \(y\) is located at

\[
t = y + (q - 0.5)/4
\quad\text{(Q1: }y+0.125,\ \ldots,\ \text{Q4: }y+0.875\text{)}.
\]

Between adjacent annual observations the interpolant is the unique straight
line in \(t\). Quarters whose midpoint lies **outside** \([t_{\text{first}},
t_{\text{last}}]\) are dropped (no extrapolation). This is interpolation of
a stock, not Denton benchmarking: the four-quarter average equals the
annual level only if employment is linear in calendar time.

---

## Country sample

Keep ISO 3166-1 alpha-3 codes (and Kosovo `KOS` if present). Drop IMF
aggregates (`G163` is used only as an FX donor, never as a panel country).
The panel is the **inner join** of constructed GDP and employment on
`(country, period)`.

---

## Download links (exact extracts)

IMF SDMX 3.0, `Accept: text/csv`. Key wildcards `*` = all countries.
Dataset pages: [QNEA](https://data.imf.org/en/datasets/IMF.STA:QNEA),
[ANEA](https://data.imf.org/en/datasets/IMF.STA:ANEA),
[ER](https://data.imf.org/en/datasets/IMF.STA:ER).

| File in `raw/` | URL |
|---|---|
| `imf_qnea_b1gq_q_sa_xdc_q.csv` | https://api.imf.org/external/sdmx/3.0/data/dataflow/IMF.STA/QNEA/~/*.B1GQ.Q.SA.XDC.Q |
| `imf_qnea_b1gq_q_nsa_xdc_q.csv` | https://api.imf.org/external/sdmx/3.0/data/dataflow/IMF.STA/QNEA/~/*.B1GQ.Q.NSA.XDC.Q |
| `imf_anea_b1gq_q_xdc_a.csv` | https://api.imf.org/external/sdmx/3.0/data/dataflow/IMF.STA/ANEA/~/*.B1GQ.Q.XDC.A |
| `imf_anea_b1gq_v_xdc_a.csv` | https://api.imf.org/external/sdmx/3.0/data/dataflow/IMF.STA/ANEA/~/*.B1GQ.V.XDC.A |
| `imf_er_xdc_usd_pa_rt_a.csv` | https://api.imf.org/external/sdmx/3.0/data/dataflow/IMF.STA/ER/~/*.XDC_USD.PA_RT.A |
| `imf_cl_country.csv` | https://api.imf.org/external/sdmx/3.0/structure/codelist/IMF/CL_COUNTRY |

QNEA key order: `COUNTRY.INDICATOR.PRICE_TYPE.S_ADJUSTMENT.TYPE_OF_TRANSFORMATION.FREQUENCY`.  
ANEA key order: `COUNTRY.INDICATOR.PRICE_TYPE.TYPE_OF_TRANSFORMATION.FREQUENCY`.  
ER key order: `COUNTRY.INDICATOR.TYPE_OF_TRANSFORMATION.FREQUENCY`.

ILOSTAT bulk (Rplumber). Documentation: https://ilostat.ilo.org/data/

| File in `raw/` | URL |
|---|---|
| `ilo_emp_temp_sex_age_nb_q.csv` | https://rplumber.ilo.org/data/indicator/?id=EMP_TEMP_SEX_AGE_NB_Q |
| `ilo_emp_temp_sex_age_nb_a.csv` | https://rplumber.ilo.org/data/indicator/?id=EMP_TEMP_SEX_AGE_NB_A |
| `ilo_emp_2emp_sex_age_nb_a.csv` | https://rplumber.ilo.org/data/indicator/?id=EMP_2EMP_SEX_AGE_NB_A |

---

## Code map

| File | Role |
|---|---|
| `download_sources.py` | URLs, streaming download, `raw/` cache |
| `construct.py` | Every transformation (scale, SA choice, 2015 rebase, euro FX fill, ILO filters, interpolation, metadata) |
| `build_gdp_per_worker.py` | CLI: download if needed → `build_panel` → `output/` |
| `tests/test_construct.py` | Specification tests against the shipped functions |

`construct.build_panel(...)` is the construction entry point. The CLI only
reads the cached CSVs into that function.

---

## Metadata field

Each observation’s `metadata` records, in order: QNEA series and SA/NSA and
the ×4 annualization; the 2015 real and nominal level sources; the 2015 FX
source (including `G163` when used); the ILO series, source code, sex, age,
and thousands→persons conversion; and that the ratio is 2015 USD market FX,
not PPP.

---

## Citations

* IMF, National Economic Accounts, Quarterly (QNEA) and Annual (ANEA).
* IMF, Exchange Rates (ER), domestic currency per USD, period average.
* ILOSTAT, Employment by sex and age (`EMP_TEMP_SEX_AGE_NB`); ILO modelled
  estimates (`EMP_2EMP_SEX_AGE_NB`) where official employment is missing.

Cite the vintage of the cached files in `raw/` (download date = file
timestamps).

---

## CSV quirks handled in code

* IMF ANEA `TIME_PERIOD` arrives as float (`2015.0`) when pandas reads the
  CSV. `parse_year` accepts `2015`, `2015.0`, and integer 2015; it rejects
  `nan`.
* IMF QNEA `TIME_PERIOD` is `YYYY-Qn`; some header-only country rows have
  missing time and are dropped.
* ILO bulk time labels are `YYYYQn` (no hyphen). Both that form and IMF
  `YYYY-Qn` parse to the same `(year, quarter)`.
* IMF `SCALE` is a power-of-ten unit multiplier. Observed GDP extracts use
  `SCALE=0` (units of 1 domestic-currency unit).
* Euro-area national `XDC_USD` series end at euro adoption; 2015 conversion
  uses Euro Area `G163`.
