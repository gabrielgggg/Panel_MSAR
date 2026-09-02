# Demo OECD panel

Unbalanced quarterly country panel of **real GDP per worker** for `demo_oecd_msar.py`.

| | |
|---|---|
| File | `demo_oecd_gdp_per_worker_q.csv` |
| Source | OECD Economic Outlook No. 118, SDMX `DSD_EO@DF_EO` |
| Refresh | `python data/fetch_demo_oecd.py` |
| Fit | `python demo_oecd_msar.py` |
| Coverage | 34 countries, 6,784 quarterly observations, 1960-Q1–2025-Q4 (unbalanced) |
| `gdp_real_usd` | Chain-linked real GDP, USD, constant exchange rates (`GDPV_USD`) |
| `emp` | Total employment, persons, LFS (`ET`) |
| `gdp_per_worker` | `gdp_real_usd / emp` — **per worker, not per capita** |
| `y` | `log(gdp_per_worker)` |
| `time` | Calendar year-fraction (`1960.00`, `1960.25`, …) |

Plots: `demo_oecd_log_gdp_per_worker.pdf`, `demo_oecd_log_gdp_per_worker_detrended.pdf`.

This is **not** PPP (Penn World Table is annual). Levels are comparable at constant FX, not at PPP. The last few quarters of 2025 may include Economic Outlook nowcasts/forecasts.

**Not in this extract:** HUN, MEX, ZAF (missing GDP or employment), and most non-OECD economies.

Cite: OECD (2026), OECD Economic Outlook No. 118, https://www.oecd.org/economic-outlook/
