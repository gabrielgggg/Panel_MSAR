"""Transform raw IMF + ILO extracts into quarterly real GDP per worker.

Units of the constructed series
-------------------------------
``gdp_per_worker`` is **annualized real GDP per employed person**, expressed in
**constant 2015 US dollars at 2015 market exchange rates** (not PPP).

The exchange rate enters **once**, in the base year. Quarterly (or monthly)
exchange rates are never applied to the volume path, so the constructed series
does not inherit bilateral-FX volatility.

Formula
-------
Let

* ``Y^{r,q}_{i,t}`` = quarterly real GDP in domestic currency (not annualized),
  from IMF QNEA ``B1GQ`` / constant prices / ``XDC``. Prefer seasonally
  adjusted (``SA``); use ``NSA`` only if the country has no SA series.
* ``Y^{r,A}_{i,2015}`` = annual real GDP in domestic currency in 2015
  (IMF ANEA ``B1GQ`` / constant prices / ``XDC``; if missing, the sum of the
  four 2015 quarterly real observations from the same SA/NSA choice).
* ``Y^{n,A}_{i,2015}`` = annual **current-price** GDP in domestic currency in
  2015 (IMF ANEA ``B1GQ`` / current prices / ``XDC``).
* ``E_{i,2015}`` = 2015 period-average domestic currency per USD
  (IMF ER ``XDC_USD`` / ``PA_RT`` / annual). Euro-area members as of 1 Jan 2015
  and euro-using economies with no 2015 national series inherit the Euro Area
  aggregate rate (IMF country code ``G163``).
* ``L_{i,t}`` = employed persons (see employment section).

Then the annualized real GDP in 2015 USD is

    Y^{USD}_{i,t}
        = (Y^{r,q}_{i,t} * 4 / Y^{r,A}_{i,2015})
          * (Y^{n,A}_{i,2015} / E_{i,2015})

and

    gdp_per_worker_{i,t} = Y^{USD}_{i,t} / L_{i,t}.

Why multiply by 4
-----------------
IMF QNEA constant-price GDP is a **quarterly flow**, not a seasonally adjusted
annual rate (SAAR). For the United States, the four 2015 quarters sum to the
ANEA 2015 annual total (about 18.80 trillion 2017-price dollars). Multiplying
each quarter by 4 puts the series on an annualized flow basis so that
GDP / employment has the usual interpretation "output per worker per year".
The factor is a unit conversion, not a seasonal adjustment.

Why this (and not quarterly FX)
-------------------------------
Rebasing the volume index to 2015 and converting **only** 2015 current-price
GDP with the 2015 average FX yields a common international unit
("2015 USD at 2015 exchange rates") without putting ``E_{i,t}`` in every
quarter. Growth of ``Y^{USD}`` equals growth of real GDP in local currency.

Employment
----------
ILO ``EMP_TEMP_SEX_AGE_NB`` is official employment (national LFS/HS/etc.).
``EMP_2EMP_SEX_AGE_NB`` is the ILO modelled estimate. Both are published in
**thousands of persons**; we multiply by 1,000.

Filters (both official and modelled):

* ``sex = SEX_T`` (both sexes).
* Age: prefer ``AGE_YTHADULT_YGE15`` (15+), then ``AGE_AGGREGATE_YGE15``,
  then ``AGE_AGGREGATE_TOTAL``, then ``AGE_YTHADULT_Y15-64``. Age
  definitions are **not mixed within a country**.
* Among official sources for a country (ILO ``source`` codes such as
  ``BA:…`` labour-force surveys), keep the single source with the most
  observations so a country is not spliced across surveys.

Priority, applied **at the country level** (not spliced quarter-by-quarter):

1. Official **quarterly** employment, if the country has at least
   ``MIN_OFFICIAL_Q`` quarterly observations after filters.
2. Otherwise official **annual** employment, linearly interpolated to
   quarter midpoints (see ``interpolate_annual_to_quarterly``).
3. Otherwise ILO **modelled annual** employment, interpolated the same way.

Annual observations are treated as annual averages located at mid-year
``t = year + 0.5``. Quarter ``q`` is located at ``t = year + (q-0.5)/4``.
Interpolation is linear in ``t`` and is **not** extrapolated outside the
closed interval between the first and last annual observation.

NSA countries: SARIMAX seasonal adjustment
------------------------------------------
If QNEA has no SA GDP, the constructed ``gdp_per_worker`` is still seasonal
(NSA GDP, and often NSA employment). Those countries are then adjusted with
statsmodels ``SARIMAX`` on log ``gdp_per_worker``: linear trend, quarterly
dummies (calendar quarter, not positional), ARMA errors, ``d=0`` so dummy
coefficients are level seasonal factors. Factors are recentered to sum to
zero. ``gdp_sa`` becomes ``SARIMAX``. Spells shorter than ``MIN_T_SARIMAX``
or missing a calendar quarter stay ``NSA``.

Country sample
--------------
Keep ISO 3166-1 alpha-3 codes (and the user-assigned Kosovo code ``KOS``).
Drop IMF aggregates (``G163``, ``EUR``, …) and non-three-letter codes.
Staff projections in QNEA/ANEA (``DERIVATION_TYPE`` in ``SP``, ``SE``,
``SEME``, ``SEHI``) are dropped.

The delivered panel is an inner join of constructed GDP and employment on
``(country, period)``. Each row carries a ``metadata`` string that records
the GDP seasonal-adjustment choice, the 2015 level sources, the FX treatment,
and the employment source.
"""

from __future__ import annotations

import math
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

BASE_YEAR = 2015
# ILO EMP_*_NB series are documented as thousands of persons.
ILO_THOUSANDS_TO_PERSONS = 1000.0
# Quarterly QNEA GDP is a non-annualized flow (see module docstring).
QUARTER_TO_ANNUAL = 4.0
MIN_OFFICIAL_Q = 8
# Minimum observed quarters to estimate a quarterly seasonal pattern with
# SARIMAX (three complete years, all four quarters present).
MIN_T_SARIMAX = 12

# Euro-area members whose legal tender on 1 Jan 2015 was the euro.
# Lithuania joined on that date; Latvia (2014), Estonia (2011), Slovakia (2009),
# Cyprus/Malta (2008), Slovenia (2007); the original 1999 members plus Greece
# (2001). Croatia (2023) is intentionally absent: in 2015 it still had the kuna,
# so its own 2015 XDC_USD series is the correct converter.
EURO_AREA_AS_OF_2015 = frozenset(
    {
        "AUT",
        "BEL",
        "CYP",
        "DEU",
        "ESP",
        "EST",
        "FIN",
        "FRA",
        "GRC",
        "IRL",
        "ITA",
        "LTU",
        "LUX",
        "LVA",
        "MLT",
        "NLD",
        "PRT",
        "SVK",
        "SVN",
    }
)
# Economies that used the euro in 2015 without being EA members (no separate
# 2015 national-currency series in IMF ER).
EURO_USERS_2015 = frozenset({"AND", "KOS", "MNE", "SMR", "VAT"})
# IMF ER country code for the euro-area aggregate rate (EUR per USD).
EURO_FX_DONOR = "G163"

# IMF derivation codes that are forecasts / imputations rather than published
# outturns. Dropped from GDP and from the 2015 annual levels.
PROJECTION_DERIVATIONS = frozenset({"SP", "SE", "SEME", "SEHI"})

# Three-letter codes that are regions/aggregates, not economies.
AGGREGATE_ISO3 = frozenset(
    {
        "EUR",  # Europe (IMF)
        "EMU",
        "WLD",
        "EUU",
        "OED",
        "OSS",
        "TSS",
        "PSS",
        "CSS",
        "LCN",
        "NAC",
        "EAS",
        "ECS",
        "MEA",
        "SAS",
        "SSF",
    }
)

ISO3_RE = re.compile(r"^[A-Z]{3}$")
# IMF: 2015-Q1 ; ILO: 2015Q1 / 2015q1
QUARTER_RE = re.compile(r"^(\d{4})[-Qq]Q?([1-4])$")
YEAR_RE = re.compile(r"^(\d{4})$")

AGE_PREFERENCE = (
    "AGE_YTHADULT_YGE15",
    "AGE_AGGREGATE_YGE15",
    "AGE_AGGREGATE_TOTAL",
    "AGE_YTHADULT_Y15-64",
)


def is_economy_code(code: object) -> bool:
    """True for ISO 3166-1 alpha-3 (plus we allow KOS if it appears).

    IMF aggregate codes are mostly ``G###`` / ``U###`` (not three letters).
    A small denylist catches three-letter region codes.
    """
    s = str(code).strip().upper()
    if not ISO3_RE.fullmatch(s):
        return False
    return s not in AGGREGATE_ISO3


def apply_imf_scale(obs_value: object, scale: object) -> float:
    """Return the observation in units of 1 (not millions, billions, …).

    IMF ``SCALE`` is the unit multiplier exponent: ``value * 10**SCALE``.
    Missing scale is treated as 0 (already in units of 1). The US QNEA GDP
    series has ``SCALE=0`` and values around 4.7e12 per quarter in 2015.
    """
    y = pd.to_numeric(obs_value, errors="coerce")
    if pd.isna(y):
        return np.nan
    s = pd.to_numeric(scale, errors="coerce")
    if pd.isna(s):
        s = 0.0
    return float(y) * (10.0 ** float(s))


def parse_quarter(period: object) -> tuple[int, int] | None:
    """Parse ``2015-Q1`` / ``2015Q1`` / ``2015q1`` to ``(year, quarter)``."""
    s = str(period).strip()
    m = QUARTER_RE.match(s)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def parse_year(period: object) -> int | None:
    """Parse an annual period. IMF ANEA CSV often yields floats (``2015.0``)."""
    if period is None:
        return None
    if isinstance(period, (int, np.integer)):
        y = int(period)
        return y if 1000 <= y <= 3000 else None
    if isinstance(period, (float, np.floating)):
        if not np.isfinite(period):
            return None
        y = int(period)
        return y if y == period and 1000 <= y <= 3000 else None
    s = str(period).strip()
    if s.lower() in {"nan", "none", ""}:
        return None
    m = re.match(r"^(\d{4})(?:\.0+)?$", s)
    if not m:
        return None
    return int(m.group(1))


def period_label(year: int, quarter: int) -> str:
    return f"{int(year)}-Q{int(quarter)}"


def quarter_midpoint(year: int, quarter: int) -> float:
    """Calendar time of the quarter's midpoint: Q1→year+0.125, …, Q4→year+0.875."""
    return float(year) + (float(quarter) - 0.5) / 4.0


def drop_projections(df: pd.DataFrame) -> pd.DataFrame:
    if "DERIVATION_TYPE" not in df.columns:
        return df
    der = df["DERIVATION_TYPE"].astype(str).str.strip().str.upper()
    keep = ~der.isin(PROJECTION_DERIVATIONS)
    return df.loc[keep].copy()


def tidy_imf_gdp(
    df: pd.DataFrame,
    *,
    frequency: str,
    adjustment: str | None = None,
) -> pd.DataFrame:
    """Keep economy rows with positive scaled GDP; attach year / quarter.

    ``frequency`` is ``Q`` or ``A``. ``adjustment`` is stored as ``gdp_sa``
    (``SA`` / ``NSA`` / ``A``) for metadata; annual ANEA has no SA dimension.
    """
    out = drop_projections(df)
    out = out.loc[out["COUNTRY"].map(is_economy_code)].copy()
    out["gdp_lcu"] = [
        apply_imf_scale(v, s) for v, s in zip(out["OBS_VALUE"], out.get("SCALE", 0))
    ]
    out = out.loc[np.isfinite(out["gdp_lcu"]) & (out["gdp_lcu"] > 0)].copy()
    if frequency == "Q":
        parsed = out["TIME_PERIOD"].map(parse_quarter)
        out = out.loc[parsed.notna()].copy()
        out["year"] = [p[0] for p in parsed.loc[out.index]]
        out["quarter"] = [p[1] for p in parsed.loc[out.index]]
        out["period"] = [
            period_label(y, q) for y, q in zip(out["year"], out["quarter"])
        ]
    elif frequency == "A":
        years = out["TIME_PERIOD"].map(parse_year)
        out = out.loc[years.notna()].copy()
        out["year"] = years.loc[out.index].astype(int)
        out["quarter"] = pd.NA
        out["period"] = out["year"].astype(str)
    else:
        raise ValueError(f"frequency must be Q or A, got {frequency!r}")
    out["country"] = out["COUNTRY"].astype(str).str.upper()
    out["gdp_sa"] = adjustment if adjustment is not None else "A"
    cols = ["country", "year", "quarter", "period", "gdp_lcu", "gdp_sa"]
    out = out[cols].drop_duplicates(["country", "period"], keep="first")
    return out.sort_values(["country", "year", "quarter"]).reset_index(drop=True)


def power10_unit_factor(annualized_quarterly: float, annual: float) -> float:
    """Return a power-of-10 multiplier that aligns quarterly LCU with annual LCU.

    National-accounts extracts sometimes change unit (units → thousands →
    millions) without a matching ``SCALE``. The tell is that
    ``4 * mean(quarterly)`` differs from the ANEA annual total by 10^3 or
    10^6 (or the inverse). We only act when the rounded log10 gap is at
    least 3, so a SAAR-vs-flow discrepancy of 4 (log10 ≈ 0.6) is left
    untouched.
    """
    if not (np.isfinite(annualized_quarterly) and np.isfinite(annual)):
        return 1.0
    if annualized_quarterly <= 0 or annual <= 0:
        return 1.0
    k = int(round(math.log10(annualized_quarterly / annual)))
    if abs(k) >= 3:
        return 10.0 ** (-k)
    return 1.0


def align_quarterly_lcu_to_annual(
    q_gdp: pd.DataFrame,
    anea_real: pd.DataFrame,
) -> pd.DataFrame:
    """Rescale QNEA LCU by year so ``4 * mean(q)`` matches ANEA annual real.

    The multiplier is a power of ten (see ``power10_unit_factor``). Years
    without an annual counterpart keep the last known factor for that
    country (so a break in 2018 is applied to 2019 even if ANEA is late).
    A column ``gdp_unit_factor`` is stored for metadata.
    """
    annual = anea_real.loc[:, ["country", "year", "gdp_lcu"]].rename(
        columns={"gdp_lcu": "annual_real"}
    )
    q = q_gdp.copy()
    stats = (
        q.groupby(["country", "year"], as_index=False)
        .agg(mean_q=("gdp_lcu", "mean"))
    )
    stats["annualized_q"] = stats["mean_q"] * QUARTER_TO_ANNUAL
    stats = stats.merge(annual, on=["country", "year"], how="left")
    stats["gdp_unit_factor"] = [
        power10_unit_factor(aq, ar) if pd.notna(ar) else np.nan
        for aq, ar in zip(stats["annualized_q"], stats["annual_real"])
    ]
    # Forward/back-fill within country so years without ANEA inherit the
    # neighbouring year's unit (the break is a property of the QNEA vintage).
    stats = stats.sort_values(["country", "year"])
    stats["gdp_unit_factor"] = (
        stats.groupby("country")["gdp_unit_factor"]
        .transform(lambda s: s.ffill().bfill())
        .fillna(1.0)
    )
    q = q.merge(
        stats[["country", "year", "gdp_unit_factor"]],
        on=["country", "year"],
        how="left",
    )
    q["gdp_unit_factor"] = q["gdp_unit_factor"].fillna(1.0)
    q["gdp_lcu"] = q["gdp_lcu"] * q["gdp_unit_factor"]
    return q.reset_index(drop=True)


def drop_years_not_matching_annual(
    q_gdp: pd.DataFrame,
    anea_real: pd.DataFrame,
    *,
    max_rel: float = 5.0,
) -> pd.DataFrame:
    """Drop country-years whose quarterly LCU still disagrees with ANEA.

    After the power-of-ten rescale, ``4 * mean(quarterly)`` should be within
    a factor ``max_rel`` of ANEA annual real GDP. Residual breaks (for
    example El Salvador 2001 dollarization, where QNEA stays in colones
    for a few years while ANEA is restated in USD) are not a power of ten
    and cannot be repaired without mixing FX into the volume path. Those
    years are dropped. Years with no ANEA counterpart are kept.
    """
    annual = anea_real.loc[:, ["country", "year", "gdp_lcu"]].rename(
        columns={"gdp_lcu": "annual_real"}
    )
    stats = (
        q_gdp.groupby(["country", "year"], as_index=False)
        .agg(mean_q=("gdp_lcu", "mean"))
    )
    stats["annualized_q"] = stats["mean_q"] * QUARTER_TO_ANNUAL
    stats = stats.merge(annual, on=["country", "year"], how="left")
    ok = (
        stats["annual_real"].isna()
        | (
            (stats["annualized_q"] > 0)
            & (stats["annual_real"] > 0)
            & (stats["annualized_q"] / stats["annual_real"] >= 1.0 / max_rel)
            & (stats["annualized_q"] / stats["annual_real"] <= max_rel)
        )
    )
    keep = stats.loc[ok, ["country", "year"]]
    return q_gdp.merge(keep, on=["country", "year"], how="inner").reset_index(
        drop=True
    )


def keep_segment_containing_base_year(
    q_gdp: pd.DataFrame,
    *,
    base_year: int = BASE_YEAR,
    jump: float = 20.0,
) -> pd.DataFrame:
    """Keep the contiguous quarterly spell that includes the rebase year.

    After unit alignment, a remaining adjacent-quarter move larger than
    ``jump`` (e.g. El Salvador 2004–05, colon series vs USD restatement)
    means the two sides are not in the same LCU units as ``Y^{r,A}_{2015}``.
    Using them in the 2015 volume index would mix currency units. We keep
    only the spell that contains ``base_year`` (or, if the country has no
    ``base_year`` quarter, the spell of the last observation — rare).
    """
    parts: list[pd.DataFrame] = []
    for country, block in q_gdp.groupby("country", sort=False):
        b = block.sort_values(["year", "quarter"]).copy()
        prev = b["gdp_lcu"].shift(1)
        ratio = b["gdp_lcu"] / prev
        brk = prev.notna() & ((ratio > jump) | (ratio < 1.0 / jump))
        b["_seg"] = brk.cumsum()
        if (b["year"] == base_year).any():
            seg = int(b.loc[b["year"] == base_year, "_seg"].iloc[0])
        else:
            seg = int(b["_seg"].iloc[-1])
        parts.append(b.loc[b["_seg"] == seg].drop(columns="_seg"))
    if not parts:
        return q_gdp.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True)


def choose_sa_or_nsa(sa: pd.DataFrame, nsa: pd.DataFrame) -> pd.DataFrame:
    """Country-level SA preference: if a country has any SA GDP, use only SA.

    Mixing SA and NSA inside one country would put a seasonal-adjustment
    method change into the productivity series. Countries with no SA series
    are filled from NSA.
    """
    sa_countries = set(sa["country"].unique())
    sa_part = sa.copy()
    nsa_part = nsa.loc[~nsa["country"].isin(sa_countries)].copy()
    out = pd.concat([sa_part, nsa_part], ignore_index=True)
    return out.sort_values(["country", "year", "quarter"]).reset_index(drop=True)


def tidy_imf_fx_annual(df: pd.DataFrame) -> pd.DataFrame:
    """Annual period-average domestic currency per USD, scaled.

    ``G163`` (Euro Area) is retained even though it is not an economy: it is
    the donor rate for euro members whose national ``XDC_USD`` series ends
    at euro adoption.
    """
    out = drop_projections(df)
    iso = out["COUNTRY"].astype(str).str.upper()
    keep = iso.map(is_economy_code) | (iso == EURO_FX_DONOR)
    out = out.loc[keep].copy()
    out["fx_xdc_per_usd"] = [
        apply_imf_scale(v, s) for v, s in zip(out["OBS_VALUE"], out.get("SCALE", 0))
    ]
    years = out["TIME_PERIOD"].map(parse_year)
    out = out.loc[years.notna()].copy()
    out["year"] = years.loc[out.index].astype(int)
    out["country"] = out["COUNTRY"].astype(str).str.upper()
    out = out.loc[np.isfinite(out["fx_xdc_per_usd"]) & (out["fx_xdc_per_usd"] > 0)]
    return (
        out[["country", "year", "fx_xdc_per_usd"]]
        .drop_duplicates(["country", "year"], keep="first")
        .sort_values(["country", "year"])
        .reset_index(drop=True)
    )


def base_year_fx(fx_annual: pd.DataFrame, base_year: int = BASE_YEAR) -> pd.Series:
    """One FX rate per economy in ``base_year`` (XDC per USD).

    Euro-area members as of 1 Jan 2015 and euro users with no own 2015
    observation receive the ``G163`` 2015 rate. That is still a **single
    base-year** conversion factor, not a quarterly FX series.
    """
    sub = fx_annual.loc[fx_annual["year"] == base_year]
    rates = sub.set_index("country")["fx_xdc_per_usd"]
    if EURO_FX_DONOR not in rates.index:
        raise ValueError(
            f"missing {EURO_FX_DONOR} {base_year} EUR per USD rate; "
            "cannot convert euro-area members"
        )
    euro = float(rates.loc[EURO_FX_DONOR])
    out = rates.drop(index=EURO_FX_DONOR, errors="ignore").copy()
    need_euro = (EURO_AREA_AS_OF_2015 | EURO_USERS_2015) - set(out.index)
    for c in sorted(need_euro):
        out[c] = euro
    return out.sort_index()


def rebase_quarter_to_base_usd(
    real_lcu_quarter: float,
    real_lcu_base_annual: float,
    nom_lcu_base_annual: float,
    fx_xdc_per_usd_base: float,
    *,
    annualize: float = QUARTER_TO_ANNUAL,
) -> float:
    """Map one quarter of real GDP in LCU to annualized 2015 USD.

    See the module docstring for the formula. Returns NaN if any input is
    non-finite or non-positive (a zero FX or a zero 2015 real level would
    not be a meaningful converter).
    """
    vals = (
        real_lcu_quarter,
        real_lcu_base_annual,
        nom_lcu_base_annual,
        fx_xdc_per_usd_base,
    )
    if any(not np.isfinite(v) or v <= 0 for v in vals):
        return np.nan
    volume_index = (real_lcu_quarter * annualize) / real_lcu_base_annual
    gdp_base_usd = nom_lcu_base_annual / fx_xdc_per_usd_base
    return float(volume_index * gdp_base_usd)


def annual_real_2015(
    q_gdp: pd.DataFrame,
    anea_real: pd.DataFrame,
    base_year: int = BASE_YEAR,
) -> pd.DataFrame:
    """2015 annual real GDP in LCU, one row per country, with a source label.

    Preference:
    1. ANEA annual constant-price GDP in ``base_year`` (same SNA as QNEA).
    2. Else the sum of four quarterly QNEA observations in ``base_year``.
       (Valid because QNEA quarters are non-annualized flows.)
    """
    anea = anea_real.loc[anea_real["year"] == base_year, ["country", "gdp_lcu"]].copy()
    anea = anea.rename(columns={"gdp_lcu": "real_lcu_base"})
    anea["real_base_source"] = f"IMF ANEA B1GQ constant LCU {base_year} annual"
    qsum = (
        q_gdp.loc[q_gdp["year"] == base_year]
        .groupby("country", as_index=False)
        .agg(n=("gdp_lcu", "size"), real_lcu_base=("gdp_lcu", "sum"))
    )
    qsum = qsum.loc[qsum["n"] == 4].drop(columns="n")
    qsum["real_base_source"] = (
        f"sum of 4 IMF QNEA B1GQ constant LCU {base_year} quarters"
    )
    anea_c = set(anea["country"])
    q_only = qsum.loc[~qsum["country"].isin(anea_c)]
    out = pd.concat([anea, q_only], ignore_index=True)
    return out.drop_duplicates("country", keep="first").reset_index(drop=True)


def annual_nom_2015(anea_nom: pd.DataFrame, base_year: int = BASE_YEAR) -> pd.DataFrame:
    out = anea_nom.loc[anea_nom["year"] == base_year, ["country", "gdp_lcu"]].copy()
    out = out.rename(columns={"gdp_lcu": "nom_lcu_base"})
    out["nom_base_source"] = f"IMF ANEA B1GQ current LCU {base_year} annual"
    return out.drop_duplicates("country", keep="first").reset_index(drop=True)


def construct_real_gdp_usd(
    q_gdp: pd.DataFrame,
    real_base: pd.DataFrame,
    nom_base: pd.DataFrame,
    fx_base: pd.Series,
) -> pd.DataFrame:
    """Attach 2015-USD annualized real GDP to every quarterly LCU observation.

    ``fx_base.attrs['euro_filled']`` (optional) lists ISO3 codes that inherited
    the Euro Area (``G163``) 2015 rate because their national ``XDC_USD``
    series ends at euro adoption. That is still a single base-year rate.
    """
    filled = set(getattr(fx_base, "attrs", {}).get("euro_filled", []))
    fx = (
        fx_base.rename("fx_xdc_per_usd_base")
        .rename_axis("country")
        .reset_index()
    )
    merged = q_gdp.merge(real_base, on="country", how="inner")
    merged = merged.merge(nom_base, on="country", how="inner")
    merged = merged.merge(fx, on="country", how="inner")
    merged["gdp_real_2015usd"] = [
        rebase_quarter_to_base_usd(rq, rb, nb, fxv)
        for rq, rb, nb, fxv in zip(
            merged["gdp_lcu"],
            merged["real_lcu_base"],
            merged["nom_lcu_base"],
            merged["fx_xdc_per_usd_base"],
        )
    ]
    merged = merged.loc[
        np.isfinite(merged["gdp_real_2015usd"]) & (merged["gdp_real_2015usd"] > 0)
    ].copy()

    def fx_meta(country: str) -> str:
        if country in filled:
            return (
                f"IMF ER XDC_USD PA {BASE_YEAR} from {EURO_FX_DONOR} "
                f"(euro-area/user; national series ends at euro adoption)"
            )
        return f"IMF ER XDC_USD period-average {BASE_YEAR} (XDC per USD)"

    merged["fx_source"] = merged["country"].map(fx_meta)
    return merged.reset_index(drop=True)


def _pick_age_and_source(block: pd.DataFrame) -> pd.DataFrame:
    """Within one country, pick one age definition then one ILO source code.

    Longest series wins at each step so we do not splice survey breaks.
    """
    if block.empty:
        return block
    present = set(block["classif1"].astype(str))
    age = next((a for a in AGE_PREFERENCE if a in present), None)
    if age is None:
        return block.iloc[0:0].copy()
    aged = block.loc[block["classif1"].astype(str) == age]
    counts = aged.groupby("source").size().sort_values(ascending=False)
    src = counts.index[0]
    return aged.loc[aged["source"] == src].copy()


def tidy_ilo_employment(
    df: pd.DataFrame,
    *,
    frequency: str,
    series_label: str,
) -> pd.DataFrame:
    """Filter ILO employment to total, preferred age, one source per country.

    ``frequency`` is ``Q`` or ``A``. Values are converted to **persons**.
    """
    out = df.copy()
    out["ref_area"] = out["ref_area"].astype(str).str.upper()
    out = out.loc[out["ref_area"].map(is_economy_code)]
    out = out.loc[out["sex"].astype(str) == "SEX_T"]
    out["obs_value"] = pd.to_numeric(out["obs_value"], errors="coerce")
    out = out.loc[out["obs_value"].notna() & (out["obs_value"] > 0)]
    parts = []
    for country, block in out.groupby("ref_area", sort=False):
        picked = _pick_age_and_source(block)
        if not picked.empty:
            parts.append(picked)
    if not parts:
        empty = pd.DataFrame(
            columns=[
                "country",
                "year",
                "quarter",
                "period",
                "emp_persons",
                "emp_source",
                "emp_age",
                "emp_ilo_source",
            ]
        )
        return empty
    out = pd.concat(parts, ignore_index=True)
    out["emp_persons"] = out["obs_value"].astype(float) * ILO_THOUSANDS_TO_PERSONS
    out["country"] = out["ref_area"]
    if frequency == "Q":
        parsed = out["time"].map(parse_quarter)
        # ILO bulk often uses 1999Q1 without a hyphen; also try inserting Q.
        miss = parsed.isna()
        if miss.any():
            alt = out.loc[miss, "time"].astype(str).str.replace(
                r"^(\d{4})q([1-4])$", r"\1-Q\2", case=False, regex=True
            )
            parsed.loc[miss] = alt.map(parse_quarter)
        out = out.loc[parsed.notna()].copy()
        out["year"] = [p[0] for p in parsed.loc[out.index]]
        out["quarter"] = [p[1] for p in parsed.loc[out.index]]
        out["period"] = [
            period_label(y, q) for y, q in zip(out["year"], out["quarter"])
        ]
    elif frequency == "A":
        years = pd.to_numeric(out["time"], errors="coerce")
        out = out.loc[years.notna()].copy()
        out["year"] = years.loc[out.index].astype(int)
        out["quarter"] = pd.NA
        out["period"] = out["year"].astype(str)
    else:
        raise ValueError(frequency)
    out["emp_age"] = out["classif1"].astype(str)
    out["emp_ilo_source"] = out["source"].astype(str)
    out["emp_source"] = series_label
    cols = [
        "country",
        "year",
        "quarter",
        "period",
        "emp_persons",
        "emp_source",
        "emp_age",
        "emp_ilo_source",
    ]
    return (
        out[cols]
        .drop_duplicates(["country", "period"], keep="first")
        .sort_values(["country", "year", "quarter"])
        .reset_index(drop=True)
    )


def interpolate_annual_to_quarterly(annual: pd.DataFrame) -> pd.DataFrame:
    """Linear interpolation of an annual stock onto quarter midpoints.

    The annual value for year ``y`` is placed at ``t = y + 0.5`` (1 July).
    Quarter ``q`` of year ``y`` is placed at ``t = y + (q - 0.5) / 4``.
    Between two adjacent annual observations the interpolant is the unique
    straight line connecting them. No extrapolation: quarters with midpoint
    strictly outside ``[t_first, t_last]`` are dropped.

    This is interpolation of a stock (employment), not Denton benchmarking.
    Interior years' four-quarter average equals the annual level only if
    employment is linear in calendar time.
    """
    rows: list[dict] = []
    for country, block in annual.groupby("country", sort=False):
        b = block.dropna(subset=["emp_persons", "year"]).sort_values("year")
        b = b.drop_duplicates("year", keep="first")
        if len(b) < 2:
            # A single annual point cannot define an interpolant; drop.
            continue
        t_ann = b["year"].astype(float).to_numpy() + 0.5
        e_ann = b["emp_persons"].astype(float).to_numpy()
        y0, y1 = int(b["year"].min()), int(b["year"].max())
        meta = b.iloc[0]
        for year in range(y0, y1 + 1):
            for q in (1, 2, 3, 4):
                t = quarter_midpoint(year, q)
                if t < t_ann[0] or t > t_ann[-1]:
                    continue
                e = float(np.interp(t, t_ann, e_ann))
                if not np.isfinite(e) or e <= 0:
                    continue
                rows.append(
                    {
                        "country": country,
                        "year": year,
                        "quarter": q,
                        "period": period_label(year, q),
                        "emp_persons": e,
                        "emp_source": str(meta["emp_source"])
                        + "; linear interpolation of annual to quarter midpoints "
                        "(annual at y+0.5, quarter at y+(q-0.5)/4; no extrapolation)",
                        "emp_age": meta["emp_age"],
                        "emp_ilo_source": meta["emp_ilo_source"],
                    }
                )
    if not rows:
        return annual.iloc[0:0].copy()
    return pd.DataFrame(rows)


def patch_implausible_quarterly_employment(
    official_q: pd.DataFrame,
    official_a: pd.DataFrame,
    *,
    low: float = 0.4,
    high: float = 2.5,
) -> pd.DataFrame:
    """Replace quarterly employment that is wildly off the annual path.

    ILO official quarterly employment is occasionally a factor of ~4 too
    small for a calendar year (Israel 2017 is the textbook case: ~0.94m
    vs ~3.7m persons). Where an official annual series exists, a quarter
    whose employment / interpolated-annual ratio lies outside
    ``[low, high]`` is replaced by the interpolated annual value and the
    source string is annotated. Quarters without an annual interpolant
    are left unchanged.
    """
    if official_q.empty or official_a.empty:
        return official_q
    interp = interpolate_annual_to_quarterly(official_a)
    if interp.empty:
        return official_q
    ref = interp.rename(columns={"emp_persons": "emp_annual_interp"})
    merged = official_q.merge(
        ref[["country", "period", "emp_annual_interp"]],
        on=["country", "period"],
        how="left",
    )
    ratio = merged["emp_persons"] / merged["emp_annual_interp"]
    ratio_num = pd.to_numeric(ratio, errors="coerce")
    bad = (
        merged["emp_annual_interp"].notna()
        & ratio_num.notna()
        & ((ratio_num < low) | (ratio_num > high))
    )
    out = merged.copy()
    if bad.any():
        out.loc[bad, "emp_persons"] = out.loc[bad, "emp_annual_interp"]
        note = (
            "; replaced with interpolated official annual because quarterly "
            f"employment was outside {low}–{high} times the annual path"
        )
        out.loc[bad, "emp_source"] = out.loc[bad, "emp_source"].astype(str) + note
    cols = [
        "country",
        "year",
        "quarter",
        "period",
        "emp_persons",
        "emp_source",
        "emp_age",
        "emp_ilo_source",
    ]
    return out[cols].reset_index(drop=True)


def select_employment(
    official_q: pd.DataFrame,
    official_a: pd.DataFrame,
    modelled_a: pd.DataFrame,
    *,
    min_official_q: int = MIN_OFFICIAL_Q,
) -> pd.DataFrame:
    """Country-level employment source hierarchy (no intra-country splicing)."""
    official_q = patch_implausible_quarterly_employment(official_q, official_a)
    q_counts = official_q.groupby("country").size()
    q_countries = set(q_counts[q_counts >= min_official_q].index)
    q_part = official_q.loc[official_q["country"].isin(q_countries)].copy()
    q_part["emp_source"] = q_part["emp_source"].astype(str)

    rest = set(official_a["country"].unique()) - q_countries
    a_part = interpolate_annual_to_quarterly(
        official_a.loc[official_a["country"].isin(rest)]
    )

    used = q_countries | set(a_part["country"].unique())
    m_rest = modelled_a.loc[~modelled_a["country"].isin(used)]
    m_part = interpolate_annual_to_quarterly(m_rest)

    chunks = [p for p in (q_part, a_part, m_part) if p is not None and len(p)]
    if not chunks:
        cols = [
            "country",
            "year",
            "quarter",
            "period",
            "emp_persons",
            "emp_source",
            "emp_age",
            "emp_ilo_source",
        ]
        return pd.DataFrame(columns=cols)
    out = pd.concat(chunks, ignore_index=True)
    return out.sort_values(["country", "year", "quarter"]).reset_index(drop=True)


def _quarter_dummies(quarters: np.ndarray) -> pd.DataFrame:
    """Q2–Q4 dummies; Q1 is the omitted reference quarter."""
    q = pd.Series(pd.to_numeric(quarters, errors="coerce"), dtype="Int64")
    out = pd.DataFrame(
        {
            "q_2": (q == 2).astype(float),
            "q_3": (q == 3).astype(float),
            "q_4": (q == 4).astype(float),
        }
    )
    return out


def _centered_quarter_effects(beta_q2: float, beta_q3: float, beta_q4: float) -> np.ndarray:
    """Four log-seasonal factors that sum to zero (Q1 is the omitted dummy)."""
    g = np.array([0.0, float(beta_q2), float(beta_q3), float(beta_q4)])
    return g - g.mean()


def sarimax_log_seasonal_adjust(
    values: np.ndarray,
    quarters: np.ndarray,
    *,
    min_t: int = MIN_T_SARIMAX,
) -> tuple[np.ndarray | None, str]:
    """Seasonally adjust a positive quarterly series with statsmodels SARIMAX.

    Model (logs)::

        log y_t = a + g t + γ_{q(t)} + u_t,
        u_t ~ ARMA(p, q)

    implemented as ``SARIMAX(log y, order=(p,0,q), trend='ct', exog=Q2–Q4)``.
    ``d=0`` is required so dummy coefficients are *level* seasonal factors
    (differencing would turn dummies into pulses). ``γ`` is recentered so
    the four quarter effects sum to zero; the sample geometric mean of ``y``
    is therefore (approximately) preserved.

    ``(p,q)`` is chosen by AIC among a small set. Returns
    ``(adjusted_values, note)`` or ``(None, reason)`` if the series is too
    short or the fit fails.

    Dummies use the **calendar quarter** (1–4), not the positional index, so
    a series that starts in Q3 or has gaps still gets the right season.
    """
    y = np.asarray(values, dtype=float)
    q = np.asarray(quarters, dtype=float)
    ok = np.isfinite(y) & (y > 0) & np.isin(q, (1.0, 2.0, 3.0, 4.0))
    if int(ok.sum()) < min_t:
        return None, f"too few observations for SARIMAX SA (need>={min_t})"
    if len(np.unique(q[ok])) < 4:
        return None, "not all four quarters present; cannot identify seasonal dummies"
    logy = np.log(y)
    exog = _quarter_dummies(q)
    # statsmodels wants a 1-d endogenous with possible NaN; keep full length
    # so dummy rows align. Mark invalid points as NaN (Kalman skips them).
    endog = np.where(ok, logy, np.nan)
    candidates = ((1, 0, 1), (0, 0, 1), (1, 0, 0), (2, 0, 1), (0, 0, 0))
    best = None
    best_order = None
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p, d, q_ma in candidates:
            try:
                mod = SARIMAX(
                    endog,
                    order=(p, d, q_ma),
                    trend="ct",
                    exog=exog,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                res = mod.fit(disp=False, maxiter=400, method="lbfgs")
                if not np.isfinite(res.aic):
                    continue
                if best is None or res.aic < best.aic:
                    best = res
                    best_order = (p, d, q_ma)
            except (ValueError, np.linalg.LinAlgError, RuntimeError):
                continue
    if best is None or best_order is None:
        return None, "SARIMAX fit failed"
    params = best.params
    if not isinstance(params, pd.Series):
        params = pd.Series(params, index=getattr(best.model, "param_names", None))
    g = _centered_quarter_effects(
        float(params.get("q_2", 0.0)),
        float(params.get("q_3", 0.0)),
        float(params.get("q_4", 0.0)),
    )
    seas = np.array([g[int(qi) - 1] if np.isfinite(qi) else 0.0 for qi in q])
    log_sa = logy - seas
    sa = np.exp(log_sa)
    sa = np.where(ok, sa, np.nan)
    note = (
        f"statsmodels SARIMAX{best_order} trend=ct + quarterly dummies "
        f"on log(gdp_per_worker); seasonal factors recentered to sum to 0; "
        f"AIC={best.aic:.2f}"
    )
    return sa, note


def seasonally_adjust_nsa_countries(
    panel: pd.DataFrame,
    *,
    min_t: int = MIN_T_SARIMAX,
) -> pd.DataFrame:
    """Replace NSA ``gdp_per_worker`` with SARIMAX-adjusted values in place.

    IMF-SA countries are left unchanged. NSA countries that cannot be
    adjusted stay ``gdp_sa='NSA'``. Successful adjustments set
    ``gdp_sa='SARIMAX'``, rebuild ``gdp_real_2015usd`` as
    ``gdp_per_worker * emp_persons`` so the identity still holds, and append
    the SARIMAX note to ``metadata``.
    """
    out = panel.copy()
    if "gdp_sa" not in out.columns:
        return out
    notes: dict[str, str] = {}
    for country, block in out.groupby("country", sort=False):
        if str(block["gdp_sa"].iloc[0]) != "NSA":
            continue
        sa, note = sarimax_log_seasonal_adjust(
            block["gdp_per_worker"].to_numpy(),
            block["quarter"].to_numpy(),
            min_t=min_t,
        )
        notes[str(country)] = note
        if sa is None:
            continue
        idx = block.index
        out.loc[idx, "gdp_per_worker"] = sa
        out.loc[idx, "gdp_real_2015usd"] = (
            out.loc[idx, "gdp_per_worker"] * out.loc[idx, "emp_persons"]
        )
        out.loc[idx, "gdp_sa"] = "SARIMAX"
        out.loc[idx, "metadata"] = (
            out.loc[idx, "metadata"].astype(str)
            + " | seasonal adjustment: "
            + note
        )
    # Record skip reasons on remaining NSA rows so the audit trail is complete.
    still_nsa = out["gdp_sa"].astype(str) == "NSA"
    if still_nsa.any():
        def _skip_meta(row: pd.Series) -> str:
            reason = notes.get(str(row["country"]), "SARIMAX SA not applied")
            if "seasonal adjustment:" in str(row["metadata"]):
                return str(row["metadata"])
            return str(row["metadata"]) + " | seasonal adjustment: not applied (" + reason + ")"

        out.loc[still_nsa, "metadata"] = out.loc[still_nsa].apply(_skip_meta, axis=1)
    return out.reset_index(drop=True)


def compose_metadata(row: pd.Series) -> str:
    """One-line audit trail of sources and transformations for the observation."""
    parts = [
        (
            f"gdp: IMF QNEA B1GQ constant-price LCU {row['gdp_sa']} "
            f"{row['period']}; not annualized in source, multiplied by "
            f"{int(QUARTER_TO_ANNUAL)} to annualize"
            + (
                f"; QNEA LCU rescaled by {row['gdp_unit_factor']:g} so that "
                "4*mean(quarterly) matches ANEA annual real (power-of-10 unit break)"
                if float(row.get("gdp_unit_factor", 1.0) or 1.0) != 1.0
                else ""
            )
        ),
        (
            f"rebase: volume index vs {BASE_YEAR} real LCU "
            f"[{row['real_base_source']}] times {BASE_YEAR} current-price LCU "
            f"[{row['nom_base_source']}] divided by {row['fx_source']}"
        ),
        (
            f"employment: {row['emp_source']}; ILO source={row['emp_ilo_source']}; "
            f"sex=SEX_T; age={row['emp_age']}; "
            f"values in thousands converted to persons (x{int(ILO_THOUSANDS_TO_PERSONS)})"
        ),
        (
            f"gdp_per_worker = annualized real GDP in {BASE_YEAR} USD "
            f"(market FX, not PPP) / employed persons"
        ),
    ]
    return " | ".join(parts)


def build_panel(
    *,
    qnea_sa: pd.DataFrame,
    qnea_nsa: pd.DataFrame,
    anea_real: pd.DataFrame,
    anea_nom: pd.DataFrame,
    er_annual: pd.DataFrame,
    ilo_official_q: pd.DataFrame,
    ilo_official_a: pd.DataFrame,
    ilo_modelled_a: pd.DataFrame,
    country_names: pd.DataFrame | None = None,
    base_year: int = BASE_YEAR,
) -> pd.DataFrame:
    """Run the full construction. All arguments are **raw** IMF/ILO extracts.

    This is the function tests should call: it is the shipped entry point
    from in-memory frames (the CLI only downloads then reads CSV into these
    arguments).
    """
    sa = tidy_imf_gdp(qnea_sa, frequency="Q", adjustment="SA")
    nsa = tidy_imf_gdp(qnea_nsa, frequency="Q", adjustment="NSA")
    q_gdp = choose_sa_or_nsa(sa, nsa)

    real_a = tidy_imf_gdp(anea_real, frequency="A", adjustment="A")
    nom_a = tidy_imf_gdp(anea_nom, frequency="A", adjustment="A")
    q_gdp = align_quarterly_lcu_to_annual(q_gdp, real_a)
    q_gdp = drop_years_not_matching_annual(q_gdp, real_a)
    q_gdp = keep_segment_containing_base_year(q_gdp, base_year=base_year)
    real_base = annual_real_2015(q_gdp, real_a, base_year=base_year)
    nom_base = annual_nom_2015(nom_a, base_year=base_year)

    fx_annual = tidy_imf_fx_annual(er_annual)
    # Record which euro countries lacked a national 2015 rate before fill.
    raw_2015 = set(
        fx_annual.loc[fx_annual["year"] == base_year, "country"]
    ) - {EURO_FX_DONOR}
    fx_base = base_year_fx(fx_annual, base_year=base_year)
    filled = (EURO_AREA_AS_OF_2015 | EURO_USERS_2015) - raw_2015
    fx_base.attrs["euro_filled"] = filled

    gdp_usd = construct_real_gdp_usd(q_gdp, real_base, nom_base, fx_base)

    emp_q = tidy_ilo_employment(
        ilo_official_q,
        frequency="Q",
        series_label="ILOSTAT EMP_TEMP_SEX_AGE_NB quarterly (official)",
    )
    emp_a = tidy_ilo_employment(
        ilo_official_a,
        frequency="A",
        series_label="ILOSTAT EMP_TEMP_SEX_AGE_NB annual (official)",
    )
    emp_m = tidy_ilo_employment(
        ilo_modelled_a,
        frequency="A",
        series_label="ILOSTAT EMP_2EMP_SEX_AGE_NB annual (ILO modelled estimates)",
    )
    emp = select_employment(emp_q, emp_a, emp_m)

    panel = gdp_usd.merge(
        emp,
        on=["country", "year", "quarter", "period"],
        how="inner",
        validate="one_to_one",
    )
    panel = panel.loc[
        (panel["gdp_real_2015usd"] > 0) & (panel["emp_persons"] > 0)
    ].copy()
    panel["gdp_real_2015usd"] = pd.to_numeric(panel["gdp_real_2015usd"], errors="coerce")
    panel["emp_persons"] = pd.to_numeric(panel["emp_persons"], errors="coerce")
    panel["gdp_per_worker"] = panel["gdp_real_2015usd"] / panel["emp_persons"]
    gpw = panel["gdp_per_worker"]
    panel = panel.loc[gpw.notna() & np.isfinite(gpw.to_numpy(dtype=float)) & (gpw > 0)]
    panel["time"] = panel["year"] + (panel["quarter"] - 1.0) / 4.0
    panel["metadata"] = panel.apply(compose_metadata, axis=1)
    panel = seasonally_adjust_nsa_countries(panel)

    if country_names is not None and not country_names.empty:
        names = country_names.copy()
        names["iso3"] = names["iso3"].astype(str).str.upper()
        names = names.drop_duplicates("iso3")
        panel = panel.merge(
            names.rename(columns={"iso3": "country", "country_name": "country_name"}),
            on="country",
            how="left",
        )
    else:
        panel["country_name"] = pd.NA

    cols = [
        "country",
        "period",
        "gdp_per_worker",
        "metadata",
        "country_name",
        "year",
        "quarter",
        "time",
        "gdp_real_2015usd",
        "emp_persons",
        "gdp_sa",
        "emp_source",
    ]
    panel = panel[cols].sort_values(["country", "time"]).reset_index(drop=True)
    dup = panel.duplicated(["country", "period"])
    if dup.any():
        raise ValueError(f"{int(dup.sum())} duplicate country-period rows")
    return panel


def coverage_table(panel: pd.DataFrame) -> pd.DataFrame:
    g = (
        panel.groupby(["country", "country_name", "gdp_sa"], dropna=False)
        .agg(
            n=("gdp_per_worker", "size"),
            t0=("period", "min"),
            t1=("period", "max"),
            emp_source=("emp_source", "first"),
        )
        .reset_index()
        .sort_values("n", ascending=False)
    )
    return g.reset_index(drop=True)


def read_raw(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """Load the cached extracts produced by ``download_sources.ensure_raw``."""
    raw_dir = Path(raw_dir)

    def csv(name: str) -> pd.DataFrame:
        path = raw_dir / name
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing; run download_sources.py first"
            )
        return pd.read_csv(path, low_memory=False)

    return {
        "qnea_sa": csv("imf_qnea_b1gq_q_sa_xdc_q.csv"),
        "qnea_nsa": csv("imf_qnea_b1gq_q_nsa_xdc_q.csv"),
        "anea_real": csv("imf_anea_b1gq_q_xdc_a.csv"),
        "anea_nom": csv("imf_anea_b1gq_v_xdc_a.csv"),
        "er_annual": csv("imf_er_xdc_usd_pa_rt_a.csv"),
        "ilo_official_q": csv("ilo_emp_temp_sex_age_nb_q.csv"),
        "ilo_official_a": csv("ilo_emp_temp_sex_age_nb_a.csv"),
        "ilo_modelled_a": csv("ilo_emp_2emp_sex_age_nb_a.csv"),
        "country_names": csv("imf_cl_country.csv"),
    }
