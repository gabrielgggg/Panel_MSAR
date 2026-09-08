"""Load soe_sample/realGDP_empl.csv into the PanelMSAR (country, time, y) layout.

y is log real GDP per worker (column realGDP_empl). SVK 2000–01 and ISR 2017
quarterly employment is replaced by the interpolated annual series (those
windows are ~4x too small in empl_q) and productivity is rebuilt from
GDPrealsa. Time is year-fraction
(1970.0, 1970.25, ...) so g is per year and rho is per quarter. Country id
is ISO3. Missing or non-positive realGDP_empl rows are dropped; the
estimator later keeps each country's longest contiguous spell.

Usage:
    python soe_sample/prepare_soe_panel.py
    from soe_sample.prepare_soe_panel import load_panel
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE / "realGDP_empl.csv"
OUT = HERE / "soe_realgdp_empl_q.csv"

# Quarterly employment is ~4x too small vs the annual series in these
# windows (SVK 2000–01, ISR 2017). Use the interpolated annual path
# and rebuild realGDP_empl = GDPrealsa / (empl * 1000), which is the
# identity that holds on every other row.
EMPL_X4_WINDOWS = (
    ("SVK", 2000, 2001),
    ("ISR", 2017, 2017),
)


def _patch_empl_x4(raw: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Replace 4x-too-small quarterly employment; recompute productivity."""
    out = raw.copy()
    gdp = pd.to_numeric(out.get("GDPrealsa"), errors="coerce")
    interp = pd.to_numeric(out.get("empl_interp"), errors="coerce")
    iso = out["iso3"].astype(str)
    year = pd.to_numeric(out["year"], errors="coerce")
    hit = pd.Series(False, index=out.index)
    for code, y0, y1 in EMPL_X4_WINDOWS:
        hit |= (iso == code) & year.between(y0, y1)
    use = hit & interp.notna() & (interp > 0) & gdp.notna() & (gdp > 0)
    n = int(use.sum())
    if n:
        out.loc[use, "realGDP_empl"] = gdp[use] / (interp[use] * 1000.0)
    return out, n


def load_panel(path: Path | None = None) -> pd.DataFrame:
    path = Path(path) if path is not None else RAW
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    raw = pd.read_csv(path)
    need = {"year", "quarter", "country", "iso3", "realGDP_empl"}
    missing = need - set(raw.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    if "GDPrealsa" not in raw.columns or "empl_interp" not in raw.columns:
        raise ValueError(
            f"{path} needs GDPrealsa and empl_interp to patch SVK/ISR employment."
        )
    raw, n_patch = _patch_empl_x4(raw)

    df = raw.loc[:, ["iso3", "country", "year", "quarter", "realGDP_empl"]].copy()
    df = df.rename(columns={"iso3": "country", "country": "country_name"})
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["quarter"] = pd.to_numeric(df["quarter"], errors="coerce")
    df["realGDP_empl"] = pd.to_numeric(df["realGDP_empl"], errors="coerce")
    bad_q = df["quarter"].notna() & ~df["quarter"].isin((1, 2, 3, 4))
    if bad_q.any():
        raise ValueError(
            f"quarter must be in {{1,2,3,4}}; "
            f"got {sorted(df.loc[bad_q, 'quarter'].unique())}."
        )
    df["time"] = df["year"] + (df["quarter"] - 1.0) / 4.0
    df["period"] = (
        df["year"].astype("Int64").astype(str)
        + "-Q"
        + df["quarter"].astype("Int64").astype(str)
    )
    pos = df["realGDP_empl"].notna() & (df["realGDP_empl"] > 0)
    df = df.loc[pos].copy()
    df["y"] = np.log(df["realGDP_empl"].astype(float))
    df = df.dropna(subset=["country", "time", "y"])
    df = df.sort_values(["country", "time"], kind="mergesort").reset_index(drop=True)
    dup = df.duplicated(["country", "time"])
    if dup.any():
        raise ValueError(
            f"{int(dup.sum())} duplicate country-time rows after cleaning."
        )
    out = df[
        [
            "country",
            "country_name",
            "year",
            "quarter",
            "time",
            "period",
            "realGDP_empl",
            "y",
        ]
    ]
    out.attrs["n_empl_x4_patched"] = n_patch
    return out


def coverage(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby(["country", "country_name"], sort=True)
        .agg(
            n=("y", "size"),
            t0=("period", "min"),
            t1=("period", "max"),
        )
        .reset_index()
    )
    return g.sort_values("n", ascending=False).reset_index(drop=True)


def main() -> None:
    df = load_panel(RAW)
    cov = coverage(df)
    n_patch = int(df.attrs.get("n_empl_x4_patched", 0))
    print(
        f"SOE sample: log real GDP per worker (realGDP_empl)\n"
        f"  countries={df.country.nunique()}  obs={len(df)}  "
        f"{df.period.min()}–{df.period.max()}\n"
        f"  empl x4 patch (SVK 2000-01, ISR 2017): {n_patch} quarters"
    )
    print(cov.to_string(index=False))
    df.to_csv(OUT, index=False)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
