"""Load soe_sample/realGDP_empl.csv into the PanelMSAR (country, time, y) layout.

y is log real GDP per worker (column realGDP_empl). Time is year-fraction
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


def load_panel(path: Path | None = None) -> pd.DataFrame:
    path = Path(path) if path is not None else RAW
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    raw = pd.read_csv(path)
    need = {"year", "quarter", "country", "iso3", "realGDP_empl"}
    missing = need - set(raw.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")

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
    return df[
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
    print(
        f"SOE sample: log real GDP per worker (realGDP_empl)\n"
        f"  countries={df.country.nunique()}  obs={len(df)}  "
        f"{df.period.min()}–{df.period.max()}"
    )
    print(cov.to_string(index=False))
    df.to_csv(OUT, index=False)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
