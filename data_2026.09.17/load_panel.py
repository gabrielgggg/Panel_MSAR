"""Load the 2026-09-17 seasonally adjusted real GDP per worker panel."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CSV = HERE / "realGDP_sa_empl.csv"
YCOL = "realGDPsa_usd_pa_empl"


def load_panel(path: Path | None = None) -> pd.DataFrame:
    path = Path(path) if path is not None else CSV
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    raw = pd.read_csv(path)
    need = {"iso3", "t", YCOL}
    missing = need - set(raw.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    df = raw.copy()
    df["iso3"] = df["iso3"].astype(str).str.strip()
    parsed = df["t"].astype(str).str.extract(
        r"(?P<year>\d{4})q(?P<quarter>[1-4])", flags=re.I
    )
    df["year"] = pd.to_numeric(parsed["year"], errors="coerce")
    df["quarter"] = pd.to_numeric(parsed["quarter"], errors="coerce")
    df["time"] = df["year"] + (df["quarter"] - 1) / 4.0
    df["period"] = [
        f"{int(y)}-Q{int(q)}" if pd.notna(y) and pd.notna(q) else ""
        for y, q in zip(df["year"], df["quarter"])
    ]
    df[YCOL] = pd.to_numeric(df[YCOL], errors="coerce")
    pos = df[YCOL].notna() & (df[YCOL] > 0)
    df = df.loc[pos].copy()
    df["y"] = np.log(df[YCOL].astype(float))
    if "country" in df.columns:
        df["country_name"] = df["country"].astype(str)
    else:
        df["country_name"] = df["iso3"]
    df["country"] = df["iso3"]
    df = df.dropna(subset=["country", "time", "y"])
    df = df.sort_values(["country", "time"], kind="mergesort").reset_index(drop=True)
    dup = df.duplicated(["country", "time"])
    if dup.any():
        raise ValueError(
            f"{int(dup.sum())} duplicate country-time rows after cleaning."
        )
    keep = [
        c
        for c in (
            "country",
            "country_name",
            "iso3",
            "year",
            "quarter",
            "time",
            "period",
            "t",
            YCOL,
            "y",
        )
        if c in df.columns
    ]
    return df.loc[:, keep]
