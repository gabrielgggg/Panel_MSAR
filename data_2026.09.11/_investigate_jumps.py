"""Detect and diagnose sharp jumps in log real GDP per worker."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE / "realGDP_nsa_empl.csv"

KNOWN_REDENOM = [
    # (iso3, year, scale, note) year is calendar year of reform (approx)
    ("ARG", 1992, 10000, "austral to peso (1e4)"),
    ("BRA", 1986, 1000, "cruzado"),
    ("BRA", 1989, 1000, "cruzado novo"),
    ("BRA", 1990, 1, "cruzeiro (name)"),
    ("BRA", 1993, 1000, "cruzeiro real"),
    ("BRA", 1994, 2750, "real (2750 cruzeiros reais)"),
    ("BOL", 1987, 1e6, "boliviano (1e6 pesos)"),
    ("PER", 1985, 1000, "inti"),
    ("PER", 1991, 1e6, "nuevo sol"),
    ("MEX", 1993, 1000, "nuevo peso"),
    ("POL", 1995, 10000, "new zloty"),
    ("ROU", 2005, 10000, "new leu"),
    ("TUR", 2005, 1e6, "new lira"),
    ("BGR", 1999, 1000, "new lev"),
    ("RUS", 1998, 1000, "redenomination"),
    ("UKR", 1996, 100000, "hryvnia"),
    ("BLR", 2000, 1000, "redenomination"),
    ("BLR", 2016, 10000, "redenomination"),
    ("GHA", 2007, 10000, "new cedi"),
    ("ZMB", 2013, 1000, "kwacha"),
    ("VEN", 2008, 1000, "bolivar fuerte"),
    ("VEN", 2018, 100000, "bolivar soberano"),
    ("VEN", 2021, 1e6, "digital bolivar"),
    ("ECU", 2000, None, "dollarization"),
    ("PLW", None, None, None),
]


def load():
    raw = pd.read_csv(RAW)
    m = raw["t"].astype(str).str.extract(r"(?P<ty>\d{4})q(?P<tq>[1-4])", flags=re.I)
    raw["year_t"] = pd.to_numeric(m["ty"])
    raw["quarter_t"] = pd.to_numeric(m["tq"])
    raw["time"] = raw["year_t"] + (raw["quarter_t"] - 1) / 4.0
    raw["period"] = [
        f"{int(y)}-Q{int(q)}" for y, q in zip(raw["year_t"], raw["quarter_t"])
    ]
    for c in [
        "GDP",
        "GDP_usd_pa",
        "empl",
        "realGDP_usd_pa_empl",
        "NER_pa_base",
        "realGDP_empl",
        "empl_q",
        "empl_interp",
        "empl_annual",
    ]:
        if c in raw.columns:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")
    df = raw.loc[
        raw["realGDP_usd_pa_empl"].notna() & (raw["realGDP_usd_pa_empl"] > 0)
    ].copy()
    df["y"] = np.log(df["realGDP_usd_pa_empl"].astype(float))
    df = df.sort_values(["iso3", "time"]).reset_index(drop=True)
    return df


def add_deltas(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("iso3", sort=False)
    df = df.copy()
    df["dy"] = g["y"].diff()
    df["dtime"] = g["time"].diff()
    df["dlog_GDP"] = g["GDP"].transform(lambda s: np.log(s.where(s > 0)).diff())
    df["dlog_usd"] = g["GDP_usd_pa"].transform(lambda s: np.log(s.where(s > 0)).diff())
    df["dlog_empl"] = g["empl"].transform(lambda s: np.log(s.where(s > 0)).diff())
    df["dlog_ner"] = g["NER_pa_base"].transform(lambda s: np.log(s.where(s > 0)).diff())
    df["dlog_local_pw"] = g["realGDP_empl"].transform(
        lambda s: np.log(s.where(s > 0)).diff()
    )
    df["prev_source"] = g["source"].shift(1)
    df["prev_base"] = g["baseyear"].shift(1)
    df["prev_empl_src"] = g["empl_source"].shift(1)
    df["prev_er"] = g["er_iso3"].shift(1)
    df["prev_period"] = g["period"].shift(1)
    df["prev_y"] = g["y"].shift(1)
    df["prev_gdp"] = g["GDP"].shift(1)
    df["prev_usd"] = g["GDP_usd_pa"].shift(1)
    df["prev_empl"] = g["empl"].shift(1)
    df["prev_ner"] = g["NER_pa_base"].shift(1)
    df["prev_local_pw"] = g["realGDP_empl"].shift(1)
    return df


def nearest_factor(x):
    if not np.isfinite(x) or x == 0:
        return np.nan, np.nan
    scale = np.exp(abs(x))
    candidates = np.array(
        [10.0, 100.0, 1000.0, 10000.0, 1e5, 1e6, 2750.0, 1e9], dtype=float
    )
    ratios = np.maximum(scale / candidates, candidates / scale)
    i = int(np.argmin(ratios))
    return float(candidates[i]), float(ratios[i])


def classify(row) -> str:
    dy = row["dy"]
    dner = row["dlog_ner"]
    dgdp = row["dlog_GDP"]
    dempl = row["dlog_empl"]
    dusd = row["dlog_usd"]
    dloc = row["dlog_local_pw"]
    notes = []
    src_chg = (
        pd.notna(row["source"])
        and pd.notna(row["prev_source"])
        and row["source"] != row["prev_source"]
    )
    base_chg = (
        pd.notna(row["baseyear"])
        and pd.notna(row["prev_base"])
        and row["baseyear"] != row["prev_base"]
    )
    empl_chg = (
        pd.notna(row["empl_source"])
        and pd.notna(row["prev_empl_src"])
        and row["empl_source"] != row["prev_empl_src"]
    )
    if src_chg:
        notes.append(f"source {row['prev_source']} -> {row['source']}")
    if base_chg:
        notes.append(f"baseyear {int(row['prev_base'])} -> {int(row['baseyear'])}")
    if empl_chg:
        notes.append(f"empl_source {row['prev_empl_src']} -> {row['empl_source']}")

    ner_fac, ner_r = nearest_factor(dner) if pd.notna(dner) else (np.nan, np.nan)
    gdp_fac, gdp_r = nearest_factor(dgdp) if pd.notna(dgdp) else (np.nan, np.nan)

    redenom_ner = pd.notna(ner_r) and ner_r < 1.15 and ner_fac >= 10
    redenom_gdp = pd.notna(gdp_r) and gdp_r < 1.15 and gdp_fac >= 10

    if redenom_ner and redenom_gdp and abs(dner - dgdp) < 0.2:
        notes.append(
            f"matched local-GDP and NER scale ~{ner_fac:g} (USD series should be stable)"
        )
        if abs(dy) > 0.2:
            notes.append("USD/worker still jumps: conversion not internally consistent")
    elif redenom_ner and not redenom_gdp:
        notes.append(f"NER scale break ~{ner_fac:g} without matching local-GDP break")
    elif redenom_gdp and not redenom_ner:
        notes.append(f"local-GDP scale break ~{gdp_fac:g} without matching NER break")
    elif redenom_ner and redenom_gdp:
        notes.append(
            f"NER~{ner_fac:g} and GDP~{gdp_fac:g} but scales do not match"
        )

    if pd.notna(dempl) and abs(dempl) > 0.25 and abs(dempl) > 0.5 * abs(dy):
        notes.append(f"employment jump dlog={dempl:.2f}")
    if pd.notna(dusd) and abs(dusd) > 0.25:
        notes.append(f"USD GDP jump dlog={dusd:.2f}")
    if pd.notna(dloc) and abs(dloc) > 0.25:
        notes.append(f"local real GDP/worker jump dlog={dloc:.2f}")
    if pd.notna(dy) and pd.notna(dusd) and pd.notna(dempl):
        if abs(dy - (dusd - dempl)) > 0.05:
            notes.append("y-change not equal to dlog USD GDP minus dlog empl")
    if not notes:
        if abs(dy) > 0.5:
            notes.append("unexplained level break in USD GDP per worker")
        else:
            notes.append("large but not round-scale; possible crisis/revision/seasonal")
    return "; ".join(notes)


def main():
    df = add_deltas(load())
    cons = df["dtime"].between(0.24, 0.26)
    j = df.loc[cons & df["dy"].notna()].copy()
    j["abs_dy"] = j["dy"].abs()
    print("consecutive QoQ", len(j))
    print(j["abs_dy"].quantile([0.5, 0.9, 0.95, 0.99, 0.995, 0.999, 1]).to_string())
    print(
        "counts",
        {t: int((j.abs_dy > t).sum()) for t in (0.15, 0.2, 0.3, 0.5, 1.0, 2.0)},
    )

    # Visual jumps: |dy|>=0.25 (~28% QoQ) is already extreme for productivity.
    cuts = j.loc[j["abs_dy"] >= 0.25].copy()
    cuts["cause"] = cuts.apply(classify, axis=1)
    cuts["pct"] = 100 * (np.exp(cuts["dy"]) - 1)
    cols = [
        "iso3",
        "country",
        "prev_period",
        "period",
        "dy",
        "pct",
        "dlog_GDP",
        "dlog_ner",
        "dlog_usd",
        "dlog_empl",
        "dlog_local_pw",
        "GDP",
        "prev_gdp",
        "NER_pa_base",
        "prev_ner",
        "GDP_usd_pa",
        "prev_usd",
        "empl",
        "prev_empl",
        "source",
        "prev_source",
        "baseyear",
        "prev_base",
        "empl_source",
        "cause",
    ]
    cuts = cuts.sort_values(["abs_dy"], ascending=False)
    out = cuts[cols]
    path = HERE / "z_jumps.csv"
    out.to_csv(path, index=False)
    print(f"\nWrote {path}  n={len(out)}")

    print("\n=== JUMPS |dlog y| >= 0.25 (sorted by size) ===")
    for _, r in out.iterrows():
        print(
            f"{r.iso3:3} {r.country:20} {r.prev_period} -> {r.period}  "
            f"dy={r.dy:+.3f} ({r.pct:+.1f}%)  "
            f"dGDP={_fmt(r.dlog_GDP)} dNER={_fmt(r.dlog_ner)} "
            f"dUSD={_fmt(r.dlog_usd)} dE={_fmt(r.dlog_empl)}"
        )
        print(f"     {r.cause}")

    print("\n=== country max |dy| (all consecutive) top 30 ===")
    mx = (
        j.groupby(["iso3", "country"], as_index=False)["abs_dy"]
        .max()
        .sort_values("abs_dy", ascending=False)
    )
    print(mx.head(30).to_string(index=False))


def _fmt(x):
    if pd.isna(x):
        return "   NA"
    return f"{x:+6.2f}"


if __name__ == "__main__":
    main()
