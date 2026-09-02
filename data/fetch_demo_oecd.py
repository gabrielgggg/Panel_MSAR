"""Demo OECD: download quarterly real GDP and employment (Economic Outlook SDMX)
and write an unbalanced country panel of real GDP per worker.

GDP is chain-linked volume in USD at constant exchange rates (not PPP).
Employment is total employment, LFS basis, persons. Ratio = real GDP / worker.

Source: OECD Economic Outlook No. 118, SDMX dataflow OECD.ECO.MAD:DSD_EO@DF_EO(1.4).
Cite: OECD (2026), OECD Economic Outlook No. 118.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import urllib.request

URL = (
    "https://sdmx.oecd.org/public/rest/data/"
    "OECD.ECO.MAD,DSD_EO@DF_EO,1.4/.GDPV_USD+ET.Q"
    "?endPeriod=2025-Q4&format=csvfilewithlabels"
)
DROP = {"OECD", "EA17", "EA", "EU", "G7", "G20"}
OUT_DIR = Path(__file__).resolve().parent


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "panel-msar"})
    with urllib.request.urlopen(req, timeout=120) as r, dest.open("wb") as f:
        f.write(r.read())
    return dest


def build_panel(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw[~raw.REF_AREA.isin(DROP)].copy()
    raw = raw[raw.REF_AREA.str.fullmatch(r"[A-Z]{3}")]
    wide = (
        raw.pivot_table(
            index=["REF_AREA", "Reference area", "TIME_PERIOD"],
            columns="MEASURE",
            values="OBS_VALUE",
            aggfunc="first",
        )
        .reset_index()
        .rename(
            columns={
                "REF_AREA": "country",
                "Reference area": "country_name",
                "TIME_PERIOD": "period",
                "GDPV_USD": "gdp_real_usd",
                "ET": "emp",
            }
        )
    )
    wide = wide.dropna(subset=["gdp_real_usd", "emp"])
    wide = wide[(wide.gdp_real_usd > 0) & (wide.emp > 0)]
    yq = wide["period"].str.extract(r"(\d{4})-Q(\d)")
    wide["year"] = yq[0].astype(int)
    wide["quarter"] = yq[1].astype(int)
    wide["time"] = wide["year"] + (wide["quarter"] - 1) / 4.0
    wide["gdp_per_worker"] = wide["gdp_real_usd"] / wide["emp"]
    wide["y"] = np.log(wide["gdp_per_worker"])
    cols = [
        "country",
        "country_name",
        "year",
        "quarter",
        "time",
        "period",
        "gdp_real_usd",
        "emp",
        "gdp_per_worker",
        "y",
    ]
    return wide[cols].sort_values(["country", "time"]).reset_index(drop=True)


def main():
    raw_path = download(URL, OUT_DIR / "demo_oecd_eo_raw.csv")
    raw = pd.read_csv(raw_path)
    panel = build_panel(raw)
    out = OUT_DIR / "demo_oecd_gdp_per_worker_q.csv"
    panel.to_csv(out, index=False)
    print(
        f"wrote {out}  N={panel.country.nunique()} countries  "
        f"{len(panel)} obs  {panel.period.min()}–{panel.period.max()}"
    )


if __name__ == "__main__":
    main()
