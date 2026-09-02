"""Demo OECD: fit the joint panel MS-AR on quarterly log real GDP per worker."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from panel_msar import PanelMSAR

DATA = Path(__file__).resolve().parent / "data" / "demo_oecd_gdp_per_worker_q.csv"


def load_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python data/fetch_demo_oecd.py"
        )
    df = pd.read_csv(path)
    need = {"country", "time"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    if "y" in df.columns:
        df = df.copy()
    elif "gdp_per_worker" in df.columns:
        df = df.copy()
        df["y"] = np.log(df["gdp_per_worker"].astype(float))
    else:
        raise ValueError("Need column y or gdp_per_worker.")
    df = df.dropna(subset=["country", "time", "y"])
    return df


def main():
    df = load_panel(DATA)
    print(
        f"Demo OECD: real GDP per worker (log)\n"
        f"  countries={df.country.nunique()}  obs={len(df)}  "
        f"{df.period.min() if 'period' in df.columns else ''}–"
        f"{df.period.max() if 'period' in df.columns else ''}"
    )

    out_pdf = Path(__file__).resolve().parent / "data" / "demo_oecd_log_gdp_per_worker_detrended.pdf"
    mod = PanelMSAR(
        n_regimes=3,
        common_rho=True,
        common_sigma=False,
        country_intercepts=False,
        country_trends=False,
        min_t=24,
    )
    res = mod.fit(
        df["country"],
        df["time"],
        df["y"],
        n_starts=8,
        maxiter=400,
        seed=1,
        compute_se=True,
        store_filtered=True,
        verbose=True,
        detrend_pdf=str(out_pdf),
    )
    print()
    print(res)
    print(f"\nDetrended PDF: {out_pdf}")


if __name__ == "__main__":
    main()
