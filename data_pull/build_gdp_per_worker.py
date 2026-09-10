"""CLI: download sources if needed, construct the quarterly GDP-per-worker panel.

Usage (from this folder)::

    python build_gdp_per_worker.py
    python build_gdp_per_worker.py --force-download

Writes:

* ``output/gdp_per_worker_q.csv`` — country, period, gdp_per_worker, metadata
  (plus optional audit columns).
* ``output/coverage.csv`` — one row per country with sample span and sources.
* ``output/construction_log.txt`` — counts of countries by GDP SA/NSA and
  employment source.

All methodology, URLs, and formulas are in ``README.md`` and in the module
docstring of ``construct.py``. This file only orchestrates download → build
→ write.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from construct import build_panel, coverage_table, read_raw
from download_sources import ensure_raw

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "output"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("Usage")[0].strip())
    p.add_argument(
        "--force-download",
        action="store_true",
        help="re-download IMF/ILO extracts even if raw/ already exists",
    )
    args = p.parse_args(argv)

    ensure_raw(force=args.force_download)
    frames = read_raw(HERE / "raw")
    panel = build_panel(
        qnea_sa=frames["qnea_sa"],
        qnea_nsa=frames["qnea_nsa"],
        anea_real=frames["anea_real"],
        anea_nom=frames["anea_nom"],
        er_annual=frames["er_annual"],
        ilo_official_q=frames["ilo_official_q"],
        ilo_official_a=frames["ilo_official_a"],
        ilo_modelled_a=frames["ilo_modelled_a"],
        country_names=frames["country_names"],
    )
    cov = coverage_table(panel)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel_path = OUT_DIR / "gdp_per_worker_q.csv"
    cov_path = OUT_DIR / "coverage.csv"
    log_path = OUT_DIR / "construction_log.txt"
    panel.to_csv(panel_path, index=False)
    cov.to_csv(cov_path, index=False)

    n_c = int(panel["country"].nunique())
    n_obs = len(panel)
    t0, t1 = panel["period"].min(), panel["period"].max()
    sa_flag = panel.groupby("country")["gdp_sa"].first()
    sa_c = int((sa_flag == "SA").sum())
    sarimax_c = int((sa_flag == "SARIMAX").sum())
    nsa_c = int((sa_flag == "NSA").sum())
    emp_kind = (
        panel.groupby("country")["emp_source"]
        .first()
        .map(lambda s: str(s).split(";")[0].strip())
        .value_counts()
        .to_string()
    )
    log = (
        f"real GDP per worker, annualized, {panel_path.name}\n"
        f"units: constant 2015 USD at 2015 market FX (not PPP)\n"
        f"countries={n_c}  obs={n_obs}  {t0}–{t1}\n"
        f"GDP seasonal adjustment (country-level): "
        f"SA={sa_c}  SARIMAX={sarimax_c}  NSA={nsa_c}\n"
        f"employment source (country-level, first token):\n{emp_kind}\n"
        f"wrote {panel_path}\n"
        f"wrote {cov_path}\n"
    )
    log_path.write_text(log, encoding="utf-8")
    print(log, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
