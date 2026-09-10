"""Download the raw source files used to build quarterly real GDP per worker.

This module does **not** transform the data. It only fetches the published
extracts and writes them under ``raw/``. Re-run with ``--force`` to refresh.

Sources and exact URLs
----------------------
IMF SDMX 3.0 (CSV, ``Accept: text/csv``), agency ``IMF.STA``:

* Quarterly National Accounts (QNEA), dataflow ``QNEA``.
  Key order: ``COUNTRY.INDICATOR.PRICE_TYPE.S_ADJUSTMENT.TYPE_OF_TRANSFORMATION.FREQUENCY``.
  Indicator ``B1GQ`` = GDP. ``Q`` = constant prices, ``XDC`` = domestic currency,
  ``Q`` frequency. Seasonal adjustment ``SA`` or ``NSA``.

  SA volumes::

    https://api.imf.org/external/sdmx/3.0/data/dataflow/IMF.STA/QNEA/~/*.B1GQ.Q.SA.XDC.Q

  NSA volumes::

    https://api.imf.org/external/sdmx/3.0/data/dataflow/IMF.STA/QNEA/~/*.B1GQ.Q.NSA.XDC.Q

  Dataset page: https://data.imf.org/en/datasets/IMF.STA:QNEA

* Annual National Accounts (ANEA), dataflow ``ANEA``.
  Key order: ``COUNTRY.INDICATOR.PRICE_TYPE.TYPE_OF_TRANSFORMATION.FREQUENCY``.

  Constant-price GDP, domestic currency, annual::

    https://api.imf.org/external/sdmx/3.0/data/dataflow/IMF.STA/ANEA/~/*.B1GQ.Q.XDC.A

  Current-price GDP, domestic currency, annual::

    https://api.imf.org/external/sdmx/3.0/data/dataflow/IMF.STA/ANEA/~/*.B1GQ.V.XDC.A

  Dataset page: https://data.imf.org/en/datasets/IMF.STA:ANEA

* Exchange Rates (ER), dataflow ``ER``.
  Key order: ``COUNTRY.INDICATOR.TYPE_OF_TRANSFORMATION.FREQUENCY``.
  ``XDC_USD`` = domestic currency per US dollar; ``PA_RT`` = period average.

    https://api.imf.org/external/sdmx/3.0/data/dataflow/IMF.STA/ER/~/*.XDC_USD.PA_RT.A

  Dataset page: https://data.imf.org/en/datasets/IMF.STA:ER

* IMF country names (codelist ``CL_COUNTRY``)::

    https://api.imf.org/external/sdmx/3.0/structure/codelist/IMF/CL_COUNTRY

ILOSTAT bulk CSV (Rplumber indicator extracts):

* Official employment by sex and age, persons, **quarterly**
  (``EMP_TEMP_SEX_AGE_NB_Q``)::

    https://rplumber.ilo.org/data/indicator/?id=EMP_TEMP_SEX_AGE_NB_Q

* Official employment by sex and age, persons, **annual**
  (``EMP_TEMP_SEX_AGE_NB_A``)::

    https://rplumber.ilo.org/data/indicator/?id=EMP_TEMP_SEX_AGE_NB_A

* ILO modelled estimates, employment by sex and age, **annual**
  (``EMP_2EMP_SEX_AGE_NB_A``)::

    https://rplumber.ilo.org/data/indicator/?id=EMP_2EMP_SEX_AGE_NB_A

ILOSTAT documentation: https://ilostat.ilo.org/data/
ILO SDMX guide: https://www.ilo.org/resource/other/ilostat-sdmx-user-guide
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "raw"

# Polite, identifying UA. Some ILO endpoints reject empty/generic agents.
UA = "regime-switch-panel-data-pull/1.0 (research; real GDP per worker panel)"

IMF_HEADERS = {"User-Agent": UA, "Accept": "text/csv"}
ILO_HEADERS = {"User-Agent": UA, "Accept": "*/*"}
JSON_HEADERS = {"User-Agent": UA, "Accept": "application/json"}

# Each record: dest filename under raw/, URL, HTTP headers.
DOWNLOADS: tuple[tuple[str, str, dict[str, str]], ...] = (
    (
        "imf_qnea_b1gq_q_sa_xdc_q.csv",
        "https://api.imf.org/external/sdmx/3.0/data/dataflow/IMF.STA/QNEA/~/*.B1GQ.Q.SA.XDC.Q",
        IMF_HEADERS,
    ),
    (
        "imf_qnea_b1gq_q_nsa_xdc_q.csv",
        "https://api.imf.org/external/sdmx/3.0/data/dataflow/IMF.STA/QNEA/~/*.B1GQ.Q.NSA.XDC.Q",
        IMF_HEADERS,
    ),
    (
        "imf_anea_b1gq_q_xdc_a.csv",
        "https://api.imf.org/external/sdmx/3.0/data/dataflow/IMF.STA/ANEA/~/*.B1GQ.Q.XDC.A",
        IMF_HEADERS,
    ),
    (
        "imf_anea_b1gq_v_xdc_a.csv",
        "https://api.imf.org/external/sdmx/3.0/data/dataflow/IMF.STA/ANEA/~/*.B1GQ.V.XDC.A",
        IMF_HEADERS,
    ),
    (
        "imf_er_xdc_usd_pa_rt_a.csv",
        "https://api.imf.org/external/sdmx/3.0/data/dataflow/IMF.STA/ER/~/*.XDC_USD.PA_RT.A",
        IMF_HEADERS,
    ),
    (
        "ilo_emp_temp_sex_age_nb_q.csv",
        "https://rplumber.ilo.org/data/indicator/?id=EMP_TEMP_SEX_AGE_NB_Q",
        ILO_HEADERS,
    ),
    (
        "ilo_emp_temp_sex_age_nb_a.csv",
        "https://rplumber.ilo.org/data/indicator/?id=EMP_TEMP_SEX_AGE_NB_A",
        ILO_HEADERS,
    ),
    (
        "ilo_emp_2emp_sex_age_nb_a.csv",
        "https://rplumber.ilo.org/data/indicator/?id=EMP_2EMP_SEX_AGE_NB_A",
        ILO_HEADERS,
    ),
)

CL_COUNTRY_URL = (
    "https://api.imf.org/external/sdmx/3.0/structure/codelist/IMF/CL_COUNTRY"
)


def download_file(
    url: str,
    dest: Path,
    headers: dict[str, str],
    *,
    timeout: int = 600,
    retries: int = 4,
    chunk_bytes: int = 1024 * 1024,
) -> Path:
    """Stream ``url`` to ``dest`` via a ``.part`` file; retry on transient errors.

    The file is replaced atomically (``Path.replace``) only after a complete
    read, so a killed download cannot leave a truncated CSV that later looks
    like a successful cache.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp, tmp.open("wb") as f:
                while True:
                    chunk = resp.read(chunk_bytes)
                    if not chunk:
                        break
                    f.write(chunk)
            tmp.replace(dest)
            return dest
        except (urllib.error.URLError, TimeoutError, OSError) as err:
            last_err = err
            if tmp.exists():
                tmp.unlink()
            wait = 2 ** attempt
            print(f"  retry {attempt}/{retries} after {err!r}; sleeping {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"failed to download {url} -> {dest}") from last_err


def download_country_codelist(dest: Path, *, force: bool = False) -> Path:
    """Write ``iso3,country_name`` from IMF ``CL_COUNTRY`` (JSON structure API)."""
    if dest.exists() and not force:
        return dest
    req = urllib.request.Request(CL_COUNTRY_URL, headers=JSON_HEADERS)
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    codes = payload["data"]["codelists"][0]["codes"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["iso3", "country_name"])
        for item in codes:
            w.writerow([item.get("id", ""), item.get("name", "")])
    return dest


def ensure_raw(*, force: bool = False) -> list[Path]:
    """Download every source listed in ``DOWNLOADS`` plus the country codelist."""
    written: list[Path] = []
    for filename, url, headers in DOWNLOADS:
        dest = RAW_DIR / filename
        if dest.exists() and not force:
            print(f"skip (exists) {dest.name}  {dest.stat().st_size:,} bytes", flush=True)
            written.append(dest)
            continue
        print(f"download {dest.name}\n  {url}", flush=True)
        download_file(url, dest, headers)
        print(f"  wrote {dest.stat().st_size:,} bytes", flush=True)
        written.append(dest)
    names = RAW_DIR / "imf_cl_country.csv"
    if names.exists() and not force:
        print(f"skip (exists) {names.name}", flush=True)
    else:
        print(f"download {names.name}\n  {CL_COUNTRY_URL}", flush=True)
        download_country_codelist(names, force=True)
        print(f"  wrote {names.stat().st_size:,} bytes", flush=True)
    written.append(names)
    return written


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("This module")[0].strip())
    p.add_argument("--force", action="store_true", help="re-download even if raw/ files exist")
    args = p.parse_args(argv)
    ensure_raw(force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
