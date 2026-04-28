#!/usr/bin/env python3
"""
Simple downloader for Sabadell Parc Agrari (Sbd_nord_daily) and Vacarisses (Vacarisses_daily).
Downloads per-day data from Meteocat HTML pages and writes two CSVs under Raw Data.
"""
from __future__ import annotations

import datetime as dt
import io
import os
import requests
import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

START_SAB = dt.date(2008, 5, 1)
END_SAB = dt.date(2026, 1, 31)
START_VAC = dt.date(1996, 2, 16)
END_VAC = dt.date(2026, 1, 31)

PRIMARY_BASE = "https://www.meteocat.cat/observacions/xema/dades?codi={code}&dia={day}T00:00Z"
SAB_CODE = "XF"  # Sabadell Parc Agrari
VAC_CODE = "D2"  # Vacarisses

OUT_SAB = os.path.join("Raw Data", "Sabadell_data.csv")
OUT_VAC = os.path.join("Raw Data", "Vacarisses_data.csv")
FAILED_SAB = os.path.join("Raw Data", "Sabadell_failed_dates.csv")
FAILED_VAC = os.path.join("Raw Data", "Vacarisses_failed_dates.csv")

HEADERS = {
    "PeríodeTU": "period_utc",
    "TM°C": "temp_mean (°C)",
    "TX°C": "temp_max (°C)",
    "TN°C": "temp_min (°C)",
    "HRM%": "humidity (%)",
    "PPTmm": "rain_mm",
    "VVM (10 m)km/h": "wind_mean_10m (km/h)",
    "DVM (10 m)graus": "wind_dir_10m (degrees)",
    "VVX (10 m)km/h": "wind_max_10m (km/h)",
    "PMhPa": "pressure (hPa)",
    "RSW/m2": "radiation (W/m²)"
}

def parse_day(code: str, day: dt.date):
    day_str = day.strftime("%Y-%m-%d")
    url = PRIMARY_BASE.format(code=code, day=day_str)
    # Use a session with a User-Agent header to mimic a browser request
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    try:
        resp = session.get(url, timeout=20)
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text))
        if len(tables) < 3:
            return day, None, f"Less than 3 tables at {url}"
        df = tables[2].copy()
        df = df.rename(columns={k: HEADERS[k] for k in df.columns if k in HEADERS})
        df = df.assign(time_utc=df["period_utc"].astype(str).str.split(" - ").str[0])
        df = df.assign(datetime_utc=pd.to_datetime(day_str + " " + df["time_utc"], errors="coerce"))
        df = df.dropna(subset=["datetime_utc"]).copy()
        df = df.drop(columns=["period_utc", "time_utc"])
        df = df.set_index("datetime_utc")
        numeric_cols = [c for c in df.columns if c not in {"date","year","month","day","station","datetime_utc"}]
        for c in numeric_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df["station"] = code
        return day, df, None
    except Exception as e:
        return day, None, str(e)

def download_station(code: str, start: dt.date, end: dt.date, out_csv: str, fail_csv: str, label: str) -> tuple:
    days = []
    d = start
    while d <= end:
        days.append((code, d))
        d += dt.timedelta(days=1)
    dfs = []
    fails = []
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=12) as ex:
        futures = {ex.submit(parse_day, code, day): day for code, day in days}
        for fut in as_completed(futures):
            day = futures[fut]
            day_dt, df, err = fut.result()
            if df is not None:
                dfs.append(df)
            else:
                fails.append((day.strftime("%Y-%m-%d"), err))
    if dfs:
        final = pd.concat(dfs, ignore_index=True)
        final.to_csv(out_csv, index=False)
    else:
        final = pd.DataFrame()
    if fails:
        pd.DataFrame(fails, columns=["date","error"]).to_csv(fail_csv, index=False)
    return out_csv, final.shape[0], len(fails)

def main():
    os.makedirs("Raw Data", exist_ok=True)
    sab_out, sab_rows, sab_fails = download_station(SAB_CODE, START_SAB, END_SAB, OUT_SAB, FAILED_SAB, 'Sabadell')
    vac_out, vac_rows, vac_fails = download_station(VAC_CODE, START_VAC, END_VAC, OUT_VAC, FAILED_VAC, 'Vacarisses')
    print(f"Sabadell: {sab_rows} rows -> {sab_out}; fails: {sab_fails}")
    print(f"Vacarisses: {vac_rows} rows -> {vac_out}; fails: {vac_fails}")

if __name__ == '__main__':
    main()
