#!/usr/bin/env python3
"""
MariettaDataV4_Backfill.py

Backfills missing COMMUNITY / CRIME / CLIMATE (and school) data
for properties already in MariettaDataV4.csv.

Key design:

- We DO NOT re-pull sale snapshots.
- We ONLY target rows whose community fields are completely empty
  (i.e., no community_geoIdV4, crime_index, or weather_index).
  This avoids wasting calls on the first ~1000 rows that already
  have community data.

- For each such row (up to MAX_BACKFILL_PER_RUN):
    * Get attom_id from the row.
    * Call:
        - /propertyapi/v1.0.0/property/detailwithschools
        - /propertyapi/v1.0.0/property/detail        (fallback for geoIdV4)
        - /v4/neighborhood/community                 (crime, climate, etc.)
    * Fill school_* and community_* fields if they are empty.

- Writes to MariettaDataV4_backfill.csv so the original file
  remains unchanged. Once you're happy with the result, you can
  manually replace the original if you want.

Usage:
    python MariettaDataV4_Backfill.py
"""

import os
import sys
import csv
import time
from typing import Any, Dict, Optional, List

import requests

# ----------------- Config -----------------

API_KEY = os.getenv("ATTOM_API_KEY", "7dc417b696f3970f0b8d293ac7ef1493")

BASE = "https://api.gateway.attomdata.com"
HEADERS = {
    "accept": "application/json",
    "apikey": API_KEY,
    "ApiKey": API_KEY,
}

INPUT_CSV = "MariettaDataV4.csv"
OUTPUT_CSV = "MariettaDataV4_backfill.csv"

# How many rows to backfill per run (tune this based on your free-trial limits)
MAX_BACKFILL_PER_RUN = 500

TIMEOUT = 30
PAUSE = 0.20
MAX_RETRIES = 6
BACKOFF_BASE = 0.8

# Columns we might fill for schools / community
SCHOOL_COLS = [
    "school_district_name",
    "school_district_geoIdV4",
    "num_schools_assigned",
    "nearest_school_name",
    "nearest_school_type",
    "nearest_school_distance_miles",
    "nearest_school_rating",
    "nearest_school_gs_test_rating",
    "avg_assigned_school_gs_test_rating",
]

COMMUNITY_COLS = [
    "community_geoIdV4",
    "community_geography_name",
    "population",
    "population_density_sq_mi",
    "median_family_income",
    "avg_household_income",
    "household_income_per_capita",
    "crime_index",
    "murder_index",
    "burglary_index",
    "motor_vehicle_theft_index",
    "air_pollution_index",
    "ozone_index",
    "annual_avg_temp",
    "annual_precip_in",
    "tornado_index",
    "earthquake_index",
    "weather_index",
]


# ----------------- HTTP Helper -----------------


def _normalize_msg(s: str) -> str:
    s = (s or "").lower().strip()
    return s.replace(" ", "").replace("-", "").replace("_", "")


def http_get_json(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Robust helper for GET requests against ATTOM.
    """
    url = f"{BASE}{path}"
    retries = 0

    while True:
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=TIMEOUT)
            text = r.text

            try:
                data = r.json()
            except Exception:
                print(f"[ERROR] Non-JSON response from {url}: {text[:400]!r}",
                      file=sys.stderr)
                return {}

            if 200 <= r.status_code < 300:
                time.sleep(PAUSE)
                return data or {}

            status = data.get("status") or {}
            msg = _normalize_msg(status.get("msg", ""))

            # ATTOM convention for "no results but not an error"
            if r.status_code == 400 and "successwithoutresult" in msg:
                time.sleep(PAUSE)
                return data or {}

            if r.status_code in (429, 500, 502, 503, 504) and retries < MAX_RETRIES:
                sleep_s = BACKOFF_BASE * (2 ** retries)
                print(f"[WARN] HTTP {r.status_code} for {url} – retrying in {sleep_s:.2f}s",
                      file=sys.stderr)
                time.sleep(sleep_s)
                retries += 1
                continue

            print(f"[ERROR] HTTP {r.status_code} for {url} params={params} "
                  f"body={text[:400]!r}",
                  file=sys.stderr)
            return {}

        except requests.RequestException as e:
            if retries < MAX_RETRIES:
                sleep_s = BACKOFF_BASE * (2 ** retries)
                print(f"[WARN] Network error for {url}: {e!r} – retrying in {sleep_s:.2f}s",
                      file=sys.stderr)
                time.sleep(sleep_s)
                retries += 1
                continue
            print(f"[ERROR] Network error for {url} after retries: {e!r}", file=sys.stderr)
            return {}


def _safe_get_property(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Given an ATTOM payload with a 'property' array, return the first property.
    """
    if not isinstance(data, dict):
        return {}
    props = data.get("property") or []
    if isinstance(props, list):
        return props[0] if props else {}
    if isinstance(props, dict):
        return props
    return {}


# ----------------- Backfill helpers -----------------


def get_attom_id_from_row(row: Dict[str, Any]) -> Optional[str]:
    """
    Extract attom ID from a CSV row. Supports both 'attom_id' and 'attomId'.
    """
    attom_id = row.get("attom_id") or row.get("attomId")
    if attom_id is None:
        return None
    attom_id = str(attom_id).strip()
    return attom_id or None


def row_needs_backfill(row: Dict[str, Any]) -> bool:
    """
    Decide if this row needs backfill.

    We ONLY target rows whose community block is *completely* empty.
    Specifically, if all of these are empty:
        - community_geoIdV4
        - crime_index
        - weather_index

    That corresponds to the long block of rows after ~1157 in your CSV.
    """
    comm_id = (row.get("community_geoIdV4") or "").strip()
    crime = (row.get("crime_index") or "").strip()
    weather = (row.get("weather_index") or "").strip()

    return comm_id == "" and crime == "" and weather == ""


_COMMUNITY_CACHE: Dict[str, Dict[str, Any]] = {}
_SCHOOLS_CACHE: Dict[str, Dict[str, Any]] = {}
_DETAIL_CACHE: Dict[str, Dict[str, Any]] = {}


def get_schools_detail(attom_id: str) -> Dict[str, Any]:
    if attom_id in _SCHOOLS_CACHE:
        return _SCHOOLS_CACHE[attom_id]
    data = http_get_json("/propertyapi/v1.0.0/property/detailwithschools", {"id": attom_id})
    prop = _safe_get_property(data)
    _SCHOOLS_CACHE[attom_id] = prop
    return prop


def get_detail(attom_id: str) -> Dict[str, Any]:
    if attom_id in _DETAIL_CACHE:
        return _DETAIL_CACHE[attom_id]
    data = http_get_json("/propertyapi/v1.0.0/property/detail", {"id": attom_id})
    prop = _safe_get_property(data)
    _DETAIL_CACHE[attom_id] = prop
    return prop


def get_community(geo_id_v4: str) -> Dict[str, Any]:
    if not geo_id_v4:
        return {}
    if geo_id_v4 in _COMMUNITY_CACHE:
        return _COMMUNITY_CACHE[geo_id_v4]
    data = http_get_json("/v4/neighborhood/community", {"geoIdV4": geo_id_v4})
    _COMMUNITY_CACHE[geo_id_v4] = data.get("community") or {}
    return _COMMUNITY_CACHE[geo_id_v4]


def backfill_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Backfill ONE row in-place (returns updated row dict).
    We:
        - Get geoIdV4 + school info via detailwithschools
        - Fallback to property/detail for geoIdV4 if needed
        - Use neighborhood/community to populate crime/climate/demographics
    """
    attom_id = get_attom_id_from_row(row)
    if not attom_id:
        return row  # can't do anything without an ID

    def empty_or_missing(col: str) -> bool:
        return (col not in row) or row[col] is None or str(row[col]).strip() == ""

    def fill_if_empty(col: str, value: Any) -> None:
        if col not in row:
            return
        if empty_or_missing(col) and value is not None:
            row[col] = value

    # -------- 1) Schools + geoIdV4 via detailwithschools --------
    zip_geoIdV4: Optional[str] = None

    p_sch = get_schools_detail(attom_id)
    if p_sch:
        school_district = p_sch.get("schoolDistrict") or {}
        fill_if_empty("school_district_name",
                      school_district.get("districtname") or school_district.get("districtName"))
        fill_if_empty("school_district_geoIdV4", school_district.get("geoIdV4"))

        loc = p_sch.get("location") or {}
        geo_map = loc.get("geoIdV4") or {}
        if isinstance(geo_map, dict):
            zip_geoIdV4 = geo_map.get("ZI") or geo_map.get("PZ")

        schools = p_sch.get("school") or []
        if isinstance(schools, dict):
            schools = [schools]

        if isinstance(schools, list) and schools:
            fill_if_empty("num_schools_assigned", len(schools))

            # nearest school by distance
            best_school = None
            best_dist = None
            for s in schools:
                dist_raw = s.get("distance")
                try:
                    d = float(dist_raw) if dist_raw is not None else None
                except Exception:
                    d = None
                if d is not None and (best_dist is None or d < best_dist):
                    best_dist = d
                    best_school = s

            if best_school is not None:
                fill_if_empty("nearest_school_name", best_school.get("InstitutionName"))
                fill_if_empty("nearest_school_type", best_school.get("Filetypetext"))
                fill_if_empty("nearest_school_distance_miles", best_dist)
                fill_if_empty("nearest_school_rating", best_school.get("schoolRating"))
                fill_if_empty("nearest_school_gs_test_rating", best_school.get("GSTestRating"))

            ratings: List[float] = []
            for s in schools:
                val = s.get("GSTestRating")
                try:
                    f = float(val)
                    if f > 0:
                        ratings.append(f)
                except Exception:
                    continue
            if ratings:
                avg_rating = sum(ratings) / float(len(ratings))
                fill_if_empty("avg_assigned_school_gs_test_rating", avg_rating)

    # -------- 2) Fallback geoIdV4 from property/detail if needed --------
    needs_community = any(empty_or_missing(c) for c in COMMUNITY_COLS)

    if needs_community and not zip_geoIdV4:
        p_det = get_detail(attom_id)
        loc_det = p_det.get("location") or {}
        geo_map_det = loc_det.get("geoIdV4") or {}
        if isinstance(geo_map_det, dict):
            zip_geoIdV4 = geo_map_det.get("ZI") or geo_map_det.get("PZ")

    # -------- 3) Community / crime / climate --------
    if needs_community and zip_geoIdV4:
        comm = get_community(zip_geoIdV4)
        geography = comm.get("geography") or {}
        dem = comm.get("demographics") or {}
        crime = comm.get("crime") or {}
        airq = comm.get("airQuality") or {}
        climate = comm.get("climate") or {}
        nat = comm.get("naturalDisasters") or {}

        fill_if_empty("community_geoIdV4", geography.get("geoIdV4"))
        fill_if_empty("community_geography_name", geography.get("geographyName"))

        fill_if_empty("population", dem.get("population"))
        fill_if_empty("population_density_sq_mi", dem.get("population_Density_Sq_Mi"))
        fill_if_empty("median_family_income", dem.get("family_Median_Income"))
        fill_if_empty("avg_household_income", dem.get("avg_Household_Income"))
        fill_if_empty("household_income_per_capita",
                      dem.get("household_Income_Per_Capita"))

        fill_if_empty("crime_index", crime.get("crime_Index"))
        fill_if_empty("murder_index", crime.get("murder_Index"))
        fill_if_empty("burglary_index", crime.get("burglary_Index"))
        fill_if_empty("motor_vehicle_theft_index",
                      crime.get("motor_Vehicle_Theft_Index"))

        fill_if_empty("air_pollution_index", airq.get("air_Pollution_Index"))
        fill_if_empty("ozone_index", airq.get("ozone_Index"))

        fill_if_empty("annual_avg_temp", climate.get("annual_Avg_Temp"))
        fill_if_empty("annual_precip_in", climate.get("annual_Precip_In"))

        fill_if_empty("tornado_index", nat.get("tornado_Index"))
        fill_if_empty("earthquake_index", nat.get("earthquake_Index"))
        fill_if_empty("weather_index", nat.get("weather_Index"))

    return row


# ----------------- Main -----------------


def main() -> None:
    print(f"Using ATTOM API key: {API_KEY!r}", file=sys.stderr)
    if not API_KEY or API_KEY == "YOUR_API_KEY_HERE":
        print("[ERROR] Please set ATTOM_API_KEY before running.", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(INPUT_CSV):
        print(f"[ERROR] Input CSV {INPUT_CSV!r} not found.", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Reading from {INPUT_CSV} and writing to {OUTPUT_CSV}",
          file=sys.stderr)
    print(f"[INFO] MAX_BACKFILL_PER_RUN = {MAX_BACKFILL_PER_RUN}", file=sys.stderr)

    backfilled_count = 0
    total_rows = 0

    with open(INPUT_CSV, mode="r", newline="", encoding="utf-8") as fin, \
         open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as fout:

        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames
        if not fieldnames:
            print("[ERROR] No header row in input CSV.", file=sys.stderr)
            sys.exit(1)

        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            total_rows += 1

            if backfilled_count < MAX_BACKFILL_PER_RUN and row_needs_backfill(row):
                attom_id = get_attom_id_from_row(row)
                print(
                    f"[INFO] Backfilling row {total_rows} attomId={attom_id} "
                    f"(backfilled so far: {backfilled_count})",
                    file=sys.stderr,
                )
                try:
                    row = backfill_row(row)
                    backfilled_count += 1
                except Exception as e:
                    print(
                        f"[ERROR] Failed to backfill row {total_rows}: {e!r}",
                        file=sys.stderr,
                    )

            # Write row (patched or not)
            writer.writerow(row)

    print(
        f"[DONE] Processed {total_rows} rows, backfilled {backfilled_count} rows. "
        f"Output: {OUTPUT_CSV}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
