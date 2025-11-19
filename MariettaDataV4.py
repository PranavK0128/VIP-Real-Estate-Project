#!/usr/bin/env python3
"""
MariettaDataV4.py

Goal:
- Build up an enriched property dataset for Marietta, GA over multiple runs
  *without* reprocessing properties we've already collected.

Key ideas:
- We use /propertyapi/v1.0.0/sale/snapshot as the "seed list" of properties.
- For each new property, we call the richer endpoints (detail, assessment,
  sales history, schools, community).
- We write rows incrementally to MariettaDataV4.csv so we can resume across runs.
- On each run:
    1. Load all existing attom_ids from MariettaDataV4.csv.
    2. Call /sale/snapshot per ZIP, ordered by SaleSearchDate DESC.
    3. For each page:
         * Keep only properties whose attom_id is NOT already in the CSV.
         * If a page has 0 new attom_ids, we assume all later (older) pages
           are already covered and stop paging that ZIP.
    4. Enrich ONLY the new attom_ids and append them to the CSV.

This way:
- First run: pulls everything in these ZIPs.
- Later runs: usually only touch the first page per ZIP to pick up new sales.
"""

import os
import sys
import csv
import time
from typing import Any, Dict, List, Optional, Set

import requests

# ----------------------- Config -----------------------

API_KEY = os.getenv("ATTOM_API_KEY", "b7fb03e3d1459021572ee6ce3916f7d9")

BASE = "https://api.gateway.attomdata.com"
HEADERS = {
    "accept": "application/json",
    "apikey": API_KEY,
    "ApiKey": API_KEY,
}

# Marietta/Cobb ZIPs
MARIETTA_ZIPS = ["30060", "30062", "30064", "30066", "30067", "30068"]

PAGE_SIZE = 100
TIMEOUT = 30
PAUSE = 0.20
MAX_RETRIES = 6
BACKOFF_BASE = 0.8

OUTPUT_CSV = "MariettaDataV4.csv"

# ----------------- HTTP + Numeric Helpers -----------------


def _normalize_msg(s: str) -> str:
    s = (s or "").lower().strip()
    return s.replace(" ", "").replace("-", "").replace("_", "")


def http_get_json(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Robust helper for GET requests against ATTOM.

    - Retries with exponential backoff on 429/5xx and network errors.
    - Treats ATTOM 400 'SuccessWithoutResult' as an empty-ish payload.
    - Returns {} if a hard error occurs instead of crashing the whole script.
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

            # Hard error – log and return empty
            print(f"[ERROR] HTTP {r.status_code} for {url} with params={params} "
                  f"body={text[:400]!r}", file=sys.stderr)
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


def _first_num(value: Any) -> Optional[float]:
    """Return value as a positive float if it looks numeric; otherwise None."""
    if isinstance(value, (int, float)):
        return float(value) if value > 0 else None
    try:
        s = str(value).replace(",", "").strip()
        if not s:
            return None
        parts = s.split(".")
        if len(parts) > 2:
            return None
        for p in parts:
            if p and not p.isdigit():
                return None
        num = float(s)
        return num if num > 0 else None
    except Exception:
        return None


def _deep_find_numeric(obj: Any, key_pred) -> Optional[float]:
    """
    Recursively search dict/list looking for a numeric value whose key satisfies key_pred.
    Returns the first matching positive numeric value.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            if key_pred(str(k)):
                num = _first_num(v)
                if num is not None:
                    return num
            found = _deep_find_numeric(v, key_pred)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _deep_find_numeric(item, key_pred)
            if found is not None:
                return found
    return None


def extract_sale_price(sale: Dict[str, Any]) -> Optional[float]:
    """
    Try multiple shapes for sale amount/price commonly seen across ATTOM payloads.
    Falls back to a deep search for keys including 'sale' & ('amt'|'amount'|'price').
    """
    if not isinstance(sale, dict):
        return _first_num(sale)

    candidates: List[Any] = []

    for key in ("saleAmt", "saleAmount", "salePrice", "price"):
        if key in sale:
            candidates.append(sale.get(key))

    amount = sale.get("amount")
    if isinstance(amount, dict):
        for key in ("saleAmt", "saleAmount", "price", "amount"):
            if key in amount:
                candidates.append(amount.get(key))

    for cand in candidates:
        num = _first_num(cand)
        if num is not None:
            return num

    def pred(k: str) -> bool:
        k = k.lower()
        return ("sale" in k) and ("amt" in k or "amount" in k or "price" in k)

    return _deep_find_numeric(sale, pred)


# ----------------- CSV helpers & ID helpers -----------------


def append_row(csv_path: str, fieldnames: List[str], row: Dict[str, Any]) -> None:
    """
    Append a single row to csv_path. If the file does not exist yet, write the header first.
    """
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists or os.path.getsize(csv_path) == 0:
            writer.writeheader()
        trimmed = {k: row.get(k) for k in fieldnames}
        writer.writerow(trimmed)


def load_existing_attom_ids(csv_path: str) -> Set[str]:
    """
    Load all existing attom IDs from an existing CSV, if present.
    Supports either 'attom_id' or 'attomId' column names.
    """
    existing: Set[str] = set()
    if not os.path.isfile(csv_path):
        return existing

    with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            att = row.get("attom_id") or row.get("attomId")
            if att:
                existing.add(str(att).strip())

    return existing


def get_attom_id_from_seed(seed_rec: Dict[str, Any]) -> Optional[str]:
    """
    Given a /sale/snapshot property record, extract attomId as a string.
    """
    if not isinstance(seed_rec, dict):
        return None

    ident = seed_rec.get("identifier") or {}
    attom_id = (
        ident.get("attomId")
        or ident.get("attomid")
        or ident.get("Id")
        or ident.get("id")
    )

    if attom_id is None:
        return None

    return str(attom_id)


# ----------------- Field list & flattening -----------------


FIELDNAMES = [
    # Identification & location
    "attom_id",
    "apn",
    "fips",
    "address_oneLine",
    "address_line1",
    "city",
    "state",
    "zip",
    "latitude",
    "longitude",
    # Sale outcome (label)
    "sale_date",
    "sale_price",
    # Property characteristics
    "property_type",
    "building_arch_style",
    "building_levels",
    "building_quality",
    "sqft",
    "beds",
    "baths_total",
    "lot_size_sqft",
    "has_pool",
    "year_built",
    # Assessment & tax
    "assessed_total_value",
    "assessed_land_value",
    "assessed_improvement_value",
    "market_total_value",
    "tax_amount",
    "tax_year",
    # Sales history
    "num_sales_history",
    "last_sale_date",
    "last_sale_amount",
    "prev_sale_date",
    "prev_sale_amount",
    # School features
    "school_district_name",
    "school_district_geoIdV4",
    "num_schools_assigned",
    "nearest_school_name",
    "nearest_school_type",
    "nearest_school_distance_miles",
    "nearest_school_rating",
    "nearest_school_gs_test_rating",
    "avg_assigned_school_gs_test_rating",
    # Community / neighborhood features
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


def _safe_get_property(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Given an ATTOM payload with a 'property' array, return the first element or {}.
    """
    if not isinstance(data, dict):
        return {}
    props = data.get("property") or []
    if isinstance(props, list):
        return props[0] if props else {}
    if isinstance(props, dict):
        return props
    return {}


def flatten_property(seed_rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    For one sale snapshot record (from /sale/snapshot), call additional endpoints and
    assemble a single flat feature dict suitable for ML.
    """
    row: Dict[str, Any] = {k: None for k in FIELDNAMES}

    address = seed_rec.get("address") or {}
    ident = seed_rec.get("identifier") or {}
    location = seed_rec.get("location") or {}
    summary = seed_rec.get("summary") or {}
    building = seed_rec.get("building") or {}
    rooms = building.get("rooms") if isinstance(building, dict) else {}
    size = building.get("size") if isinstance(building, dict) else {}
    lot = seed_rec.get("lot") or {}
    sale = seed_rec.get("sale") or {}

    attom_id = get_attom_id_from_seed(seed_rec)

    # Basic ID & location
    row["attom_id"] = attom_id
    row["apn"] = ident.get("apn")
    row["fips"] = ident.get("fips")
    row["address_oneLine"] = address.get("oneLine")
    row["address_line1"] = address.get("line1") or address.get("line")
    row["city"] = address.get("locality") or address.get("city")
    row["state"] = address.get("countrySubd") or address.get("state")
    row["zip"] = (
        address.get("postal1")
        or address.get("postalCode")
        or address.get("postalcode")
    )
    row["latitude"] = location.get("latitude") if isinstance(location, dict) else None
    row["longitude"] = (
        location.get("longitude") if isinstance(location, dict) else None
    )

    # Property characteristics from snapshot
    row["property_type"] = (
        summary.get("propertyType")
        or summary.get("propsubtype")
        or summary.get("propclass")
    )
    row["sqft"] = (size or {}).get("universalsize")
    row["beds"] = (rooms or {}).get("beds")
    row["baths_total"] = (rooms or {}).get("bathstotal")
    row["year_built"] = summary.get("yearbuilt") or summary.get("yearBuilt")
    row["lot_size_sqft"] = lot.get("lotsize2")

    pool = lot.get("pooltype")
    if isinstance(pool, str):
        row["has_pool"] = 0 if pool.strip().upper() in {"N", "NO", "NONE"} else 1
    elif pool is not None:
        row["has_pool"] = 1
    else:
        row["has_pool"] = None

    # Sale outcome (label)
    row["sale_date"] = sale.get("saleTransDate") or sale.get("saleSearchDate")
    row["sale_price"] = extract_sale_price(sale)

    # --- property/detail for richer building attributes ---
    if attom_id:
        detail_data = http_get_json(
            "/propertyapi/v1.0.0/property/detail", {"id": attom_id}
        )
        p_det = _safe_get_property(detail_data)
        if p_det:
            b = p_det.get("building") or {}
            b_sum = b.get("summary") or {}
            b_rooms = b.get("rooms") or {}
            b_size = b.get("size") or {}
            lot_det = p_det.get("lot") or {}

            row["sqft"] = row["sqft"] or b_size.get("universalsize")
            row["beds"] = row["beds"] or b_rooms.get("beds")
            row["baths_total"] = row["baths_total"] or b_rooms.get("bathstotal")
            row["year_built"] = (
                row["year_built"]
                or b_sum.get("yearbuilt")
                or b_sum.get("yearBuilt")
            )
            row["lot_size_sqft"] = row["lot_size_sqft"] or lot_det.get("lotsize2")

            row["building_arch_style"] = b_sum.get("archStyle")
            row["building_levels"] = b_sum.get("levels")
            row["building_quality"] = b_sum.get("quality")

    # --- assessment/detail for tax & assessed values ---
    if attom_id:
        ass_data = http_get_json(
            "/propertyapi/v1.0.0/assessment/detail", {"id": attom_id}
        )
        p_ass = _safe_get_property(ass_data)
        if p_ass:
            assessment = p_ass.get("assessment") or {}
            assd = assessment.get("assessed") or {}
            market = assessment.get("market") or {}
            tax = assessment.get("tax") or {}

            row["assessed_total_value"] = assd.get("assdttlvalue")
            row["assessed_land_value"] = assd.get("assdlandvalue")
            row["assessed_improvement_value"] = assd.get("assdimprvalue")
            row["market_total_value"] = market.get("mktttlvalue")
            row["tax_amount"] = tax.get("taxamt")
            row["tax_year"] = tax.get("taxyear")

    # --- saleshistory/detail for prior sales ---
    if attom_id:
        sh_data = http_get_json(
            "/propertyapi/v1.0.0/saleshistory/detail", {"id": attom_id}
        )
        p_sh = _safe_get_property(sh_data)
        sale_hist = p_sh.get("salehistory") or []
        if isinstance(sale_hist, dict):
            sale_hist = [sale_hist]

        if isinstance(sale_hist, list):
            def _sale_date_key(h: Dict[str, Any]) -> str:
                return (
                    h.get("saleTransDate")
                    or h.get("saleDate")
                    or ((h.get("amount") or {}).get("salerecdate"))
                    or ""
                )

            sale_hist_sorted = sorted(
                sale_hist, key=_sale_date_key, reverse=True
            )
            row["num_sales_history"] = len(sale_hist_sorted)

            if sale_hist_sorted:
                last = sale_hist_sorted[0]
                row["last_sale_date"] = (
                    last.get("saleTransDate")
                    or last.get("saleDate")
                    or ((last.get("amount") or {}).get("salerecdate"))
                )
                row["last_sale_amount"] = extract_sale_price(last)

            if len(sale_hist_sorted) >= 2:
                prev = sale_hist_sorted[1]
                row["prev_sale_date"] = (
                    prev.get("saleTransDate")
                    or prev.get("saleDate")
                    or ((prev.get("amount") or {}).get("salerecdate"))
                )
                row["prev_sale_amount"] = extract_sale_price(prev)

    # --- property/detailwithschools for school district + assigned schools ---
    zip_geoIdV4 = None
    if attom_id:
        schools_data = http_get_json(
            "/propertyapi/v1.0.0/property/detailwithschools", {"id": attom_id}
        )
        p_sch = _safe_get_property(schools_data)
        if p_sch:
            school_district = p_sch.get("schoolDistrict") or {}
            row["school_district_name"] = (
                school_district.get("districtname")
                or school_district.get("districtName")
            )
            row["school_district_geoIdV4"] = school_district.get("geoIdV4")

            loc = p_sch.get("location") or {}
            geo_map = loc.get("geoIdV4") or {}
            if isinstance(geo_map, dict):
                zip_geoIdV4 = geo_map.get("ZI") or geo_map.get("PZ")

            schools = p_sch.get("school") or []
            if isinstance(schools, dict):
                schools = [schools]

            if isinstance(schools, list):
                row["num_schools_assigned"] = len(schools)

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
                    row["nearest_school_name"] = best_school.get("InstitutionName")
                    row["nearest_school_type"] = best_school.get("Filetypetext")
                    row["nearest_school_distance_miles"] = best_dist
                    row["nearest_school_rating"] = best_school.get("schoolRating")
                    row["nearest_school_gs_test_rating"] = best_school.get(
                        "GSTestRating"
                    )

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
                    row["avg_assigned_school_gs_test_rating"] = (
                        sum(ratings) / float(len(ratings))
                    )

    if zip_geoIdV4 is None and isinstance(location, dict):
        geo_map = location.get("geoIdV4") or {}
        if isinstance(geo_map, dict):
            zip_geoIdV4 = geo_map.get("ZI") or geo_map.get("PZ")

    # --- neighborhood/community (demographics, crime, climate) ---
    if zip_geoIdV4:
        comm_data = http_get_json(
            "/v4/neighborhood/community", {"geoIdV4": zip_geoIdV4}
        )
        community = comm_data.get("community") or {}
        geography = community.get("geography") or {}
        dem = community.get("demographics") or {}
        crime = community.get("crime") or {}
        airq = community.get("airQuality") or {}
        climate = community.get("climate") or {}
        nat = community.get("naturalDisasters") or {}

        row["community_geoIdV4"] = geography.get("geoIdV4")
        row["community_geography_name"] = geography.get("geographyName")

        row["population"] = dem.get("population")
        row["population_density_sq_mi"] = dem.get("population_Density_Sq_Mi")
        row["median_family_income"] = dem.get("family_Median_Income")
        row["avg_household_income"] = dem.get("avg_Household_Income")
        row["household_income_per_capita"] = dem.get(
            "household_Income_Per_Capita"
        )

        row["crime_index"] = crime.get("crime_Index")
        row["murder_index"] = crime.get("murder_Index")
        row["burglary_index"] = crime.get("burglary_Index")
        row["motor_vehicle_theft_index"] = crime.get(
            "motor_Vehicle_Theft_Index"
        )

        row["air_pollution_index"] = airq.get("air_Pollution_Index")
        row["ozone_index"] = airq.get("ozone_Index")

        row["annual_avg_temp"] = climate.get("annual_Avg_Temp")
        row["annual_precip_in"] = climate.get("annual_Precip_In")

        row["tornado_index"] = nat.get("tornado_Index")
        row["earthquake_index"] = nat.get("earthquake_Index")
        row["weather_index"] = nat.get("weather_Index")

    return row


# ----------------- Fetch only NEW seeds per run -----------------


def fetch_new_sale_seeds(existing_ids: Set[str]) -> List[Dict[str, Any]]:
    """
    For each Marietta ZIP:

    - Call /sale/snapshot ordered by SaleSearchDate DESC.
    - For each page:
        * Identify which properties have attom_ids NOT in existing_ids.
        * Add those to the seed list.
        * If a page has 0 new attom_ids, assume all older pages are already
          covered and stop paging for that ZIP.

    On the very first run (existing_ids is empty), this will walk all pages and
    return all properties. On later runs, we usually only touch the first page.
    """
    new_seeds: List[Dict[str, Any]] = []

    for zp in MARIETTA_ZIPS:
        page = 1
        while True:
            params = {
                "postalcode": zp,
                "pagesize": PAGE_SIZE,
                "page": page,
                "orderBy": "SaleSearchDate desc",
            }
            data = http_get_json("/propertyapi/v1.0.0/sale/snapshot", params)
            props = (data.get("property") or []) if isinstance(data, dict) else []

            if isinstance(props, dict):
                props = [props]

            if not props:
                print(
                    f"[INFO] ZIP={zp} page={page} – no properties returned, "
                    f"stop paging this ZIP.",
                    file=sys.stderr,
                )
                break

            new_this_page: List[Dict[str, Any]] = []
            for p in props:
                attom_id = get_attom_id_from_seed(p)
                if attom_id is None:
                    continue
                if attom_id in existing_ids:
                    continue
                new_this_page.append(p)

            print(
                f"[INFO] ZIP={zp} page={page} – got {len(props)} props, "
                f"{len(new_this_page)} new",
                file=sys.stderr,
            )

            new_seeds.extend(new_this_page)

            # If the entire page was already known, we assume everything
            # "behind" it (older SaleSearchDate) is already processed.
            if len(new_this_page) == 0:
                break

            # If the API returned fewer than PAGE_SIZE, that was the last page
            # for this ZIP.
            if len(props) < PAGE_SIZE:
                break

            page += 1

    return new_seeds


# ----------------- Main -----------------


def main() -> None:
    print(f"Using ATTOM API key: {API_KEY!r}", file=sys.stderr)
    if not API_KEY or API_KEY == "YOUR_API_KEY_HERE":
        print(
            "[ERROR] Please set the ATTOM_API_KEY environment variable before running.",
            file=sys.stderr,
        )
        sys.exit(1)

    existing_ids = load_existing_attom_ids(OUTPUT_CSV)
    print(
        f"[INFO] Loaded {len(existing_ids)} existing attomIds from {OUTPUT_CSV}",
        file=sys.stderr,
    )

    seed_recs = fetch_new_sale_seeds(existing_ids)

    if not seed_recs:
        print(
            "[DONE] No new sale snapshots found in these Marietta ZIPs. "
            "Everything returned by /sale/snapshot is already in the CSV.",
            file=sys.stderr,
        )
        return

    processed_new = 0

    for rec in seed_recs:
        attom_id = get_attom_id_from_seed(rec)
        if attom_id is None:
            print("[WARN] Seed record missing attomId; skipping.", file=sys.stderr)
            continue

        if attom_id in existing_ids:
            # Shouldn't happen because we filter in fetch_new_sale_seeds, but be safe.
            continue

        try:
            row = flatten_property(rec)
            append_row(OUTPUT_CSV, FIELDNAMES, row)
            existing_ids.add(attom_id)
            processed_new += 1
            print(
                f"[INFO] Added new property attomId={attom_id}. "
                f"Total new this run: {processed_new}",
                file=sys.stderr,
            )
        except Exception as e:
            print(
                f"[ERROR] Failed to process property attomId={attom_id}: {e!r}",
                file=sys.stderr,
            )

    print(
        f"[DONE] Added {processed_new} new rows to {OUTPUT_CSV}", file=sys.stderr
    )


if __name__ == "__main__":
    main()
