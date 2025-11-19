#!/usr/bin/env python3
"""
testATTOMEndpoints.py

Sample ~200 properties from MariettaDataV2.csv and test a wide range of ATTOM
API endpoints for each property. The goal is to see which endpoints actually
return useful data for the VIP Real Estate project.

Usage:
    python testATTOMEndpoints.py

Setup:
    1. Put this file in the same folder as MariettaDataV2.py and MariettaDataV2.csv
    2. Set your API key, e.g. on macOS/Linux:
           export ATTOM_API_KEY="your_real_key_here"
       or on Windows (PowerShell):
           $env:ATTOM_API_KEY="your_real_key_here"
    3. Run:
           python testATTOMEndpoints.py

Output:
    testATTOMEndpoints.csv – one row per sampled property, with:
      • Original columns from MariettaDataV2.csv
      • For each endpoint: *_has_data, *_http, *_attom_code, *_attom_msg

    NOTE:
    Rows are appended to testATTOMEndpoints.csv as they are processed, so if the
    script terminates early you still have partial results.
"""

import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from urllib.parse import quote

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# ATTOM API key – override via environment variable when running locally.
API_KEY = os.getenv("ATTOM_API_KEY", "b7fb03e3d1459021572ee6ce3916f7d9")

# Host (do not include trailing slash)
BASE_HOST = "https://api.gateway.attomdata.com"

# Input: existing CSV of Marietta properties (created by MariettaDataV2.py)
INPUT_CSV = "MariettaDataV2.csv"

# Output: CSV summarizing which endpoints return data per property
OUTPUT_CSV = "testATTOMEndpoints.csv"

# How many properties to test (take the first N rows from INPUT_CSV)
SAMPLE_SIZE = 5

# Networking / politeness
TIMEOUT = 30           # seconds per request
MAX_RETRIES = 3
PAUSE_BETWEEN_CALLS = 0.2  # seconds – keep small but non-zero to avoid rate limits

HEADERS = {
    "accept": "application/json",
    "apikey": API_KEY,
    "ApiKey": API_KEY,   # some examples use ApiKey capitalization
}

# ---------------------------------------------------------------------------
# Endpoint definitions
# ---------------------------------------------------------------------------

# Property-centric endpoints that can be queried directly by attomId
PROPERTY_ENDPOINTS_BY_ATTOM: Dict[str, str] = {
    # Property Search
    "property_id": "/propertyapi/v1.0.0/property/id",
    "property_snapshot": "/propertyapi/v1.0.0/property/snapshot",
    "property_basicprofile": "/propertyapi/v1.0.0/property/basicprofile",
    "property_detail": "/propertyapi/v1.0.0/property/detail",
    "property_expandedprofile": "/propertyapi/v1.0.0/property/expandedprofile",
    # Rich variants
    "property_detailwithschools": "/propertyapi/v1.0.0/property/detailwithschools",
    "property_detailowner": "/propertyapi/v1.0.0/property/detailowner",
    "property_detailmortgage": "/propertyapi/v1.0.0/property/detailmortgage",
    "property_detailmortgageowner": "/propertyapi/v1.0.0/property/detailmortgageowner",
    # Valuation & Equity
    "attomavm_detail": "/propertyapi/v1.0.0/attomavm/detail",
    "valuation_homeequity": "/propertyapi/v1.0.0/valuation/homeequity",
    "avm_snapshot": "/propertyapi/v1.0.0/avm/snapshot",
    # Assessment / taxes
    "assessment_detail": "/propertyapi/v1.0.0/assessment/detail",
    "assessment_snapshot": "/propertyapi/v1.0.0/assessment/snapshot",
    # Sales & history
    "saleshistory_snapshot": "/propertyapi/v1.0.0/saleshistory/snapshot",
    "saleshistory_detail": "/propertyapi/v1.0.0/saleshistory/detail",
    "sale_snapshot": "/propertyapi/v1.0.0/sale/snapshot",
    # Historical trends (property-level snapshot of trends)
    "salestrend_snapshot": "/propertyapi/v1.0.0/salestrend/snapshot",
}

# Endpoints that expect address or latitude/longitude rather than just attomId
SPECIAL_ENDPOINTS = {
    # Property Address search – use one-line address from CSV
    "property_address": "/propertyapi/v1.0.0/property/address",
    # Area / geocode hierarchy for a point
    "area_hierarchy_lookup": "/v4/area/hierarchy/lookup",
    # Neighborhood / community stats
    "neighborhood_community": "/v4/neighborhood/community",
    # Area boundary (GeoJSON)
    "area_boundary": "/v4/area/boundary",
    # Location lookup (city-level – called once per city/state, cached)
    "location_lookup": "/v4/location/lookup",
    # School search around a point
    "school_search": "/propertyapi/v1.0.0/school/search",
    # POI search around a property
    "neighborhood_poi": "/v4/neighborhood/poi",
    # Sales comparables – path includes address pieces
    "salescomparables_address": "/propertyapi/v1.0.0/salescomparables/address",
}


# ---------------------------------------------------------------------------
# HTTP + response helpers
# ---------------------------------------------------------------------------

def http_get_json(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make a GET request to ATTOM and return a small structured dict:

    {
        "ok": bool,            # True if HTTP 200 and JSON parsed
        "status_code": int,    # HTTP status code
        "data": dict | None,   # Parsed JSON on success
        "error": str | None,   # Error string if not ok
    }
    """
    url = BASE_HOST + path
    last_error: Optional[str] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=TIMEOUT)
        except requests.RequestException as exc:
            last_error = f"request_exception: {exc}"
            time.sleep(PAUSE_BETWEEN_CALLS * attempt)
            continue

        status_code = resp.status_code

        # Successful HTTP
        if status_code == 200:
            try:
                data = resp.json()
            except ValueError:
                return {
                    "ok": False,
                    "status_code": status_code,
                    "data": None,
                    "error": "invalid_json",
                }
            return {
                "ok": True,
                "status_code": status_code,
                "data": data,
                "error": None,
            }

        # ATTOM often returns helpful JSON bodies even on errors (403, 400, 429, etc.)
        try:
            data = resp.json()
        except ValueError:
            data = None

        # For 429 / 5xx, quick exponential backoff then retry
        if status_code in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES:
            last_error = f"http_{status_code}"
            time.sleep(PAUSE_BETWEEN_CALLS * attempt)
            continue

        # Non-retriable HTTP error
        msg = None
        if isinstance(data, dict):
            status_obj = data.get("status") or {}
            msg = (
                status_obj.get("msg")
                or status_obj.get("short_description")
                or status_obj.get("long_description")
            )
        return {
            "ok": False,
            "status_code": status_code,
            "data": data,
            "error": msg or f"http_{status_code}",
        }

    # Retries exhausted
    return {
        "ok": False,
        "status_code": None,
        "data": None,
        "error": last_error or "unknown_error",
    }


def summarize_attom_body(body: Optional[Dict[str, Any]]) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Inspect ATTOM JSON body and decide:
    - whether it actually has payload beyond status / metadata
    - what the ATTOM status.code and status.msg are (if present)

    Returns:
        (has_data, attom_code, attom_msg)
    """
    if not isinstance(body, dict):
        return False, None, None

    status = body.get("status") or {}
    attom_code = None
    attom_msg = None

    if isinstance(status, dict):
        attom_code = str(status.get("code")) if status.get("code") is not None else None
        attom_msg = (
            status.get("msg")
            or status.get("short_description")
            or status.get("long_description")
        )

    # Heuristic: any top-level key other than the usual metadata that has
    # a non-empty value counts as "has data".
    IGNORE_KEYS = {
        "status",
        "xmlns",
        "xml_record",
        "page",
        "pageSize",
        "pageCount",
        "pageNumber",
        "inputparameter",
        "metadata",
        "disclaimers",
        "copyright",
    }

    has_data = False
    for key, value in body.items():
        if key in IGNORE_KEYS:
            continue
        if value:
            has_data = True
            break

    return has_data, attom_code, attom_msg


def summarize_call(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert the low-level http_get_json result into flat fields that we can
    drop into the final CSV.
    """
    if not result["ok"]:
        return {
            "has_data": False,
            "http_status": result.get("status_code"),
            "attom_code": None,
            "attom_msg": result.get("error"),
        }

    body = result.get("data")
    has_data, attom_code, attom_msg = summarize_attom_body(body)
    return {
        "has_data": has_data,
        "http_status": result.get("status_code"),
        "attom_code": attom_code,
        "attom_msg": attom_msg,
    }


# ---------------------------------------------------------------------------
# Endpoint-specific helpers
# ---------------------------------------------------------------------------

def call_property_by_attom(path: str, attom_id: Any) -> Dict[str, Any]:
    """
    Many property/valuation/assessment/sales endpoints accept attomId directly.
    This helper wraps that pattern.
    """
    params = {"attomId": str(attom_id)}
    return http_get_json(path, params)


def call_property_address_search(path: str, one_line_address: str) -> Dict[str, Any]:
    """
    Property Address / BasicProfile examples in the ATTOM docs commonly use an
    'address' parameter carrying a one-line address string.
    """
    params = {"address": one_line_address}
    return http_get_json(path, params)


def call_area_hierarchy(lat: float, lon: float) -> Dict[str, Any]:
    """
    Call the V4 Area Hierarchy Lookup endpoint for a specific point. We ask
    for all geography types and then pick out the ones we care about (e.g. ZIP).
    """
    params = {
        "latitude": f"{lat:.8f}",
        "longitude": f"{lon:.8f}",
        "geoType": "All",  # return all overlapping geographies
    }
    return http_get_json(SPECIAL_ENDPOINTS["area_hierarchy_lookup"], params)


def extract_geo_ids_from_hierarchy(body: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """
    Given a successful Area Hierarchy response, pull out a ZIP-level and
    place-level geoIdV4 if present.

    Returns:
        (zip_geo_id_v4, place_geo_id_v4)
    """
    zip_geo: Optional[str] = None
    place_geo: Optional[str] = None

    try:
        items = (
            body.get("response", {})
            .get("result", {})
            .get("package", {})
            .get("item", [])
        )
    except Exception:
        items = []

    if not isinstance(items, list):
        items = [items]

    for item in items:
        if not isinstance(item, dict):
            continue
        gtype = item.get("type")
        gid_v4 = (
            item.get("geoIdV4")
            or item.get("geoidV4")
            or item.get("geoidv4")
            or item.get("id")
        )
        if not gid_v4:
            continue

        # ZIP / Postal code areas – use first we see
        if gtype in ("PZ", "ZI") and zip_geo is None:
            zip_geo = gid_v4
        # Place / city
        if gtype in ("PL", "CI") and place_geo is None:
            place_geo = gid_v4

    return zip_geo, place_geo


def call_neighborhood_community(geo_id_v4: str) -> Dict[str, Any]:
    params = {"geoidv4": geo_id_v4}
    return http_get_json(SPECIAL_ENDPOINTS["neighborhood_community"], params)


def call_area_boundary(geo_id_v4: str) -> Dict[str, Any]:
    params = {"geoIdV4": geo_id_v4, "format": "geojson"}
    return http_get_json(SPECIAL_ENDPOINTS["area_boundary"], params)


def call_location_lookup(city: str, state: str) -> Dict[str, Any]:
    """
    Location lookup returns canonical geocodes for a named place (e.g. city).
    """
    name = f"{city}, {state}"
    params = {
        "name": name,
        # CI = City; see Location Lookup docs
        "geographyTypeAbbreviation": "CI",
    }
    return http_get_json(SPECIAL_ENDPOINTS["location_lookup"], params)


def call_school_search(lat: float, lon: float) -> Dict[str, Any]:
    params = {
        "latitude": f"{lat:.8f}",
        "longitude": f"{lon:.8f}",
        "radius": 5,  # miles – chosen just to get "nearby" schools
    }
    return http_get_json(SPECIAL_ENDPOINTS["school_search"], params)


def call_neighborhood_poi(one_line_address: str) -> Dict[str, Any]:
    """
    Use V4 Neighborhood POI endpoint, searching by address within a small radius.
    """
    params = {
        "address": one_line_address,
        "radius": 2,         # miles
        "RecordLimit": 20,   # keep payloads modest
    }
    return http_get_json(SPECIAL_ENDPOINTS["neighborhood_poi"], params)


def call_salescomparables_address(
    street: str, city: str, county: str, state: str, postal_code: str
) -> Dict[str, Any]:
    """
    Sales Comparables by address has the path structure:

        /salescomparables/address/{street}/{city}/{county}/{state}/{zip}

    We URL-encode each segment before building the final path.
    """
    street_seg = quote(str(street).strip(), safe="")
    city_seg = quote(str(city).strip(), safe="")
    county_seg = quote(str(county).strip(), safe="")
    state_seg = quote(str(state).strip(), safe="")
    zip_seg = quote(str(postal_code).strip(), safe="")

    path = (
        SPECIAL_ENDPOINTS["salescomparables_address"]
        + f"/{street_seg}/{city_seg}/{county_seg}/{state_seg}/{zip_seg}"
    )
    # This endpoint encodes everything in the path, so no query params are required
    return http_get_json(path, params={})


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    if not API_KEY or API_KEY == "REPLACE_WITH_YOUR_KEY":
        print(
            "WARNING: ATTOM_API_KEY environment variable is not set.\n"
            "Set ATTOM_API_KEY before running this script, otherwise calls will fail.",
            file=sys.stderr,
        )

    if not os.path.exists(INPUT_CSV):
        print(f"ERROR: Input CSV '{INPUT_CSV}' not found.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)

    if df.empty:
        print("ERROR: Input CSV is empty.", file=sys.stderr)
        sys.exit(1)

    # Limit to SAMPLE_SIZE rows (simply take top N for reproducibility)
    sample_df = df.head(SAMPLE_SIZE).copy()
    print(f"Loaded {len(df)} rows from {INPUT_CSV}; testing first {len(sample_df)} rows.")

    results_rows: List[Dict[str, Any]] = []

    # Caches so we don't hammer some endpoints unnecessarily
    hierarchy_cache: Dict[Tuple[float, float], Dict[str, Any]] = {}
    community_cache: Dict[str, Dict[str, Any]] = {}
    boundary_cache: Dict[str, Dict[str, Any]] = {}
    location_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}

    # Assume all Marietta properties are in Cobb County – adjust if needed
    DEFAULT_COUNTY = "Cobb County"

    # If an old output file exists, remove it so we start fresh
    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)
    first_write = True  # controls header vs append

    for idx, (_, row) in enumerate(sample_df.iterrows()):
        attom_id = row.get("attomId")
        one_line = row.get("address_oneLine") or ""
        addr_line1 = row.get("address_line1") or ""
        city = row.get("city") or ""
        state = row.get("state") or ""
        postal = str(row.get("zip") or "").split(".")[0]  # handle floats like 30060.0

        try:
            lat = float(row.get("latitude"))
            lon = float(row.get("longitude"))
        except (TypeError, ValueError):
            lat = float("nan")
            lon = float("nan")

        print(f"[{idx+1}/{len(sample_df)}] Testing attomId={attom_id} at {one_line}")

        row_out: Dict[str, Any] = {}

        # Copy over the original columns so you can compare existing vs new data
        for col in df.columns:
            row_out[col] = row.get(col)

        # -----------------------------
        # Property endpoints by attomId
        # -----------------------------
        if attom_id is None or (isinstance(attom_id, float) and pd.isna(attom_id)):
            # If sample row somehow has no attomId, just mark as missing
            for key in PROPERTY_ENDPOINTS_BY_ATTOM:
                row_out[f"{key}_has_data"] = False
                row_out[f"{key}_http"] = None
                row_out[f"{key}_attom_code"] = None
                row_out[f"{key}_attom_msg"] = "no_attomId"
        else:
            for key, path in PROPERTY_ENDPOINTS_BY_ATTOM.items():
                result = call_property_by_attom(path, attom_id)
                summary = summarize_call(result)
                row_out[f"{key}_has_data"] = summary["has_data"]
                row_out[f"{key}_http"] = summary["http_status"]
                row_out[f"{key}_attom_code"] = summary["attom_code"]
                row_out[f"{key}_attom_msg"] = summary["attom_msg"]

                time.sleep(PAUSE_BETWEEN_CALLS)

        # -----------------------------
        # Property address search
        # -----------------------------
        if one_line:
            res_addr = call_property_address_search(
                SPECIAL_ENDPOINTS["property_address"], one_line
            )
            summary_addr = summarize_call(res_addr)
            row_out["property_address_has_data"] = summary_addr["has_data"]
            row_out["property_address_http"] = summary_addr["http_status"]
            row_out["property_address_attom_code"] = summary_addr["attom_code"]
            row_out["property_address_attom_msg"] = summary_addr["attom_msg"]

            time.sleep(PAUSE_BETWEEN_CALLS)
        else:
            row_out["property_address_has_data"] = False
            row_out["property_address_http"] = None
            row_out["property_address_attom_code"] = None
            row_out["property_address_attom_msg"] = "no_address"

        # -----------------------------
        # Area hierarchy, community, boundary
        # -----------------------------
        has_valid_point = not (pd.isna(lat) or pd.isna(lon))

        zip_geo_id_v4: Optional[str] = None
        place_geo_id_v4: Optional[str] = None

        if has_valid_point:
            key_pt = (round(lat, 6), round(lon, 6))
            if key_pt in hierarchy_cache:
                hier_result = hierarchy_cache[key_pt]
            else:
                hier_raw = call_area_hierarchy(lat, lon)
                hier_summary = summarize_call(hier_raw)
                # we also keep the raw data with it so we can extract geoIdV4
                hier_result = {
                    "summary": hier_summary,
                    "raw": hier_raw.get("data") if hier_raw.get("ok") else None,
                }
                hierarchy_cache[key_pt] = hier_result

            hier_summary = hier_result["summary"]
            row_out["area_hierarchy_has_data"] = hier_summary["has_data"]
            row_out["area_hierarchy_http"] = hier_summary["http_status"]
            row_out["area_hierarchy_attom_code"] = hier_summary["attom_code"]
            row_out["area_hierarchy_attom_msg"] = hier_summary["attom_msg"]

            # Extract geoIdV4s if we have a raw body
            if hier_result["raw"] is not None:
                zip_geo_id_v4, place_geo_id_v4 = extract_geo_ids_from_hierarchy(
                    hier_result["raw"]
                )

            time.sleep(PAUSE_BETWEEN_CALLS)
        else:
            row_out["area_hierarchy_has_data"] = False
            row_out["area_hierarchy_http"] = None
            row_out["area_hierarchy_attom_code"] = None
            row_out["area_hierarchy_attom_msg"] = "no_latlon"

        chosen_geo = place_geo_id_v4 or zip_geo_id_v4

        # Neighborhood / community stats (cache by geoIdV4)
        if chosen_geo:
            if chosen_geo in community_cache:
                comm_summary = community_cache[chosen_geo]
            else:
                comm_raw = call_neighborhood_community(chosen_geo)
                comm_summary = summarize_call(comm_raw)
                community_cache[chosen_geo] = comm_summary
            row_out["neighborhood_community_has_data"] = comm_summary["has_data"]
            row_out["neighborhood_community_http"] = comm_summary["http_status"]
            row_out["neighborhood_community_attom_code"] = comm_summary["attom_code"]
            row_out["neighborhood_community_attom_msg"] = comm_summary["attom_msg"]

            time.sleep(PAUSE_BETWEEN_CALLS)
        else:
            row_out["neighborhood_community_has_data"] = False
            row_out["neighborhood_community_http"] = None
            row_out["neighborhood_community_attom_code"] = None
            row_out["neighborhood_community_attom_msg"] = "no_geoIdV4"

        # Area boundary (cache by geoIdV4 as well)
        if chosen_geo:
            if chosen_geo in boundary_cache:
                bnd_summary = boundary_cache[chosen_geo]
            else:
                bnd_raw = call_area_boundary(chosen_geo)
                bnd_summary = summarize_call(bnd_raw)
                boundary_cache[chosen_geo] = bnd_summary
            row_out["area_boundary_has_data"] = bnd_summary["has_data"]
            row_out["area_boundary_http"] = bnd_summary["http_status"]
            row_out["area_boundary_attom_code"] = bnd_summary["attom_code"]
            row_out["area_boundary_attom_msg"] = bnd_summary["attom_msg"]

            time.sleep(PAUSE_BETWEEN_CALLS)
        else:
            row_out["area_boundary_has_data"] = False
            row_out["area_boundary_http"] = None
            row_out["area_boundary_attom_code"] = None
            row_out["area_boundary_attom_msg"] = "no_geoIdV4"

        # -----------------------------
        # Location lookup (once per city/state)
        # -----------------------------
        city_state_key = (city, state)
        if city and state:
            if city_state_key in location_cache:
                loc_summary = location_cache[city_state_key]
            else:
                loc_raw = call_location_lookup(city, state)
                loc_summary = summarize_call(loc_raw)
                location_cache[city_state_key] = loc_summary

            row_out["location_lookup_has_data"] = loc_summary["has_data"]
            row_out["location_lookup_http"] = loc_summary["http_status"]
            row_out["location_lookup_attom_code"] = loc_summary["attom_code"]
            row_out["location_lookup_attom_msg"] = loc_summary["attom_msg"]

            time.sleep(PAUSE_BETWEEN_CALLS)
        else:
            row_out["location_lookup_has_data"] = False
            row_out["location_lookup_http"] = None
            row_out["location_lookup_attom_code"] = None
            row_out["location_lookup_attom_msg"] = "no_city_state"

        # -----------------------------
        # Schools near property
        # -----------------------------
        if has_valid_point:
            school_raw = call_school_search(lat, lon)
            school_summary = summarize_call(school_raw)
            row_out["school_search_has_data"] = school_summary["has_data"]
            row_out["school_search_http"] = school_summary["http_status"]
            row_out["school_search_attom_code"] = school_summary["attom_code"]
            row_out["school_search_attom_msg"] = school_summary["attom_msg"]

            time.sleep(PAUSE_BETWEEN_CALLS)
        else:
            row_out["school_search_has_data"] = False
            row_out["school_search_http"] = None
            row_out["school_search_attom_code"] = None
            row_out["school_search_attom_msg"] = "no_latlon"

        # -----------------------------
        # POI near property
        # -----------------------------
        if one_line:
            poi_raw = call_neighborhood_poi(one_line)
            poi_summary = summarize_call(poi_raw)
            row_out["neighborhood_poi_has_data"] = poi_summary["has_data"]
            row_out["neighborhood_poi_http"] = poi_summary["http_status"]
            row_out["neighborhood_poi_attom_code"] = poi_summary["attom_code"]
            row_out["neighborhood_poi_attom_msg"] = poi_summary["attom_msg"]

            time.sleep(PAUSE_BETWEEN_CALLS)
        else:
            row_out["neighborhood_poi_has_data"] = False
            row_out["neighborhood_poi_http"] = None
            row_out["neighborhood_poi_attom_code"] = None
            row_out["neighborhood_poi_attom_msg"] = "no_address"

        # -----------------------------
        # Sales comparables by address
        # -----------------------------
        if addr_line1 and city and state and postal:
            salescomp_raw = call_salescomparables_address(
                addr_line1, city, DEFAULT_COUNTY, state, postal
            )
            salescomp_summary = summarize_call(salescomp_raw)
            row_out["salescomparables_address_has_data"] = salescomp_summary["has_data"]
            row_out["salescomparables_address_http"] = salescomp_summary["http_status"]
            row_out["salescomparables_address_attom_code"] = salescomp_summary["attom_code"]
            row_out["salescomparables_address_attom_msg"] = salescomp_summary["attom_msg"]

            time.sleep(PAUSE_BETWEEN_CALLS)
        else:
            row_out["salescomparables_address_has_data"] = False
            row_out["salescomparables_address_http"] = None
            row_out["salescomparables_address_attom_code"] = None
            row_out["salescomparables_address_attom_msg"] = "missing_address_parts"

        # -----------------------------
        # Save results
        # -----------------------------
        results_rows.append(row_out)

        # Incremental write: append this row to CSV so we have progress
        row_df = pd.DataFrame([row_out])
        mode = "w" if first_write else "a"
        header = first_write
        row_df.to_csv(OUTPUT_CSV, mode=mode, header=header, index=False)
        first_write = False

    # Final full write (optional, but keeps column order consistent)
    out_df = pd.DataFrame(results_rows)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(out_df)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
