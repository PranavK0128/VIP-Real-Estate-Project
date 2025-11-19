#!/usr/bin/env python3
"""
ATTOMEndpointsWorking.py

Step 1: Test run on ~10 properties
----------------------------------
Reads MariettaDataV2.csv and, for the first N properties (default 10),
calls only the ATTOM endpoints that we know return data (from
testATTOMEndpoints.csv). It writes the enriched results to a *new* CSV:

    ATTOMEndpointsWorking.csv

This file:
    * DOES NOT overwrite MariettaDataV2.csv
    * Includes all original Marietta columns
    * Adds JSON columns for each working endpoint
    * Updates incrementally after each property, so partial progress is saved
      even if the script terminates early.

After you confirm the columns / JSON look good for those 10 properties,
we can then expand to more properties or integrate this into your main
Marietta pipeline.

Usage:
    python ATTOMEndpointsWorking.py

To change how many properties are processed, edit MAX_PROPERTIES below.
"""

import os
import sys
import time
import json
import math
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_KEY = os.getenv("ATTOM_API_KEY", "b7fb03e3d1459021572ee6ce3916f7d9")

BASE = "https://api.gateway.attomdata.com"
HEADERS = {
    "accept": "application/json",
    "apikey": API_KEY,
    "ApiKey": API_KEY,
}

# Input / output files
INPUT_CSV = "MariettaDataV2.csv"          # read-only
OUTPUT_CSV = "ATTOMEndpointsWorking.csv"  # NEW file (this script writes here)

# How many properties to process in this test run
MAX_PROPERTIES = 10

# Networking
TIMEOUT = 30
MAX_RETRIES = 6
PAUSE_BETWEEN_CALLS = 0.20  # seconds between calls

# ---------------------------------------------------------------------------
# Endpoint definitions
# Only endpoints that had _has_data == True in testATTOMEndpoints.csv
# ---------------------------------------------------------------------------

PROPERTY_ENDPOINTS_BY_ATTOM: Dict[str, str] = {
    # Core property info
    "property_id": "/propertyapi/v1.0.0/property/id",
    "property_snapshot": "/propertyapi/v1.0.0/property/snapshot",
    "property_basicprofile": "/propertyapi/v1.0.0/property/basicprofile",
    "property_detail": "/propertyapi/v1.0.0/property/detail",
    "property_expandedprofile": "/propertyapi/v1.0.0/property/expandedprofile",
    "property_detailwithschools": "/propertyapi/v1.0.0/property/detailwithschools",
    "property_detailowner": "/propertyapi/v1.0.0/property/detailowner",
    "property_detailmortgage": "/propertyapi/v1.0.0/property/detailmortgage",
    "property_detailmortgageowner": "/propertyapi/v1.0.0/property/detailmortgageowner",

    # Valuation & equity
    "attomavm_detail": "/propertyapi/v1.0.0/attomavm/detail",
    "valuation_homeequity": "/propertyapi/v1.0.0/valuation/homeequity",
    "avm_snapshot": "/propertyapi/v1.0.0/avm/snapshot",

    # Assessment & taxes
    "assessment_detail": "/propertyapi/v1.0.0/assessment/detail",
    "assessment_snapshot": "/propertyapi/v1.0.0/assessment/snapshot",

    # Sales & history
    "saleshistory_snapshot": "/propertyapi/v1.0.0/saleshistory/snapshot",
    "saleshistory_detail": "/propertyapi/v1.0.0/saleshistory/detail",
    "sale_snapshot": "/propertyapi/v1.0.0/sale/snapshot",
}

SPECIAL_ENDPOINTS = {
    "property_address": "/propertyapi/v1.0.0/property/address",
    "area_hierarchy": "/v4/area/hierarchy/lookup",
    "neighborhood_community": "/v4/neighborhood/community",
    "location_lookup": "/v4/location/lookup",
}

# ---------------------------------------------------------------------------
# HTTP + helpers
# ---------------------------------------------------------------------------

def http_get_json(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make a GET request to ATTOM and return a structured result:

    {
        "ok": bool,
        "status_code": int or None,
        "data": dict or None,
        "error": str or None,
    }
    """
    url = BASE + path
    last_error: Optional[str] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=TIMEOUT)
        except requests.RequestException as exc:
            last_error = f"request_exception: {exc}"
            time.sleep(PAUSE_BETWEEN_CALLS * attempt)
            continue

        status_code = resp.status_code

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

        # Try to parse ATTOM error body
        try:
            data = resp.json()
        except ValueError:
            data = None

        # Retry on rate limit / transient server errors
        if status_code in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES:
            last_error = f"http_{status_code}"
            time.sleep(PAUSE_BETWEEN_CALLS * attempt)
            continue

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


def safe_json_dump(obj: Any) -> Optional[str]:
    """Serialize an object to a compact JSON string for CSV storage."""
    if obj is None:
        return None
    try:
        return json.dumps(obj, separators=(",", ":"))
    except Exception:
        return None


def summarize_body(body: Any) -> Tuple[Optional[str], Optional[str]]:
    """
    From an ATTOM JSON body, extract status.code and a human-readable message.
    Returns (status_code_string, status_msg)
    """
    if not isinstance(body, dict):
        return None, None
    status = body.get("status") or {}
    code = status.get("code")
    if code is not None:
        code = str(code)
    msg = (
        status.get("msg")
        or status.get("short_description")
        or status.get("long_description")
    )
    return code, msg


def extract_geo_ids_from_hierarchy(body: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """
    From /v4/area/hierarchy/lookup, get ZIP-level and place-level geoIdV4, if present.
    Returns (zip_geoIdV4, place_geoIdV4).
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

        # ZIP / postal-level
        if gtype in ("PZ", "ZI") and zip_geo is None:
            zip_geo = gid_v4
        # Place / city-level
        if gtype in ("PL", "CI") and place_geo is None:
            place_geo = gid_v4

    return zip_geo, place_geo

# ---------------------------------------------------------------------------
# Enrichment for one property
# ---------------------------------------------------------------------------

def enrich_single_property(
    base_row: pd.Series,
    hierarchy_cache: Dict[Tuple[float, float], Dict[str, Any]],
    community_cache: Dict[str, Dict[str, Any]],
    location_cache: Dict[Tuple[str, str], Dict[str, Any]],
) -> Dict[str, Any]:
    """
    For a single MariettaDataV2 row, call all working endpoints and return a dict
    containing:

        * Original columns
        * <endpoint>_json: full JSON response (string)
        * <endpoint>_status_http: HTTP status code
        * <endpoint>_status_msg: ATTOM status or error message
        * area_zip_geoIdV4, area_place_geoIdV4
        * neighborhood_geoIdV4
    """
    out: Dict[str, Any] = {}

    # Copy original columns into output
    for col in base_row.index:
        out[col] = base_row[col]

    attom_id = base_row.get("attomId")
    one_line = base_row.get("address_oneLine") or ""
    city = base_row.get("city") or ""
    state = base_row.get("state") or ""

    lat = base_row.get("latitude")
    lon = base_row.get("longitude")

    has_valid_point = False
    try:
        if lat is not None and lon is not None and not (math.isnan(lat) or math.isnan(lon)):
            has_valid_point = True
    except TypeError:
        has_valid_point = False

    # -------------------------------
    # Property endpoints (by attomId)
    # -------------------------------
    if attom_id is None or (isinstance(attom_id, float) and math.isnan(attom_id)):
        for key in PROPERTY_ENDPOINTS_BY_ATTOM.keys():
            out[f"{key}_json"] = None
            out[f"{key}_status_http"] = None
            out[f"{key}_status_msg"] = "no_attomId"
    else:
        for key, path in PROPERTY_ENDPOINTS_BY_ATTOM.items():
            params = {"attomId": str(attom_id)}
            res = http_get_json(path, params)
            body = res.get("data")
            out[f"{key}_json"] = safe_json_dump(body) if res.get("ok") else None
            out[f"{key}_status_http"] = res.get("status_code")
            _, status_msg = summarize_body(body) if res.get("ok") else (None, res.get("error"))
            out[f"{key}_status_msg"] = status_msg
            time.sleep(PAUSE_BETWEEN_CALLS)

    # -------------------------------
    # /property/address (one-line)
    # -------------------------------
    if isinstance(one_line, str) and one_line.strip():
        res = http_get_json(
            SPECIAL_ENDPOINTS["property_address"],
            {"address": one_line},
        )
        body = res.get("data")
        out["property_address_json"] = safe_json_dump(body) if res.get("ok") else None
        out["property_address_status_http"] = res.get("status_code")
        _, status_msg = summarize_body(body) if res.get("ok") else (None, res.get("error"))
        out["property_address_status_msg"] = status_msg
        time.sleep(PAUSE_BETWEEN_CALLS)
    else:
        out["property_address_json"] = None
        out["property_address_status_http"] = None
        out["property_address_status_msg"] = "no_address"

    # -------------------------------
    # Area hierarchy (lat/lon) -> geoIds
    # -------------------------------
    zip_geo_id_v4: Optional[str] = None
    place_geo_id_v4: Optional[str] = None

    if has_valid_point:
        key_pt = (round(float(lat), 6), round(float(lon), 6))
        if key_pt in hierarchy_cache:
            hier_res = hierarchy_cache[key_pt]["res"]
            hier_body = hierarchy_cache[key_pt]["body"]
        else:
            hier_res = http_get_json(
                SPECIAL_ENDPOINTS["area_hierarchy"],
                {
                    "latitude": f"{key_pt[0]:.8f}",
                    "longitude": f"{key_pt[1]:.8f}",
                    "geoType": "All",
                },
            )
            hier_body = hier_res.get("data") if hier_res.get("ok") else None
            hierarchy_cache[key_pt] = {"res": hier_res, "body": hier_body}

        out["area_hierarchy_json"] = safe_json_dump(hier_body) if hier_res.get("ok") else None
        out["area_hierarchy_status_http"] = hier_res.get("status_code")
        _, status_msg = summarize_body(hier_body) if hier_res.get("ok") else (None, hier_res.get("error"))
        out["area_hierarchy_status_msg"] = status_msg

        if hier_body is not None:
            zip_geo_id_v4, place_geo_id_v4 = extract_geo_ids_from_hierarchy(hier_body)

        out["area_zip_geoIdV4"] = zip_geo_id_v4
        out["area_place_geoIdV4"] = place_geo_id_v4

        time.sleep(PAUSE_BETWEEN_CALLS)
    else:
        out["area_hierarchy_json"] = None
        out["area_hierarchy_status_http"] = None
        out["area_hierarchy_status_msg"] = "no_latlon"
        out["area_zip_geoIdV4"] = None
        out["area_place_geoIdV4"] = None

    # -------------------------------
    # Neighborhood / community (by geoIdV4)
    # -------------------------------
    chosen_geo = place_geo_id_v4 or zip_geo_id_v4
    if chosen_geo:
        if chosen_geo in community_cache:
            comm_res = community_cache[chosen_geo]["res"]
            comm_body = community_cache[chosen_geo]["body"]
        else:
            # NOTE: we saw data in testATTOMEndpoints using "geoidv4" param
            comm_res = http_get_json(
                SPECIAL_ENDPOINTS["neighborhood_community"],
                {"geoidv4": chosen_geo},
            )
            comm_body = comm_res.get("data") if comm_res.get("ok") else None
            community_cache[chosen_geo] = {"res": comm_res, "body": comm_body}

        out["neighborhood_geoIdV4"] = chosen_geo
        out["neighborhood_community_json"] = safe_json_dump(comm_body) if comm_res.get("ok") else None
        out["neighborhood_community_status_http"] = comm_res.get("status_code")
        _, status_msg = summarize_body(comm_body) if comm_res.get("ok") else (None, comm_res.get("error"))
        out["neighborhood_community_status_msg"] = status_msg

        time.sleep(PAUSE_BETWEEN_CALLS)
    else:
        out["neighborhood_geoIdV4"] = None
        out["neighborhood_community_json"] = None
        out["neighborhood_community_status_http"] = None
        out["neighborhood_community_status_msg"] = "no_geoIdV4"

    # -------------------------------
    # Location lookup (city/state)
    # -------------------------------
    city_key = str(city).strip().upper()
    state_key = str(state).strip().upper()
    city_state_key = (city_key, state_key)

    if city_key and state_key:
        if city_state_key in location_cache:
            loc_res = location_cache[city_state_key]["res"]
            loc_body = location_cache[city_state_key]["body"]
        else:
            loc_res = http_get_json(
                SPECIAL_ENDPOINTS["location_lookup"],
                {
                    "name": f"{city_key.title()}, {state_key}",
                    "geographyTypeAbbreviation": "CI",
                },
            )
            loc_body = loc_res.get("data") if loc_res.get("ok") else None
            location_cache[city_state_key] = {"res": loc_res, "body": loc_body}

        out["location_lookup_json"] = safe_json_dump(loc_body) if loc_res.get("ok") else None
        out["location_lookup_status_http"] = loc_res.get("status_code")
        _, status_msg = summarize_body(loc_body) if loc_res.get("ok") else (None, loc_res.get("error"))
        out["location_lookup_status_msg"] = status_msg

        time.sleep(PAUSE_BETWEEN_CALLS)
    else:
        out["location_lookup_json"] = None
        out["location_lookup_status_http"] = None
        out["location_lookup_status_msg"] = "no_city_state"

    return out

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not API_KEY or API_KEY == "REPLACE_WITH_YOUR_KEY":
        print(
            "WARNING: ATTOM_API_KEY is not set. "
            "Set it in your environment before running.",
            file=sys.stderr,
        )

    if not os.path.exists(INPUT_CSV):
        print(f"ERROR: Input CSV '{INPUT_CSV}' not found.", file=sys.stderr)
        sys.exit(1)

    base_df = pd.read_csv(INPUT_CSV)
    if base_df.empty:
        print(f"ERROR: Input CSV '{INPUT_CSV}' is empty.", file=sys.stderr)
        sys.exit(1)

    # Only take the first MAX_PROPERTIES rows for this test run
    sample_df = base_df.head(MAX_PROPERTIES).copy()
    print(
        f"Loaded {len(base_df)} total rows from {INPUT_CSV}; "
        f"processing first {len(sample_df)} properties into {OUTPUT_CSV}."
    )

    # If output exists from a previous run, start fresh
    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)

    first_write = True

    # Caches so area / community / location endpoints don't get hammered
    hierarchy_cache: Dict[Tuple[float, float], Dict[str, Any]] = {}
    community_cache: Dict[str, Dict[str, Any]] = {}
    location_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for idx, (_, row) in enumerate(sample_df.iterrows(), start=1):
        attom_id = row.get("attomId")
        one_line = row.get("address_oneLine")
        print(f"[{idx}/{len(sample_df)}] Enriching attomId={attom_id} at {one_line}")

        row_out = enrich_single_property(
            row,
            hierarchy_cache=hierarchy_cache,
            community_cache=community_cache,
            location_cache=location_cache,
        )

        # Incremental write: append row immediately so we keep partial results
        out_df = pd.DataFrame([row_out])
        mode = "w" if first_write else "a"
        header = first_write
        out_df.to_csv(OUTPUT_CSV, mode=mode, header=header, index=False)
        first_write = False

    print(f"Done. Wrote {len(sample_df)} enriched properties to {OUTPUT_CSV}.")

if __name__ == "__main__":
    main()
