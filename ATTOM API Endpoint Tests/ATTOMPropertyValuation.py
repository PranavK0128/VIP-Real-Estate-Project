import os
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
import pandas as pd


API_KEY = os.getenv("ATTOM_API_KEY", "b7fb03e3d1459021572ee6ce3916f7d9")

BASE = "https://api.gateway.attomdata.com"
HEADERS = {
    "accept": "application/json",
    "apikey": API_KEY,
    "ApiKey": API_KEY,
}

ZIP5 = "30060" # Marietta, GA (change to 30062/30064/30066/30067/30068)
TARGET_COUNT = 2000 # how many properties to try to collect
PAGE_SIZE = 100 # ATTOM snapshot page size (100 works well)
TIMEOUT = 30
PAUSE = 0.2 # polite pause between calls

# AVM throttling (per-property)
AVM_SLEEP = 0.15 # short delay 
MAX_AVM_ERRORS = 100 # stop if too many AVM errors

def http_get_json(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """GET JSON; treat ATTOM's 400 + SuccessWithoutResult as empty results."""
    url = f"{BASE}{path}"
    r = requests.get(url, headers=HEADERS, params=params, timeout=TIMEOUT)
    text = r.text
    try:
        data = r.json()
    except Exception:
        raise SystemExit(f"Non-JSON response from {url}:\n{text[:600]}")

    if r.status_code >= 400:
        msg = ((data.get("status") or {}).get("msg") or "").lower()
        if "success without result" in msg:
            # Means: no matches
            return data or {"status": {"code": 400, "msg": "Success Without Result"}, "property": []}
        raise SystemExit(f"HTTP {r.status_code} for {url}\nParams: {params}\nBody: {text[:600]}")
    return data

def fetch_snapshot_by_zip(zip5: str, target_count: int = TARGET_COUNT, page_size: int = PAGE_SIZE) -> List[Dict[str, Any]]:
    """Page through /property/snapshot to collect up to target_count properties for a ZIP."""
    results: List[Dict[str, Any]] = []
    page = 1
    while len(results) < target_count:
        params = {"postalcode": zip5, "pagesize": page_size, "page": page}
        data = http_get_json("/propertyapi/v1.0.0/property/snapshot", params)
        props = data.get("property") or []
        total = (data.get("status") or {}).get("total")

        results.extend(props)
        print(f"[snapshot] ZIP {zip5} page {page}: got {len(props)} (accum {len(results)} of ~{total})")

        if not props:
            break
        if len(results) >= 10000:
            print("Reached ~10k per-search cap; consider splitting across multiple ZIPs.")
            break

        page += 1
        time.sleep(PAUSE)

    return results[:target_count]

# AVM (Valuation)

def _compose_address_fields(rec: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """
    Build (address1, address2) for AVM call.
    address2 expects 'City, ST ZIP'.
    """
    address = rec.get("address") or {}
    line1 = address.get("line1")
    city = address.get("locality")
    state = address.get("countrySubd")
    zip5 = address.get("postal1")

    # If line1 is missing, try to parse 'oneLine'
    if not line1:
        one_line = address.get("oneLine")
        if one_line and "," in one_line:
            parts = [p.strip() for p in one_line.split(",")]
            if len(parts) >= 3:
                line1 = parts[0]
                # try to pull city/state/zip from remaining parts
                if not city:
                    city = parts[-2]
                if not state or not zip5:
                    last = parts[-1].split()
                    if len(last) >= 2:
                        state = last[0]
                        zip5 = last[1]

    if not (line1 and city and state and zip5):
        return None
    address2 = f"{city}, {state} {zip5}"
    return line1, address2

def fetch_avm_for_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call /attomavm/detail using address fields. Returns a dict with AVM fields.
    If address is insufficient or call fails, returns empty fields.
    """
    addr = _compose_address_fields(rec)
    if not addr:
        return {"avm_value": None, "avm_low": None, "avm_high": None, "avm_confidence": None}

    address1, address2 = addr
    params = {"address1": address1, "address2": address2}
    try:
        data = http_get_json("/propertyapi/v1.0.0/attomavm/detail", params)
    except SystemExit as e:
        # On error (rate limits/plan restrictions), return empty AVM fields
        return {"avm_value": None, "avm_low": None, "avm_high": None, "avm_confidence": None}

    # Response structures can vary; dig for plausible fields.
    # Common forms:
    # data['property'][0]['avm']['amount']['value']
    # data['property'][0]['avm']['amount']['rangeLow'] / ['rangeHigh']
    # data['property'][0]['avm']['score']['confidenceScore'] (or similar)
    try:
        prop = (data.get("property") or [])[0]
    except Exception:
        prop = None

    avm = (prop or {}).get("avm") if isinstance(prop, dict) else None
    amount = (avm or {}).get("amount") if isinstance(avm, dict) else None
    score = (avm or {}).get("score") if isinstance(avm, dict) else None

    val = (amount or {}).get("value") if isinstance(amount, dict) else None
    low = (amount or {}).get("rangeLow") if isinstance(amount, dict) else None
    high = (amount or {}).get("rangeHigh") if isinstance(amount, dict) else None
    conf = (score or {}).get("confidenceScore") if isinstance(score, dict) else None

    return {"avm_value": val, "avm_low": low, "avm_high": high, "avm_confidence": conf}

def flatten_snapshot(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten snapshot JSON into a single row dict of core details."""
    address = rec.get("address") or {}
    ident = rec.get("identifier") or {}
    location = rec.get("location") or {}
    building = rec.get("building") or {}
    rooms = building.get("rooms") if isinstance(building, dict) else {}
    built = building.get("built") if isinstance(building, dict) else {}
    size  = building.get("size") if isinstance(building, dict) else {}
    lot   = rec.get("lot") or {}
    summary = rec.get("summary") or {}
    last_sale = (rec.get("sale") or {}).get("lastSale") if isinstance(rec.get("sale"), dict) else {}
    tax = rec.get("assessment") or {}

    row = {
        "attomId": ident.get("attomId"),
        "apn": ident.get("apn"),
        "fips": ident.get("fips"),
        "address_oneLine": address.get("oneLine"),
        "address_line1": address.get("line1"),
        "city": address.get("locality"),
        "state": address.get("countrySubd"),
        "zip": address.get("postal1"),
        "latitude": location.get("latitude"),
        "longitude": location.get("longitude"),
        "beds": (rooms or {}).get("beds"),
        "baths_total": (rooms or {}).get("bathstotal"),
        "year_built": (built or {}).get("yearBuilt"),
        "sqft": (size or {}).get("universalsize"),
        "lot_sqft": lot.get("lotsize2"),
        "property_type": summary.get("propertyType") or summary.get("propsubtype"),
        "last_update": rec.get("lastUpdated") or summary.get("lastUpdated"),
        "last_sale_date": (last_sale or {}).get("saleSearchDate") or (last_sale or {}).get("saleTransDate"),
        "last_sale_price": (last_sale or {}).get("saleAmt"),
        "tax_year": tax.get("taxYear"),
        "tax_amount": tax.get("taxAmt"),
    }
    return row

def main():
    # 1 Collect snapshot rows
    props = fetch_snapshot_by_zip(ZIP5, target_count=TARGET_COUNT, page_size=PAGE_SIZE)
    if not props:
        print(f"No snapshot results for ZIP {ZIP5}. Exiting.")
        return

    # 2 Flatten snapshot to rows
    rows = [flatten_snapshot(p) for p in props]

    # 3 AVM valuation (best effort)
    avm_errors = 0
    for i, p in enumerate(props, start=1):
        avm = fetch_avm_for_record(p)
        rows[i - 1].update(avm)

        if (i % 50) == 0:
            print(f"[avm] processed {i} / {len(props)} rows")
        if all(v is None for v in avm.values()):
            avm_errors += 1
            if avm_errors >= MAX_AVM_ERRORS:
                print("Too many AVM errors; continuing without further AVM calls.")
                # stop trying further AVM calls, leave remaining as None
                for j in range(i, len(rows)):
                    rows[j].update({"avm_value": None, "avm_low": None, "avm_high": None, "avm_confidence": None})
                break

        time.sleep(AVM_SLEEP)

    # 4 Build DataFrame & write CSV
    df = pd.DataFrame.from_records(rows)
    out = "propertyValuationSample.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"\nSaved {len(df)} rows to: {out}")

    # 5 Show a quick preview
    try:
        print("\nPreview:")
        print(df.head(12).to_string(index=False))
    except Exception:
        print(df.head(12))

if __name__ == "__main__":
    main()
