import os
import sys
import csv
import time
import argparse
from math import ceil
from typing import Any, Dict, List, Optional, Tuple, Set, Union

import requests
import pandas as pd

API_KEY = os.getenv("ATTOM_API_KEY", "b7fb03e3d1459021572ee6ce3916f7d9")

BASE = "https://api.gateway.attomdata.com"
HEADERS = {
    "accept": "application/json",
    "apikey": API_KEY,
    "ApiKey": API_KEY,
}

DEFAULT_ZIPS = ["30060", "30062", "30064", "30066", "30067", "30068"]

PAGE_SIZE_DEFAULT = 100
TIMEOUT = 30
PAUSE = 0.20  # pause between calls
MAX_RETRIES = 6
BACKOFF_BASE = 0.8 # exponential backoff base (seconds)

FALLBACK_SLEEP = 0.12 # delay between per-id fallback calls

def _normalize_msg(s: str) -> str:
    s = (s or "").lower().strip()
    return s.replace(" ", "").replace("-", "").replace("_", "")


def http_get_json(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    # GET JSON with retry/backoff; treat ATTOM 400 'SuccessWithoutResult' as empty
    url = f"{BASE}{path}"
    retries = 0

    while True:
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=TIMEOUT)
            text = r.text

            try:
                data = r.json()
            except Exception:
                raise SystemExit(f"Non-JSON response from {url}:\n{text[:600]}")

            if 200 <= r.status_code < 300:
                return data

            msg_raw = ((data.get("status") or {}).get("msg") or "")
            msg = _normalize_msg(msg_raw)
            if r.status_code == 400 and "successwithoutresult" in msg:
                return data or {"status": {"code": 400, "msg": "SuccessWithoutResult"}, "property": []}

            if r.status_code in (429, 500, 502, 503, 504):
                if retries < MAX_RETRIES:
                    sleep_s = BACKOFF_BASE * (2 ** retries)
                    time.sleep(sleep_s)
                    retries += 1
                    continue

            raise SystemExit(
                f"HTTP {r.status_code} for {url}\nParams: {params}\nBody: {text[:600]}"
            )

        except requests.RequestException as e:
            if retries < MAX_RETRIES:
                sleep_s = BACKOFF_BASE * (2 ** retries)
                time.sleep(sleep_s)
                retries += 1
                continue
            raise SystemExit(f"Network error for {url} after retries: {e}")


def _first_num(value: Any) -> Optional[Union[int, float]]:
    # Return value if it looks numeric (and > 0), else None
    if isinstance(value, (int, float)):
        return value if value > 0 else None
    # Sometimes strings with digits appear
    try:
        s = str(value).replace(",", "").strip()
        if s and s.replace(".", "", 1).isdigit():
            num = float(s)
            return num if num > 0 else None
    except Exception:
        pass
    return None


def _deep_find_numeric(d: Any, key_pred) -> Optional[Union[int, float]]:
    
    # Recursively search dict/list looking for a numeric value whose key matches key_pred.
    # key_pred: function(str) -> bool
    
    if isinstance(d, dict):
        for k, v in d.items():
            try:
                if key_pred(str(k)):
                    num = _first_num(v)
                    if num is not None:
                        return num
            except Exception:
                pass
            val = _deep_find_numeric(v, key_pred)
            if val is not None:
                return val
    elif isinstance(d, list):
        for item in d:
            val = _deep_find_numeric(item, key_pred)
            if val is not None:
                return val
    return None


def extract_sale_price(sale: Dict[str, Any]) -> Optional[Union[int, float]]:
    
    # Try multiple shapes for sale amount/price commonly seen across ATTOM payloads.
    # Falls back to a deep search for keys including 'sale' & ('amt'|'amount'|'price').
    
    if not isinstance(sale, dict):
        return None

    # Direct candidates
    candidates = [
        sale.get("saleAmt"),
        sale.get("saleAmount"),
        (sale.get("amount") or {}).get("saleAmt") if isinstance(sale.get("amount"), dict) else None,
        (sale.get("amount") or {}).get("saleAmount") if isinstance(sale.get("amount"), dict) else None,
        sale.get("salePrice"),
        (sale.get("price") or {}).get("amount") if isinstance(sale.get("price"), dict) else sale.get("price"),
        sale.get("considerationAmount"),  # some deed records carry consideration
    ]
    for c in candidates:
        num = _first_num(c)
        if num is not None:
            return num

    # Deep search: any key like sale_amt, saleamount, saleprice etc.
    def pred(k: str) -> bool:
        k = k.lower()
        return ("sale" in k) and ("amt" in k or "amount" in k or "price" in k)

    return _deep_find_numeric(sale, pred)


def fetch_sales_history_latest_amount(attom_id: Any) -> Optional[Union[int, float]]:
    
    # Fallback: call /saleshistory/snapshot?id=<attomId> and pull the most recent amount.
    
    if not attom_id:
        return None
    try:
        data = http_get_json("/propertyapi/v1.0.0/saleshistory/snapshot", {"id": attom_id})
        props = data.get("property") or []
        if not props:
            return None
        # sales history commonly under property[0]['salehistory'] as list
        rec = props[0]
        history = rec.get("salehistory") or rec.get("saleHistory") or []
        if isinstance(history, dict):
            # sometimes wrapped in {'history': [ ... ]}
            history = history.get("history") or []
        best = None
        best_date = ""
        for h in history if isinstance(history, list) else []:
            amount = extract_sale_price(h) or _first_num(h.get("saleAmt") or h.get("saleAmount"))
            date = h.get("saleTransDate") or h.get("saleSearchDate") or h.get("recordingDate") or ""
            # take the most recent record that has an amount
            if amount is not None:
                if not best or str(date) > str(best_date):
                    best = amount
                    best_date = date
        return best
    except SystemExit:
        return None
    except Exception:
        return None


def flatten_sale_snapshot(rec: Dict[str, Any], try_fill: bool = False, fill_budget: List[int] = None) -> Dict[str, Any]:
    
    # Flatten a /sale/snapshot property record.
    # If try_fill and sale_price missing, attempts a per-id sales history lookup
    # (consumes from fill_budget[0]).
    
    address = rec.get("address") or {}
    ident = rec.get("identifier") or {}
    location = rec.get("location") or {}
    summary = rec.get("summary") or {}
    building = rec.get("building") or {}
    rooms = building.get("rooms") if isinstance(building, dict) else {}
    size  = building.get("size") if isinstance(building, dict) else {}

    sale = rec.get("sale") or {}
    sale_date = sale.get("saleTransDate") or sale.get("saleSearchDate")
    recording_date = sale.get("recordingDate")
    doc_type = sale.get("documentType") or sale.get("docType")

    buyer = None
    seller = None
    parties = sale.get("parties") if isinstance(sale, dict) else None
    if isinstance(parties, dict):
        buyers = parties.get("buyer")
        sellers = parties.get("seller")
        if isinstance(buyers, list) and buyers:
            buyer = buyers[0].get("name") or buyers[0].get("fullName")
        elif isinstance(buyers, dict):
            buyer = buyers.get("name") or buyers.get("fullName")
        if isinstance(sellers, list) and sellers:
            seller = sellers[0].get("name") or sellers[0].get("fullName")
        elif isinstance(sellers, dict):
            seller = sellers.get("name") or sellers.get("fullName")

    sale_price = extract_sale_price(sale)

    # Optional fallback to sales history for missing prices (bounded by fill_budget)
    if sale_price is None and try_fill and fill_budget is not None and fill_budget[0] > 0:
        attom_id = ident.get("attomId")
        amt = fetch_sales_history_latest_amount(attom_id)
        if amt is not None:
            sale_price = amt
        fill_budget[0] -= 1
        time.sleep(FALLBACK_SLEEP)  # be gentle

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
        "property_type": summary.get("propertyType") or summary.get("propsubtype"),
        "sqft": (size or {}).get("universalsize"),
        "beds": (rooms or {}).get("beds"),
        "baths_total": (rooms or {}).get("bathstotal"),
        # Sale fields
        "sale_date": sale_date,
        "sale_price": sale_price,
        "recording_date": recording_date,
        "document_type": doc_type,
        "buyer_name": buyer,
        "seller_name": seller,
    }
    return row


def existing_key_set(out_path: str) -> Set[Tuple[Any, Any, Any]]:
    """Read existing CSV to build a dedupe key set: (attomId, sale_date, sale_price)."""
    keys: Set[Tuple[Any, Any, Any]] = set()
    if not os.path.exists(out_path):
        return keys
    try:
        usecols = ["attomId", "sale_date", "sale_price"]
        for chunk in pd.read_csv(out_path, usecols=usecols, chunksize=100_000, dtype=str):
            for _, row in chunk.iterrows():
                keys.add((row.get("attomId"), row.get("sale_date"), row.get("sale_price")))
    except Exception as e:
        print(f"[warn] Could not load existing keys from {out_path}: {e}", file=sys.stderr)
    return keys


def write_chunk(out_path: str, rows: List[Dict[str, Any]], write_header: bool):
    """Append a chunk of rows to CSV (writes header only for the first write)."""
    if not rows:
        return
    df = pd.DataFrame.from_records(rows)
    df.to_csv(out_path, index=False, mode="a", header=write_header, encoding="utf-8")


def fetch_and_stream_zip(zip5: str,
                         out_path: str,
                         seen_keys: Set[Tuple[Any, Any, Any]],
                         page_size: int,
                         max_pages: Optional[int],
                         pause: float,
                         try_fill: bool,
                         fill_budget: List[int]) -> int:
    total_written = 0
    page = 1
    wrote_header_once = os.path.getsize(out_path) == 0 if os.path.exists(out_path) else True

    total_count = None
    total_pages = None

    while True:
        if max_pages is not None and page > max_pages:
            break
        if total_pages is not None and page > total_pages:
            break

        params = {"postalcode": zip5, "pagesize": page_size, "page": page}
        data = http_get_json("/propertyapi/v1.0.0/sale/snapshot", params)
        status = data.get("status") or {}
        props = data.get("property") or []

        if total_count is None:
            total_count = status.get("total")
            if isinstance(total_count, int) and total_count > 0:
                total_pages = ceil(total_count / page_size)

        print(f"[sale] ZIP {zip5} page {page}: got {len(props)} (target total ~{total_count})")

        if not props:
            break

        # Build chunk with dedupe + flatten + optional fill
        chunk: List[Dict[str, Any]] = []
        for rec in props:
            sale = (rec.get("sale") or {})
            ident = (rec.get("identifier") or {})
            key = (
                ident.get("attomId"),
                sale.get("saleTransDate") or sale.get("saleSearchDate"),
                extract_sale_price(sale)  # use extracted amount in key for stability
            )
            if key in seen_keys:
                continue
            row = flatten_sale_snapshot(rec, try_fill=try_fill, fill_budget=fill_budget)
            seen_keys.add((row["attomId"], row["sale_date"], row["sale_price"]))
            chunk.append(row)

        write_chunk(out_path, chunk, write_header=wrote_header_once)
        if wrote_header_once:
            wrote_header_once = False

        total_written += len(chunk)
        page += 1
        time.sleep(pause)

    return total_written


def main():
    parser = argparse.ArgumentParser(description="Stream sale data for Marietta/Cobb ZIPs to CSV.")
    parser.add_argument("--zips", type=str, default=",".join(DEFAULT_ZIPS),
                        help="Comma-separated ZIP list (default: Marietta/Cobb ZIPs)")
    parser.add_argument("--page-size", type=int, default=PAGE_SIZE_DEFAULT,
                        help="Page size for ATTOM sale snapshot (default: 100)")
    parser.add_argument("--max-pages", type=int, default=None,
                        help="Optional page limit per ZIP (default: no explicit limit)")
    parser.add_argument("--out", type=str, default="mariettaDataV2.csv",
                        help="Output CSV filename (default: mariettaDataV2.csv)")
    parser.add_argument("--truncate", action="store_true",
                        help="Delete output file before writing (start fresh).")
    parser.add_argument("--resume", action="store_true",
                        help="Resume dedup from existing file (reads keys to skip duplicates).")
    parser.add_argument("--fill-missing-price", action="store_true",
                        help="If set, try to backfill missing sale_price via /saleshistory/snapshot per attomId.")
    parser.add_argument("--fill-limit", type=int, default=300,
                        help="Max number of fallback lookups this run (default: 300).")

    args = parser.parse_args()

    zip_list = [z.strip() for z in args.zips.split(",") if z.strip()]
    if not zip_list:
        print("No ZIPs provided; exiting.", file=sys.stderr)
        sys.exit(1)

    # Prepare output
    if args.truncate and os.path.exists(args.out):
        os.remove(args.out)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # Build seen-keys set (for resume or fresh)
    seen_keys: Set[Tuple[Any, Any, Any]] = set()
    if args.resume:
        seen_keys = existing_key_set(args.out)
        print(f"[resume] Loaded {len(seen_keys)} existing keys from {args.out}")

    # Fill budget for backfill lookups
    fill_budget = [max(0, int(args.fill_limit))] if args.fill_missing_price else [0]

    grand_total = 0
    for z in zip_list:
        wrote = fetch_and_stream_zip(
            z,
            out_path=args.out,
            seen_keys=seen_keys,
            page_size=args.page_size,
            max_pages=args.max_pages,
            pause=PAUSE,
            try_fill=args.fill_missing_price,
            fill_budget=fill_budget,
        )
        grand_total += wrote
        print(f"[zip {z}] wrote {wrote} new rows; cumulative written this run: {grand_total}")

    print(f"\nDone. Rows written this run: {grand_total}. Output: {args.out}\n")


if __name__ == "__main__":
    main()
