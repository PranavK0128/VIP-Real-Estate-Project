#!/usr/bin/env python3
"""
GeorgiaPredictPropertyPrice.py

Use the trained Gradient-Boosted Trees pipeline to predict a property's price.

Workflow:
- Load georgia_gbdt_model.joblib (pipeline + feature lists).
- Load processed CSVs from "Georgia Real Estate Data Processed".
- Allow the user to:
    1. Pick a random property from the combined dataset, or
    2. Search for a property by address substring.
- For the chosen property, build the feature row using the exact
  numeric + categorical columns used in training, and:
    - Predict price using its original sale_date-based features.
    - Predict a "today" price by overwriting sale_year/month/quarter
      with today's date (rough approximation of current value).
"""

import os
import sys
from typing import List
from datetime import date

import numpy as np
import pandas as pd
from joblib import load


# ---------------------------------------------------------------------------
# Helpers to find the processed-data folder
# ---------------------------------------------------------------------------

def get_processed_directory() -> str:
    """
    Returns the path to the "Georgia Real Estate Data Processed" folder,
    assuming this script is inside "Georgia Real Estate ML Model"
    in the same parent directory.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    processed_dir = os.path.join(parent_dir, "Georgia Real Estate Data Processed")
    return processed_dir


def list_processed_csvs(processed_dir: str) -> List[str]:
    """
    Return a sorted list of processed CSV filenames ending in '_processed.csv'
    in the given directory.
    """
    if not os.path.isdir(processed_dir):
        print(f"[ERROR] Processed folder not found: {processed_dir}")
        return []

    files = [
        name
        for name in os.listdir(processed_dir)
        if name.lower().endswith("_processed.csv")
    ]
    files.sort()
    return files


def choose_files(csv_files: List[str], processed_dir: str) -> List[str]:
    """
    Allow the user to choose which processed CSV files to use for picking
    properties.

    Options:
    - Command-line arguments:
        python GeorgiaPredictPropertyPrice.py all
        python GeorgiaPredictPropertyPrice.py 1 3
        python GeorgiaPredictPropertyPrice.py Athens_PropertyData_processed.csv
    - Interactive prompt:
        Type 'all' or a comma-separated list of indices (e.g., 1,3).
    """
    if not csv_files:
        print("[ERROR] No *_processed.csv files found in processed folder.")
        sys.exit(1)

    print("\nAvailable processed CSV files in:")
    print(f"  {processed_dir}\n")
    for idx, name in enumerate(csv_files, start=1):
        print(f"  {idx}. {name}")

    # Command-line override
    if len(sys.argv) > 1:
        args = sys.argv[1:]
        print(f"\n[INFO] Using command-line arguments: {args}")

        if len(args) == 1 and args[0].lower() in ("all", "*"):
            return csv_files

        chosen: List[str] = []
        for arg in args:
            if arg.isdigit():
                idx_int = int(arg)
                if 1 <= idx_int <= len(csv_files):
                    chosen.append(csv_files[idx_int - 1])
                else:
                    print(f"[ERROR] Index out of range: {arg}")
                    sys.exit(1)
            else:
                candidate = arg
                if not candidate.lower().endswith(".csv"):
                    candidate = candidate + ".csv"
                if candidate in csv_files:
                    chosen.append(candidate)
                else:
                    print(f"[ERROR] File not found in list: {candidate}")
                    sys.exit(1)

        if not chosen:
            print("[ERROR] No valid files from command line arguments.")
            sys.exit(1)

        return chosen

    # Interactive input
    user_input = input(
        "\nType 'all' to use all files, or a comma-separated list of indices "
        "(e.g., 1,3): "
    ).strip()

    if user_input.lower() in ("all", "*", ""):
        return csv_files

    indices: List[int] = []
    for part in user_input.split(","):
        part = part.strip()
        if not part:
            continue
        if not part.isdigit():
            print(f"[ERROR] Invalid index '{part}'")
            sys.exit(1)
        idx_int = int(part)
        if 1 <= idx_int <= len(csv_files):
            indices.append(idx_int)
        else:
            print(f"[ERROR] Index out of range: {idx_int}")
            sys.exit(1)

    chosen_files = [csv_files[i - 1] for i in indices]
    return chosen_files


def load_and_combine(processed_dir: str, chosen_files: List[str]) -> pd.DataFrame:
    """
    Load each chosen CSV from the processed directory and concatenate them
    into a single DataFrame.
    """
    dfs = []
    for name in chosen_files:
        path = os.path.join(processed_dir, name)
        print(f"[INFO] Loading {path}")
        df = pd.read_csv(path, low_memory=False)
        dfs.append(df)

    combined = pd.concat(dfs, axis=0, ignore_index=True)
    print(f"[INFO] Combined rows from all chosen files: {len(combined)}")
    return combined


def engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert sale_date to datetime and engineer the same time-based features
    used during training: sale_year, sale_month, sale_quarter, sale_year_month.
    """
    df = df.copy()
    df["sale_date"] = pd.to_datetime(df.get("sale_date"), errors="coerce")
    df["sale_year"] = df["sale_date"].dt.year
    df["sale_month"] = df["sale_date"].dt.month
    df["sale_quarter"] = df["sale_date"].dt.quarter
    df["sale_year_month"] = df["sale_year"] * 12 + df["sale_month"]
    return df


# ---------------------------------------------------------------------------
# Prediction logic
# ---------------------------------------------------------------------------

def select_property(df: pd.DataFrame) -> pd.Series:
    """
    Interactively select a property from df:
    - Option 1: random property
    - Option 2: search by address substring and choose a row ID

    Returns a single-row Series representing the chosen property.
    """
    while True:
        print("\n=== Georgia Property Price Predictor ===")
        print("1. Pick a RANDOM property from the data")
        print("2. SEARCH property by address substring")
        print("3. Quit")
        choice = input("Enter choice [1-3]: ").strip()

        if choice == "3" or choice.lower() == "q":
            print("Goodbye.")
            sys.exit(0)

        if choice == "1":
            row = df.sample(1, random_state=None).iloc[0]
            return row

        elif choice == "2":
            # Find an address-like column
            address_cols = [c for c in df.columns if "address" in c.lower()]
            if not address_cols:
                print("[ERROR] No address-like column found in data.")
                continue
            addr_col = address_cols[0]
            print(f"[INFO] Using column '{addr_col}' for address search.")

            query = input(
                f"Type part of the address to search in '{addr_col}': "
            ).strip()
            if not query:
                print("Empty query; try again.")
                continue

            matches = df[df[addr_col].astype(str).str.contains(
                query, case=False, na=False
            )]

            if matches.empty:
                print("No matches found.")
                continue

            max_show = 10
            to_show = matches.head(max_show).reset_index()
            print(f"\nFound {len(matches)} matching rows, showing first {max_show}:")
            for _, r in to_show.iterrows():
                idx = r["index"]
                addr = r[addr_col]
                # Try to show city if present
                city_val = None
                for cc in ["city", "City", "CITY"]:
                    if cc in df.columns:
                        city_val = r[cc]
                        break
                price_str = ""
                if "sale_price" in df.columns and not pd.isna(r["sale_price"]):
                    price_str = f" | sale_price=${r['sale_price']:,.0f}"
                city_part = f" - {city_val}" if city_val is not None else ""
                print(f"  RowID {idx}: {addr}{city_part}{price_str}")

            chosen_id = input(
                "Enter RowID to use (or 'c' to cancel and go back): "
            ).strip()
            if chosen_id.lower() == "c":
                continue
            if not chosen_id.isdigit():
                print("Invalid RowID.")
                continue
            idx_int = int(chosen_id)
            if idx_int not in df.index:
                print("RowID not in dataframe.")
                continue
            row = df.loc[idx_int]
            return row

        else:
            print("Invalid choice; please enter 1, 2, or 3.")


def main():
    # ------------------------------------------------------------------
    # 1. Load trained model bundle
    # ------------------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "georgia_gbdt_model.joblib")

    if not os.path.exists(model_path):
        print(f"[ERROR] Trained model file not found:\n  {model_path}")
        print("Run GeorgiaRealEstateModel.py first to train and save the model.")
        sys.exit(1)

    bundle = load(model_path)
    pipeline = bundle["pipeline"]
    numeric_cols = bundle["numeric_cols"]
    categorical_cols = bundle["categorical_cols"]

    print(f"[INFO] Loaded trained model from:\n  {model_path}")
    print(f"[INFO] Numeric features: {len(numeric_cols)}")
    print(f"[INFO] Categorical features: {len(categorical_cols)}")

    # ------------------------------------------------------------------
    # 2. Load processed CSVs to pick properties from
    # ------------------------------------------------------------------
    processed_dir = get_processed_directory()
    print(f"\n[INFO] Processed data folder: {processed_dir}")

    csv_files = list_processed_csvs(processed_dir)
    if not csv_files:
        print("[ERROR] No processed CSVs found. Did you run PropertyProcessed.py?")
        sys.exit(1)

    chosen_files = choose_files(csv_files, processed_dir)

    print("\n[INFO] Using these processed files:")
    for name in chosen_files:
        print(f"  - {name}")

    df_raw = load_and_combine(processed_dir, chosen_files)

    # Make sure sale_price is numeric where present
    if "sale_price" in df_raw.columns:
        df_raw["sale_price"] = pd.to_numeric(df_raw["sale_price"], errors="coerce")

    # Add time-based features
    df = engineer_time_features(df_raw)

    # Ensure all categorical columns exist and are strings
    for col in categorical_cols:
        if col not in df.columns:
            df[col] = np.nan
    if categorical_cols:
        df[categorical_cols] = df[categorical_cols].astype(str)

    # ------------------------------------------------------------------
    # 3. Interactive loop: pick a property and predict its price
    # ------------------------------------------------------------------
    while True:
        row = select_property(df)

        # Build feature row with correct columns
        feature_cols = numeric_cols + categorical_cols
        x_dict = {}
        for col in feature_cols:
            if col in row.index:
                x_dict[col] = [row[col]]
            else:
                # Feature was present during training but not in this row → NaN
                x_dict[col] = [np.nan]

        X_row = pd.DataFrame(x_dict)

        # Prediction using original sale_date-based features
        log_pred = pipeline.predict(X_row)[0]
        price_pred = float(np.exp(log_pred))

        # "Today" prediction: override time features (approx)
        price_pred_today = None
        time_cols = {"sale_year", "sale_month", "sale_quarter", "sale_year_month"}
        if time_cols.issubset(set(feature_cols)):
            today = date.today()
            year = today.year
            month = today.month
            quarter = (month - 1) // 3 + 1

            X_row_today = X_row.copy()
            if "sale_year" in X_row_today.columns:
                X_row_today["sale_year"] = year
            if "sale_month" in X_row_today.columns:
                X_row_today["sale_month"] = month
            if "sale_quarter" in X_row_today.columns:
                X_row_today["sale_quarter"] = quarter
            if "sale_year_month" in X_row_today.columns:
                X_row_today["sale_year_month"] = year * 12 + month

            log_pred_today = pipeline.predict(X_row_today)[0]
            price_pred_today = float(np.exp(log_pred_today))

        # ------------------------------------------------------------------
        # 4. Print results
        # ------------------------------------------------------------------
        print("\n================= PREDICTION RESULT =================")

        # Show some basic identifying info
        for col in [
            "property_address",
            "site_address",
            "address",
            "mail_address",
        ]:
            if col in df.columns and pd.notna(row.get(col, None)):
                print(f"Address ({col}): {row[col]}")
                break

        for col in ["city", "City", "CITY"]:
            if col in df.columns and pd.notna(row.get(col, None)):
                print(f"City: {row[col]}")
                break

        for col in ["state", "State", "STATE"]:
            if col in df.columns and pd.notna(row.get(col, None)):
                print(f"State: {row[col]}")
                break

        if "zip" in df.columns and pd.notna(row.get("zip", None)):
            print(f"ZIP: {row['zip']}")

        if "sqft" in df.columns and pd.notna(row.get("sqft", None)):
            print(f"Sqft: {row['sqft']}")

        if "year_built" in df.columns and pd.notna(row.get("year_built", None)):
            print(f"Year built: {row['year_built']}")

        if "sale_date" in df.columns and pd.notna(row.get("sale_date", None)):
            print(f"Recorded sale_date: {row['sale_date']}")

        if "sale_price" in df.columns and pd.notna(row.get("sale_price", None)):
            print(f"Recorded sale_price: ${row['sale_price']:,.0f}")

        print(f"\nModel predicted price (using original sale_date features): "
              f"${price_pred:,.0f}")

        if price_pred_today is not None:
            print(
                f"Model predicted price (using TODAY's date for time features): "
                f"${price_pred_today:,.0f}"
            )
            print(
                "  Note: this 'today' estimate only updates sale_year/month/quarter.\n"
                "  It does NOT yet include external macro features "
                "(interest rates, inflation, etc.)."
            )

        # Ask whether to do another prediction
        again = input("\nWould you like to predict another property? [Y/N]: ").strip()
        if again.lower() not in ("y", "yes"):
            print("Done")
            break


if __name__ == "__main__":
    main()
