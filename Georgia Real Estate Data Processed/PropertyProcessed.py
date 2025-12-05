"""
Goal:
- Read a property CSV from the "Georgia Real Estate Data and Code" folder.
- Remove rows that do not have both:
    - sale_date
    - sale_price
"""

import os
import sys
import pandas as pd


def get_base_directories():
    """
    Determine the input and output directories based on the script location.

    Script is expected to live in:
        .../Georgia Real Estate Data Processed/PropertyProcessed.py

    Raw CSVs are expected to live in:
        .../Georgia Real Estate Data and Code/
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)

    raw_dir = os.path.join(parent_dir, "Georgia Real Estate Data and Code")
    processed_dir = script_dir  # "Georgia Real Estate Data Processed"

    return raw_dir, processed_dir


def list_csv_files(raw_dir):
    """
    Return a sorted list of CSV file names in the input directory.
    """
    csv_files = []

    if not os.path.isdir(raw_dir):
        print(f"[ERROR] Input folder not found: {raw_dir}")
        return csv_files

    for name in os.listdir(raw_dir):
        lower_name = name.lower()
        if lower_name.endswith(".csv"):
            csv_files.append(name)

    csv_files.sort()
    return csv_files


def build_output_name(input_name):
    """
    Given 'Athens_PropertyData.csv', return 'Athens_PropertyData_processed.csv'.
    """
    base, ext = os.path.splitext(input_name)
    if ext == "":
        ext = ".csv"
    output_name = base + "_Processed" + ext
    return output_name


def load_and_filter(input_path):
    """
    Load the CSV, drop rows missing sale_date or sale_price, and return
    the filtered DataFrame.
    """
    print(f"[INFO] Reading CSV: {input_path}")
    df = pd.read_csv(input_path)

    before_rows = len(df)

    # Ensure required columns are present
    required_cols = ["sale_date", "sale_price"]
    missing_cols = []

    for col in required_cols:
        if col not in df.columns:
            missing_cols.append(col)

    if len(missing_cols) > 0:
        raise ValueError(
            f"Missing required column(s) in {os.path.basename(input_path)}: {missing_cols}"
        )

    # Optional: treat empty strings as missing for sale_date
    if df["sale_date"].dtype == object:
        df["sale_date"] = df["sale_date"].replace("", pd.NA)

    # Ensure sale_price is numeric; coerce bad values to NaN
    df["sale_price"] = pd.to_numeric(df["sale_price"], errors="coerce")

    # Drop rows where either sale_date or sale_price is missing
    df_filtered = df.dropna(subset=["sale_date", "sale_price"])

    after_rows = len(df_filtered)
    removed = before_rows - after_rows

    print(f"[INFO] Rows before filtering : {before_rows}")
    print(f"[INFO] Rows after filtering  : {after_rows}")
    print(f"[INFO] Rows removed (missing sale_date or sale_price): {removed}")

    return df_filtered


def choose_file(csv_files, raw_dir):
    """
    Let the user choose which CSV to process.

    The user can:
    - Type a number corresponding to the displayed list, OR
    - Type a file name (with or without .csv).
    """

    print("\nAvailable CSV files in:")
    print(f"  {raw_dir}\n")

    for index, name in enumerate(csv_files, start=1):
        print(f"  {index}. {name}")

    # Option 1: file or index provided as a command-line argument
    if len(sys.argv) > 1:
        user_input = sys.argv[1]
        print(f"\n[INFO] Using command-line argument: {user_input}")
    else:
        user_input = input(
            "\nType the CSV file name (with or without .csv) "
            "or the number from the list above: "
        ).strip()

    # If the user typed a number, map it to the list index
    if user_input.isdigit():
        chosen_index = int(user_input)
        if chosen_index < 1 or chosen_index > len(csv_files):
            print(f"[ERROR] Invalid index: {chosen_index}")
            sys.exit(1)
        chosen_name = csv_files[chosen_index - 1]
        return chosen_name

    # Otherwise, treat as a file name; append .csv if needed
    candidate_name = user_input
    if not candidate_name.lower().endswith(".csv"):
        candidate_name = candidate_name + ".csv"

    if candidate_name not in csv_files:
        print(
            f"[ERROR] Could not find '{candidate_name}' "
            f"in the input folder: {raw_dir}"
        )
        sys.exit(1)

    return candidate_name


def main():
    raw_dir, processed_dir = get_base_directories()

    print(f"[INFO] Input folder (raw CSVs)     : {raw_dir}")
    print(f"[INFO] Output folder (processed)   : {processed_dir}")

    csv_files = list_csv_files(raw_dir)
    if len(csv_files) == 0:
        print("[ERROR] No CSV files found in the input folder.")
        sys.exit(1)

    chosen_name = choose_file(csv_files, raw_dir)

    input_path = os.path.join(raw_dir, chosen_name)
    output_name = build_output_name(chosen_name)
    output_path = os.path.join(processed_dir, output_name)

    try:
        df_filtered = load_and_filter(input_path)
    except Exception as e:
        print(f"[ERROR] Failed to process file: {e!r}")
        sys.exit(1)

    print(f"[INFO] Saving processed CSV to: {output_path}")
    df_filtered.to_csv(output_path, index=False)

    print("\n[DONE] Processing complete.")
    print(f"[DONE] Created file: {output_name}")


if __name__ == "__main__":
    main()
