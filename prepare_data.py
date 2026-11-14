import pandas as pd
import numpy as np
import os
import glob

DATA_FOLDER = "data"
OUTPUT_FILE = "data/mini_cicids2017.csv"

# Maximum number of examples we keep for each class
MAX_PER_CLASS = 5000

TARGET_CLASSES = [
    "BENIGN",
    "DoS",
    "DDoS",
    "Port Scanning",
    "Brute Force",
    "Botnet",
    "SQL Injection"
]


def clean_label(label):
    """
    Convert the original CICIDS2017 labels
    into our simplified project labels.
    """

    label = str(label).strip()

    label_lower = label.lower()

    # Normal traffic
    if label_lower == "benign":
        return "BENIGN"

    # DoS attacks
    if "dos" in label_lower and "ddos" not in label_lower:
        return "DoS"

    # DDoS
    if "ddos" in label_lower:
        return "DDoS"

    # Port scanning
    if "portscan" in label_lower:
        return "Port Scanning"

    # Brute force
    if "patator" in label_lower:
        return "Brute Force"

    if "web attack" in label_lower and "brute force" in label_lower:
        return "Brute Force"

    # Botnet
    if "bot" in label_lower:
        return "Botnet"

    # SQL Injection
    if "sql" in label_lower and "injection" in label_lower:
        return "SQL Injection"

    return None


def find_label_column(df):
    """
    Find the Label column even if there are extra spaces.
    """

    for column in df.columns:
        if str(column).strip().lower() == "label":
            return column

    raise ValueError("Could not find Label column.")


def process_file(file_path, collected):
    print(f"\nReading: {os.path.basename(file_path)}")

    # Read the file in chunks so we don't load the whole thing.
    for chunk in pd.read_csv(
        file_path,
        chunksize=50000,
        low_memory=False
    ):

        label_column = find_label_column(chunk)

        # Create simplified labels
        chunk["ProjectLabel"] = chunk[label_column].apply(clean_label)

        # Remove classes we don't need
        chunk = chunk[chunk["ProjectLabel"].notna()]

        if len(chunk) == 0:
            continue

        # Keep only the number of rows we need
        for class_name in TARGET_CLASSES:

            current_count = len(collected[class_name])

            if current_count >= MAX_PER_CLASS:
                continue

            class_rows = chunk[
                chunk["ProjectLabel"] == class_name
            ]

            if len(class_rows) == 0:
                continue

            remaining = MAX_PER_CLASS - current_count

            class_rows = class_rows.head(remaining)

            collected[class_name].append(class_rows)

        # Check whether all classes are full
        if all(
            sum(len(x) for x in collected[class_name])
            >= MAX_PER_CLASS
            for class_name in TARGET_CLASSES
            if class_name != "SQL Injection"
        ):
            # We still need to scan for SQL Injection,
            # so don't stop the entire process here.
            pass


def main():

    csv_files = glob.glob(
        os.path.join(DATA_FOLDER, "*.csv")
    )

    if not csv_files:
        print("ERROR: No CSV files found inside the data folder.")
        return

    print(f"Found {len(csv_files)} CSV files.")

    collected = {
        class_name: []
        for class_name in TARGET_CLASSES
    }

    for file_path in csv_files:
        process_file(file_path, collected)

    final_parts = []

    print("\nCollected data:")

    for class_name in TARGET_CLASSES:

        if collected[class_name]:

            class_df = pd.concat(
                collected[class_name],
                ignore_index=True
            )

            # Final safety limit
            class_df = class_df.head(MAX_PER_CLASS)

            print(
                f"{class_name}: {len(class_df)} rows"
            )

            final_parts.append(class_df)

        else:
            print(
                f"{class_name}: 0 rows"
            )

    if not final_parts:
        print("No data collected.")
        return

    final_df = pd.concat(
        final_parts,
        ignore_index=True
    )

    # Remove duplicate rows
    final_df = final_df.drop_duplicates()

    # Replace infinity values
    final_df = final_df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Drop rows containing missing values
    final_df = final_df.dropna()

    # Shuffle
    final_df = final_df.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    os.makedirs(DATA_FOLDER, exist_ok=True)

    final_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n===================================")
    print("DATASET CREATED")
    print("===================================")

    print(f"Rows: {len(final_df)}")
    print(f"Columns: {len(final_df.columns)}")

    print("\nClass distribution:")
    print(
        final_df["ProjectLabel"].value_counts()
    )

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()