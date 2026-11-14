import pandas as pd
import os
import glob

# ============================================================
# LARGE DEMO DATASET CREATOR
# Creates a CSV containing:
# 78 ML features + Label
# 7 classes:
# BENIGN
# Botnet
# Brute Force
# DDoS
# DoS
# Port Scanning
# SQL Injection
# ============================================================

print("=" * 60)
print("CREATING LARGE CYBER ATTACK DEMO DATASET")
print("=" * 60)

# ------------------------------------------------------------
# 1. FIND CSV FILE
# ------------------------------------------------------------

print("\nSearching for CSV files...")

csv_files = glob.glob("**/*.csv", recursive=True)

if not csv_files:
    print("ERROR: No CSV files found.")
    print("Put your CICIDS2017/processed CSV inside this project folder.")
    exit()

print("\nCSV files found:")

for i, file in enumerate(csv_files):
    print(f"{i + 1}. {file}")

# ------------------------------------------------------------
# 2. FIND THE CSV THAT CONTAINS THE LABEL COLUMN
# ------------------------------------------------------------

source_file = None
df = None

for file in csv_files:

    try:
        temp = pd.read_csv(file, nrows=5)

        columns_lower = [str(c).strip().lower() for c in temp.columns]

        if "label" in columns_lower:
            source_file = file
            df = pd.read_csv(file)
            break

    except Exception:
        continue

if source_file is None:
    print("\nERROR: Could not find a CSV containing a 'Label' column.")
    exit()

print("\nUsing dataset:")
print(source_file)

print("\nOriginal dataset shape:")
print(df.shape)

# ------------------------------------------------------------
# 3. FIND LABEL COLUMN
# ------------------------------------------------------------

label_column = None

for col in df.columns:

    if str(col).strip().lower() == "label":
        label_column = col
        break

print("\nLabel column:")
print(label_column)

# ------------------------------------------------------------
# 4. CLEAN LABELS
# ------------------------------------------------------------

df[label_column] = (
    df[label_column]
    .astype(str)
    .str.strip()
)

# ------------------------------------------------------------
# 5. KEEP ONLY OUR 7 CLASSES
# ------------------------------------------------------------

classes = [
    "BENIGN",
    "Botnet",
    "Brute Force",
    "DDoS",
    "DoS",
    "Port Scanning",
    "SQL Injection"
]

df = df[df[label_column].isin(classes)].copy()

print("\nClasses available:")
print(df[label_column].value_counts())

# ------------------------------------------------------------
# 6. REMOVE NON-NUMERIC / NON-MODEL COLUMNS
# ------------------------------------------------------------

X = df.drop(columns=[label_column])

# Keep numeric columns only
X = X.select_dtypes(include=["number"])

print("\nNumeric feature count:")
print(X.shape[1])

# ------------------------------------------------------------
# 7. MAKE SURE WE HAVE EXACTLY 78 FEATURES
# ------------------------------------------------------------

if X.shape[1] != 78:

    print("\nERROR!")
    print(f"This dataset contains {X.shape[1]} numeric features.")
    print("Your trained model expects exactly 78 features.")
    print("\nDO NOT CONTINUE.")

    exit()

# Add cleaned labels back
df_clean = X.copy()
df_clean["Label"] = df[label_column].values

# ------------------------------------------------------------
# 8. CREATE LARGE DATASET
# ------------------------------------------------------------

# Number of rows desired for EACH CLASS
ROWS_PER_CLASS = 3000

print("\nCreating demo dataset...")
print(f"Target rows per class: {ROWS_PER_CLASS}")

all_samples = []

for class_name in classes:

    class_data = df_clean[
        df_clean["Label"] == class_name
    ].copy()

    original_count = len(class_data)

    print(
        f"\n{class_name}: "
        f"{original_count} original rows"
    )

    if original_count == 0:

        print("WARNING: Class not found. Skipping.")
        continue

    # --------------------------------------------------------
    # If enough real rows exist:
    # sample without replacement
    # --------------------------------------------------------

    if original_count >= ROWS_PER_CLASS:

        sample = class_data.sample(
            n=ROWS_PER_CLASS,
            random_state=42
        )

    # --------------------------------------------------------
    # If not enough rows:
    # oversample with replacement
    # --------------------------------------------------------

    else:

        print(
            f"Only {original_count} rows available."
        )

        print(
            f"Oversampling to {ROWS_PER_CLASS} rows."
        )

        sample = class_data.sample(
            n=ROWS_PER_CLASS,
            replace=True,
            random_state=42
        )

    all_samples.append(sample)

# ------------------------------------------------------------
# 9. COMBINE ALL CLASSES
# ------------------------------------------------------------

demo_data = pd.concat(
    all_samples,
    ignore_index=True
)

# Shuffle everything
demo_data = demo_data.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# ------------------------------------------------------------
# 10. SAVE
# ------------------------------------------------------------

output_file = "large_testing_data.csv"

demo_data.to_csv(
    output_file,
    index=False
)

# ------------------------------------------------------------
# 11. SHOW RESULTS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("DATASET CREATED SUCCESSFULLY")
print("=" * 60)

print("\nFile:")
print(output_file)

print("\nRows:")
print(len(demo_data))

print("\nColumns:")
print(len(demo_data.columns))

print("\nML features:")
print(len(demo_data.columns) - 1)

print("\nClass distribution:")
print(
    demo_data["Label"].value_counts()
)

print("\nFinal shape:")
print(demo_data.shape)

print("\nFirst 5 rows:")
print(demo_data.head())

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)