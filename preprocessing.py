

import pandas as pd
from pathlib import Path


# Get project directory
BASE_DIR = Path(__file__).resolve().parent

# Dataset location
DATASET_PATH = BASE_DIR / "dataset" / "DSL-StrongPasswordData.csv"


# --------------------------------
# 1. LOAD DATASET
# --------------------------------

def load_dataset():

    if not DATASET_PATH.exists():
        raise FileNotFoundError("Dataset file not found!")

    df = pd.read_csv(DATASET_PATH)

    print("Dataset loaded successfully!")

    return df


# --------------------------------
# 2. INSPECT DATASET
# --------------------------------

def inspect_dataset(df):

    print("\n========== DATASET INFORMATION ==========")

    print("\nDataset Shape:")
    print(df.shape)

    print("\nFirst Five Records:")
    print(df.head())

    print("\nMissing Values:")
    print(df.isnull().sum().sum())

    print("\nDuplicate Records:")
    print(df.duplicated().sum())

    print("\nNumber of Unique Subjects:")
    print(df["subject"].nunique())


# --------------------------------
# 3. CLEAN DATASET
# --------------------------------

def clean_dataset(df):

    print("\n========== DATA CLEANING ==========")

    df = df.copy()

    # Check missing values
    print("\nMissing values before cleaning:")
    print(df.isnull().sum().sum())

    # Check duplicates
    print("\nDuplicate records before cleaning:")
    print(df.duplicated().sum())

    # Remove exact duplicate rows
    df = df.drop_duplicates()

    # Identify timing features
    metadata_columns = ["subject", "sessionIndex", "rep"]

    feature_columns = [
        col for col in df.columns
        if col not in metadata_columns
    ]

    # Convert timing columns into numeric format
    for col in feature_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill missing timing values using median
    for col in feature_columns:
        median_value = df[col].median()

        if pd.notna(median_value):
            df[col] = df[col].fillna(median_value)

    print("\nMissing values after cleaning:")
    print(df.isnull().sum().sum())

    print("\nDataset shape after cleaning:")
    print(df.shape)

    print("\nData cleaning completed successfully!")

    return df


# --------------------------------
# 4. PREPARE FEATURES
# --------------------------------

def prepare_features(df):

    print("\n========== FEATURE PREPARATION ==========")

    X = df.drop(
        columns=["subject", "sessionIndex", "rep"]
    )

    y = df["subject"]

    print("\nX shape:", X.shape)
    print("y shape:", y.shape)

    print("Number of features:", X.shape[1])
    print("Number of users:", y.nunique())

    return X, y


# --------------------------------
# 5. MAIN EXECUTION
# --------------------------------

if __name__ == "__main__":

    print("\nStarting KeyGuard preprocessing...\n")

    data = load_dataset()

    inspect_dataset(data)

    cleaned_data = clean_dataset(data)

    X, y = prepare_features(cleaned_data)

    # Save cleaned dataset
    OUTPUT_PATH = BASE_DIR / "dataset" / "cleaned_dataset.csv"

    cleaned_data.to_csv(OUTPUT_PATH, index=False)

    print("\n===================================")
    print("Cleaned dataset saved successfully!")
    print("Saved location:", OUTPUT_PATH)
    print("===================================")
