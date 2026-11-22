
import pandas as pd
from pathlib import Path


# Project directory
BASE_DIR = Path(__file__).resolve().parent

# Cleaned dataset location
DATASET_PATH = BASE_DIR / "dataset" / "cleaned_dataset.csv"

# Output dataset location
OUTPUT_PATH = BASE_DIR / "dataset" / "engineered_features.csv"


def load_cleaned_dataset():

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    print("Cleaned dataset loaded successfully!")
    print("Dataset shape:", df.shape)

    return df


def engineer_features(df):

    print("\n========== FEATURE ENGINEERING ==========")

    df = df.copy()

    # Original timing feature columns
    metadata_columns = ["subject", "sessionIndex", "rep"]

    feature_columns = [
        col for col in df.columns
        if col not in metadata_columns
    ]

    # Ensure timing features are numeric
    df[feature_columns] = df[feature_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    # Calculate statistical features for each keystroke sample
    df["timing_mean"] = df[feature_columns].mean(axis=1)
    df["timing_std"] = df[feature_columns].std(axis=1)
    df["timing_min"] = df[feature_columns].min(axis=1)
    df["timing_max"] = df[feature_columns].max(axis=1)
    df["timing_median"] = df[feature_columns].median(axis=1)

    # Save the engineered dataset
    df.to_csv(OUTPUT_PATH, index=False)

    print("\nOriginal timing features:", len(feature_columns))
    print("New engineered features: 5")
    print("Total timing features:", len(feature_columns) + 5)
    print("Final dataset shape:", df.shape)

    print("\nNew features:")
    print([
        "timing_mean",
        "timing_std",
        "timing_min",
        "timing_max",
        "timing_median"
    ])

    print("\nEngineered dataset saved at:")
    print(OUTPUT_PATH)

    return df


if __name__ == "__main__":

    data = load_cleaned_dataset()

    engineered_data = engineer_features(data)

    print("\nFeature engineering completed successfully!")
