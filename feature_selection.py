
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel


# Project directory
BASE_DIR = Path(__file__).resolve().parent

# Dataset paths
DATASET_PATH = BASE_DIR / "dataset" / "engineered_features.csv"

IMPORTANCE_PATH = BASE_DIR / "models" / "feature_importance.csv"

SELECTED_PATH = BASE_DIR / "dataset" / "selected_features.csv"


def load_dataset():

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    print("Engineered dataset loaded successfully!")
    print("Dataset shape:", df.shape)

    return df


def select_features(df):

    print("\n========== FEATURE SELECTION ==========")

    metadata_columns = ["subject", "sessionIndex", "rep"]

    feature_columns = [
        col for col in df.columns
        if col not in metadata_columns
    ]

    X = df[feature_columns]
    y = df["subject"]

    # Split data before fitting the feature selector
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # Train Random Forest on training data only
    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    rf.fit(X_train, y_train)

    # Calculate feature importance
    importance_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": rf.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    # Save feature importance ranking
    importance_df.to_csv(
        IMPORTANCE_PATH,
        index=False
    )

    # Select features with importance above the median
    selector = SelectFromModel(
        rf,
        threshold="median",
        prefit=True
    )

    selected_features = X.columns[
        selector.get_support()
    ].tolist()

    print("\nOriginal number of features:", len(feature_columns))
    print("Selected number of features:", len(selected_features))

    print("\nSelected features:")

    for feature in selected_features:
        print("-", feature)

    # Save selected feature data for inspection
    selected_df = df[
        metadata_columns + selected_features
    ]

    selected_df.to_csv(
        SELECTED_PATH,
        index=False
    )

    print("\nFeature importance saved at:")
    print(IMPORTANCE_PATH)

    print("\nSelected dataset saved at:")
    print(SELECTED_PATH)

    return selected_features


if __name__ == "__main__":

    data = load_dataset()

    selected = select_features(data)

    print("\nFeature selection completed successfully!")
