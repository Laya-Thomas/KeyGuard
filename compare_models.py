
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "dataset" / "cleaned_dataset.csv"
ENGINEERED_PATH = BASE_DIR / "dataset" / "engineered_features.csv"

MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = MODELS_DIR / "comparison_reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# LOAD DATASETS
# ==========================================

def load_data():

    original = pd.read_csv(DATASET_PATH)
    engineered = pd.read_csv(ENGINEERED_PATH)

    print("Original dataset:", original.shape)
    print("Engineered dataset:", engineered.shape)

    return original, engineered


# ==========================================
# SESSION-AWARE SPLIT
# ==========================================

def session_aware_split(df):

    train_indices = []
    test_indices = []

    for subject in df["subject"].unique():

        subject_data = df[df["subject"] == subject]

        sessions = sorted(
            subject_data["sessionIndex"].unique()
        )

        split_point = int(len(sessions) * 0.75)

        train_sessions = sessions[:split_point]
        test_sessions = sessions[split_point:]

        train_indices.extend(
            subject_data[
                subject_data["sessionIndex"].isin(train_sessions)
            ].index.tolist()
        )

        test_indices.extend(
            subject_data[
                subject_data["sessionIndex"].isin(test_sessions)
            ].index.tolist()
        )

    return train_indices, test_indices


# ==========================================
# PREPARE FEATURES
# ==========================================

def prepare_features(df, feature_columns):

    X = df[feature_columns].copy()

    # Replace invalid or infinite values
    X = X.replace([np.inf, -np.inf], np.nan)

    return X


# ==========================================
# MODEL COMPARISON
# ==========================================

def evaluate_model(name, model, X_train, X_test, y_train, y_test):

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test, predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test, predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test, predictions,
        average="weighted",
        zero_division=0
    )

    print(f"{name} Accuracy: {accuracy * 100:.2f}%")

    return {
        "Model": name,
        "Accuracy": accuracy * 100,
        "Precision": precision * 100,
        "Recall": recall * 100,
        "F1 Score": f1 * 100
    }


# ==========================================
# MAIN EXPERIMENT
# ==========================================

def main():

    print("\n========== KEYGUARD MODEL COMPARISON ==========")

    original, engineered = load_data()

    metadata = ["subject", "sessionIndex", "rep"]

    original_features = [
        col for col in original.columns
        if col not in metadata
    ]

    engineered_features = [
        col for col in engineered.columns
        if col not in metadata
    ]

    # Create session-aware split using original dataset
    train_idx, test_idx = session_aware_split(original)

    y_train = original.loc[train_idx, "subject"]
    y_test = original.loc[test_idx, "subject"]

    # Use the same row split for all experiments
    original_train = original.loc[train_idx]
    original_test = original.loc[test_idx]

    engineered_train = engineered.loc[train_idx]
    engineered_test = engineered.loc[test_idx]

    print("\nTraining samples:", len(train_idx))
    print("Testing samples:", len(test_idx))

    results = []

    # ======================================
    # EXPERIMENT 1: ORIGINAL 31 FEATURES
    # ======================================

    print("\n\nEXPERIMENT 1: ORIGINAL FEATURES")

    X_train = prepare_features(
        original_train, original_features
    )

    X_test = prepare_features(
        original_test, original_features
    )

    rf = Pipeline([
        ("imputer", __import__("sklearn.impute").impute.SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        ))
    ])

    svm = Pipeline([
        ("imputer", __import__("sklearn.impute").impute.SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", SVC(kernel="rbf", C=1, gamma="scale"))
    ])

    results.append(evaluate_model(
        "RF - Original 31",
        rf, X_train, X_test, y_train, y_test
    ))

    results.append(evaluate_model(
        "SVM - Original 31",
        svm, X_train, X_test, y_train, y_test
    ))

    # ======================================
    # EXPERIMENT 2: ALL 36 FEATURES
    # ======================================

    print("\n\nEXPERIMENT 2: ENGINEERED FEATURES")

    X_train = prepare_features(
        engineered_train, engineered_features
    )

    X_test = prepare_features(
        engineered_test, engineered_features
    )

    rf = Pipeline([
        ("imputer", __import__("sklearn.impute").impute.SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        ))
    ])

    svm = Pipeline([
        ("imputer", __import__("sklearn.impute").impute.SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", SVC(kernel="rbf", C=1, gamma="scale"))
    ])

    results.append(evaluate_model(
        "RF - Engineered 36",
        rf, X_train, X_test, y_train, y_test
    ))

    results.append(evaluate_model(
        "SVM - Engineered 36",
        svm, X_train, X_test, y_train, y_test
    ))

    # ======================================
    # EXPERIMENT 3: SELECTED FEATURES
    # ======================================

    print("\n\nEXPERIMENT 3: SELECTED FEATURES")

    # Fit feature selector on training data only
    selector_pipeline = Pipeline([
        ("imputer", __import__("sklearn.impute").impute.SimpleImputer(strategy="median")),
        ("selector", SelectFromModel(
            RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced"
            ),
            threshold="median"
        ))
    ])

    X_train_all = prepare_features(
        engineered_train, engineered_features
    )

    X_test_all = prepare_features(
        engineered_test, engineered_features
    )

    X_train_selected = selector_pipeline.fit_transform(
        X_train_all, y_train
    )

    X_test_selected = selector_pipeline.transform(
        X_test_all
    )

    selected_names = np.array(engineered_features)[
        selector_pipeline.named_steps["selector"].get_support()
    ]

    print("\nSelected feature count:", len(selected_names))
    print("Selected features:", list(selected_names))

    rf_selected = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    svm_selected = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(kernel="rbf", C=1, gamma="scale"))
    ])

    results.append(evaluate_model(
        "RF - Selected Features",
        rf_selected,
        X_train_selected,
        X_test_selected,
        y_train,
        y_test
    ))

    results.append(evaluate_model(
        "SVM - Selected Features",
        svm_selected,
        X_train_selected,
        X_test_selected,
        y_train,
        y_test
    ))

    # ======================================
    # SAVE RESULTS
    # ======================================

    results_df = pd.DataFrame(results)

    results_path = REPORTS_DIR / "model_comparison.csv"

    results_df.to_csv(results_path, index=False)

    print("\n\n========== FINAL COMPARISON ==========")

    print(results_df.round(2).to_string(index=False))

    print("\nResults saved at:")
    print(results_path)

    print("\nModel comparison completed successfully!")


if __name__ == "__main__":
    main()
