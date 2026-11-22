
"""
KeyGuard - CMU Pairwise Behavioral Authentication Model

Trains machine-learning classifiers to distinguish:
1 = Genuine: two samples from the same subject
0 = Impostor: two samples from different subjects

Uses absolute differences between the 31 CMU timing features.
Uses earlier sessions for training and later sessions for testing.
"""

import os
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from live_feature_processor import CMU_FEATURES


# =========================================================
# 1. PROJECT PATHS AND SETTINGS
# =========================================================

DATASET_PATH = os.path.join(
    "dataset",
    "cleaned_dataset.csv"
)

MODEL_DIR = "models"
REPORT_DIR = os.path.join(
    MODEL_DIR,
    "authentication_reports"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

RANDOM_STATE = 42

# Number of pairs generated for each class in each split.
PAIRS_PER_CLASS = 20000

# Number of latest sessions reserved for testing.
TEST_SESSION_COUNT = 2


# =========================================================
# 2. LOAD CMU DATASET
# =========================================================

def load_dataset():

    print("\n========== LOADING CMU DATASET ==========")

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    required_columns = (
        ["subject", "sessionIndex", "rep"]
        + CMU_FEATURES
    )

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required dataset columns: {missing}"
        )

    df = df[required_columns].copy()

    for column in CMU_FEATURES:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        subset=CMU_FEATURES + ["subject", "sessionIndex"]
    )

    if df.empty:
        raise ValueError("No valid dataset samples found.")

    print("Dataset shape:", df.shape)
    print("Subjects:", df["subject"].nunique())
    print("Sessions:", sorted(df["sessionIndex"].unique()))
    print("Timing features:", len(CMU_FEATURES))

    return df


# =========================================================
# 3. SESSION-AWARE DATA SPLIT
# =========================================================

def split_by_session(df):

    print("\n========== SESSION-AWARE SPLIT ==========")

    sessions = sorted(df["sessionIndex"].unique())

    if len(sessions) <= TEST_SESSION_COUNT:
        raise ValueError(
            "Not enough sessions for the selected test split."
        )

    test_sessions = sessions[-TEST_SESSION_COUNT:]
    train_sessions = sessions[:-TEST_SESSION_COUNT]

    train_df = df[
        df["sessionIndex"].isin(train_sessions)
    ].copy()

    test_df = df[
        df["sessionIndex"].isin(test_sessions)
    ].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Training or testing data is empty.")

    print("Training sessions:", train_sessions)
    print("Testing sessions:", test_sessions)
    print("Training samples:", len(train_df))
    print("Testing samples:", len(test_df))

    if set(train_df["sessionIndex"]).intersection(
        set(test_df["sessionIndex"])
    ):
        raise ValueError("Session leakage detected.")

    if set(train_df["subject"]) != set(test_df["subject"]):
        raise ValueError(
            "Every subject must appear in both partitions."
        )

    return train_df, test_df


# =========================================================
# 4. GENERATE GENUINE AND IMPOSTOR PAIRS
# =========================================================

def generate_pairs(df, pairs_per_class, random_state):

    """
    Generate balanced pairs.

    Genuine pair:
        Both samples belong to the same subject.

    Impostor pair:
        Samples belong to different subjects.

    Pair feature:
        Absolute difference between the 31 CMU features.
    """

    rng = np.random.default_rng(random_state)

    X = df[CMU_FEATURES].to_numpy(dtype=np.float32)
    subjects = df["subject"].to_numpy()

    subject_indices = {}

    for index, subject in enumerate(subjects):
        subject_indices.setdefault(
            subject, []
        ).append(index)

    valid_subjects = [
        subject
        for subject, indices in subject_indices.items()
        if len(indices) >= 2
    ]

    if len(valid_subjects) < 2:
        raise ValueError(
            "At least two subjects with two samples each are required."
        )

    subject_array = np.array(valid_subjects, dtype=object)

    pair_features = []
    pair_labels = []

    # -----------------------------------------------------
    # Genuine pairs: same subject
    # -----------------------------------------------------

    for _ in range(pairs_per_class):

        subject = rng.choice(subject_array)

        indices = subject_indices[subject]

        first, second = rng.choice(
            indices,
            size=2,
            replace=False
        )

        difference = np.abs(
            X[first] - X[second]
        )

        pair_features.append(difference)
        pair_labels.append(1)

    # -----------------------------------------------------
    # Impostor pairs: different subjects
    # -----------------------------------------------------

    for _ in range(pairs_per_class):

        first_subject, second_subject = rng.choice(
            subject_array,
            size=2,
            replace=False
        )

        first = rng.choice(
            subject_indices[first_subject]
        )

        second = rng.choice(
            subject_indices[second_subject]
        )

        difference = np.abs(
            X[first] - X[second]
        )

        pair_features.append(difference)
        pair_labels.append(0)

    pair_features = np.asarray(
        pair_features,
        dtype=np.float32
    )

    pair_labels = np.asarray(
        pair_labels,
        dtype=np.int32
    )

    # Shuffle the generated pairs.
    order = rng.permutation(len(pair_labels))

    pair_features = pair_features[order]
    pair_labels = pair_labels[order]

    print("\nGenerated pairs:")
    print("Total pairs:", len(pair_labels))
    print("Genuine pairs:", int(np.sum(pair_labels == 1)))
    print("Impostor pairs:", int(np.sum(pair_labels == 0)))
    print("Pair features:", pair_features.shape[1])

    return pair_features, pair_labels


# =========================================================
# 5. EVALUATE CLASSIFIER
# =========================================================

def evaluate_model(model, X_test, y_test, model_name):

    print(f"\n========== {model_name} EVALUATION ==========")

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    report = classification_report(
        y_test,
        predictions,
        target_names=["Impostor", "Genuine"],
        zero_division=0
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))

    print("\nClassification Report:")
    print(report)

    print("Confusion Matrix:")
    print(matrix)

    # Save report.
    report_path = os.path.join(
        REPORT_DIR,
        f"{model_name.lower().replace(' ', '_')}_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(f"Model: {model_name}\n")
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"Precision: {precision:.4f}\n")
        file.write(f"Recall: {recall:.4f}\n")
        file.write(f"F1 Score: {f1:.4f}\n\n")
        file.write(report)
        file.write("\nConfusion Matrix:\n")
        file.write(str(matrix))

    # Save confusion matrix.
    matrix_path = os.path.join(
        REPORT_DIR,
        f"{model_name.lower().replace(' ', '_')}_confusion_matrix.csv"
    )

    pd.DataFrame(
        matrix,
        index=["Actual Impostor", "Actual Genuine"],
        columns=["Predicted Impostor", "Predicted Genuine"]
    ).to_csv(matrix_path)

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }


# =========================================================
# 6. TRAIN RANDOM FOREST
# =========================================================

def train_random_forest(
    X_train,
    y_train,
    X_test,
    y_test
):

    print("\n========== TRAINING RANDOM FOREST ==========")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced"
    )

    start_time = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    print(
        "Training time:",
        round(training_time, 2),
        "seconds"
    )

    model_path = os.path.join(
        MODEL_DIR,
        "random_forest_authentication.joblib"
    )

    joblib.dump(model, model_path)

    print("Saved model:", model_path)

    importance = pd.DataFrame({
        "Feature": CMU_FEATURES,
        "Importance": model.feature_importances_
    }).sort_values(
        by="Importance",
        ascending=False
    )

    importance_path = os.path.join(
        REPORT_DIR,
        "random_forest_feature_importance.csv"
    )

    importance.to_csv(
        importance_path,
        index=False
    )

    print("\nTop 10 important pair features:")
    print(importance.head(10).to_string(index=False))

    return model, evaluate_model(
        model,
        X_test,
        y_test,
        "Random Forest"
    )


# =========================================================
# 7. TRAIN SVM
# =========================================================

def train_svm(
    X_train,
    y_train,
    X_test,
    y_test
):

    print("\n========== TRAINING SVM ==========")

    model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "svm",
            SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                class_weight="balanced",
                probability=True,
                random_state=RANDOM_STATE
            )
        )
    ])

    start_time = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    print(
        "Training time:",
        round(training_time, 2),
        "seconds"
    )

    model_path = os.path.join(
        MODEL_DIR,
        "svm_authentication.joblib"
    )

    joblib.dump(model, model_path)

    print("Saved model:", model_path)

    return model, evaluate_model(
        model,
        X_test,
        y_test,
        "SVM"
    )


# =========================================================
# 8. MAIN TRAINING FUNCTION
# =========================================================


def main():

    print("\n============================================")
    print(" KEYGUARD - RANDOM FOREST AUTHENTICATION")
    print("============================================")

    # Load CMU dataset
    df = load_dataset()

    # Session-aware split
    train_df, test_df = split_by_session(df)

    # Generate training pairs
    print("\n========== GENERATING TRAINING PAIRS ==========")

    X_train, y_train = generate_pairs(
        train_df,
        PAIRS_PER_CLASS,
        RANDOM_STATE
    )

    # Generate testing pairs
    print("\n========== GENERATING TESTING PAIRS ==========")

    X_test, y_test = generate_pairs(
        test_df,
        PAIRS_PER_CLASS,
        RANDOM_STATE + 1
    )

    # Train Random Forest only
    rf_model, rf_results = train_random_forest(
        X_train,
        y_train,
        X_test,
        y_test
    )

    # Save final authentication model
    final_model_path = os.path.join(
        MODEL_DIR,
        "authentication_model.joblib"
    )

    joblib.dump(rf_model, final_model_path)

    # Save model metadata
    metadata = {
        "feature_names": CMU_FEATURES,
        "positive_class": 1,
        "positive_class_name": "genuine",
        "negative_class": 0,
        "negative_class_name": "impostor",
        "best_model": "Random Forest"
    }

    metadata_path = os.path.join(
        MODEL_DIR,
        "authentication_model_metadata.joblib"
    )

    joblib.dump(metadata, metadata_path)

    # Save comparison report
    results_df = pd.DataFrame([rf_results])

    results_path = os.path.join(
        REPORT_DIR,
        "authentication_model_comparison.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    print("\n========== TRAINING COMPLETED ==========")
    print("Selected model: Random Forest")
    print("Final model:", final_model_path)
    print("Metadata:", metadata_path)
    print("Comparison report:", results_path)


if __name__ == "__main__":
    main()


