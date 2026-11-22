
import os
import time
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# 1. PROJECT PATHS
# ==========================================

DATASET_PATH = os.path.join(
    "dataset",
    "cleaned_dataset.csv"
)

MODEL_DIR = "models"
REPORT_DIR = os.path.join("models", "reports")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


# ==========================================
# 2. LOAD DATASET
# ==========================================

def load_data():

    print("\n========== LOADING DATASET ==========")

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    print("Dataset shape:", df.shape)

    # Target variable
    y = df["subject"]

    # Remove target and metadata columns from features
    metadata_columns = [
        "subject",
        "sessionIndex",
        "rep"
    ]

    X = df.drop(columns=metadata_columns)

    # Convert feature values to numeric
    X = X.apply(pd.to_numeric, errors="coerce")

    # Fill any missing feature values with column medians
    X = X.fillna(X.median())

    # Create a unique group for each subject-session pair
    groups = (
        df["subject"].astype(str)
        + "_"
        + df["sessionIndex"].astype(str)
    )

    # Keep the original session number
    sessions = df["sessionIndex"]

    print("Number of samples:", len(X))
    print("Number of features:", X.shape[1])
    print("Number of subjects:", y.nunique())
    print("Number of sessions:", sessions.nunique())

    return X, y, groups, sessions


# ==========================================
# 3. NORMAL RANDOM SPLIT
# ==========================================

def split_data(X, y):

    print("\n========== NORMAL RANDOM SPLIT ==========")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))
    print("Training subjects:", y_train.nunique())
    print("Testing subjects:", y_test.nunique())

    return X_train, X_test, y_train, y_test


# ==========================================
# 4. SESSION-AWARE SPLIT
# ==========================================

def session_aware_split(X, y, groups, sessions):

    print("\n========== SESSION-AWARE SPLIT ==========")

    train_indices = []
    test_indices = []

    # Process each subject separately
    for subject in y.unique():

        subject_indices = y[y == subject].index

        subject_sessions = sessions.loc[subject_indices]

        unique_sessions = sorted(
            subject_sessions.unique()
        )

        if len(unique_sessions) < 2:
            raise ValueError(
                f"Subject {subject} has fewer than 2 sessions."
            )

        # Hold out the latest 25% of sessions for testing
        number_of_test_sessions = max(
            1,
            round(len(unique_sessions) * 0.25)
        )

        test_sessions = unique_sessions[
            -number_of_test_sessions:
        ]

        train_sessions = unique_sessions[
            :-number_of_test_sessions
        ]

        subject_train = subject_indices[
            subject_sessions.isin(train_sessions)
        ]

        subject_test = subject_indices[
            subject_sessions.isin(test_sessions)
        ]

        train_indices.extend(subject_train.tolist())
        test_indices.extend(subject_test.tolist())

    # Sort indices
    train_indices = sorted(train_indices)
    test_indices = sorted(test_indices)

    # Create training and testing datasets
    X_train = X.loc[train_indices]
    X_test = X.loc[test_indices]

    y_train = y.loc[train_indices]
    y_test = y.loc[test_indices]

    # Check for session overlap
    train_groups = set(groups.loc[train_indices])
    test_groups = set(groups.loc[test_indices])

    overlap = train_groups.intersection(test_groups)

    if overlap:
        raise ValueError(
            "Session leakage detected! "
            "Some sessions appear in both training and testing."
        )

    # Check that all subjects are represented
    if set(y_train.unique()) != set(y.unique()):
        raise ValueError(
            "Some subjects are missing from training data."
        )

    if set(y_test.unique()) != set(y.unique()):
        raise ValueError(
            "Some subjects are missing from testing data."
        )

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    print(
        "Training sessions:",
        len(train_groups)
    )

    print(
        "Testing sessions:",
        len(test_groups)
    )

    print(
        "Training subjects:",
        y_train.nunique()
    )

    print(
        "Testing subjects:",
        y_test.nunique()
    )

    print(
        "Shared sessions:",
        len(overlap)
    )

    print(
        "Training percentage:",
        round(len(X_train) / len(X) * 100, 2),
        "%"
    )

    print(
        "Testing percentage:",
        round(len(X_test) / len(X) * 100, 2),
        "%"
    )

    return X_train, X_test, y_train, y_test


# ==========================================
# 5. EVALUATE MODEL
# ==========================================

def evaluate_model(
    model,
    X_test,
    y_test,
    model_name,
    split_name
):

    print(f"\n========== {model_name} RESULTS ==========")
    print("Evaluation:", split_name)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))

    # Classification report
    report = classification_report(
        y_test,
        y_pred,
        zero_division=0
    )

    print("\nClassification Report:\n")
    print(report)

    # Confusion matrix
    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=sorted(y_test.unique())
    )

    # Save classification report
    report_filename = (
        f"{model_name}_{split_name}_report.txt"
    )

    report_path = os.path.join(
        REPORT_DIR,
        report_filename
    )

    with open(report_path, "w") as file:

        file.write(f"Model: {model_name}\n")
        file.write(f"Evaluation: {split_name}\n\n")

        file.write(
            f"Accuracy: {accuracy:.4f}\n"
        )

        file.write(
            f"Precision: {precision:.4f}\n"
        )

        file.write(
            f"Recall: {recall:.4f}\n"
        )

        file.write(
            f"F1 Score: {f1:.4f}\n\n"
        )

        file.write(report)

    # Save confusion matrix
    matrix_filename = (
        f"{model_name}_{split_name}_confusion_matrix.csv"
    )

    matrix_path = os.path.join(
        REPORT_DIR,
        matrix_filename
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=sorted(y_test.unique()),
        columns=sorted(y_test.unique())
    )

    matrix_df.to_csv(matrix_path)

    return {
        "Model": model_name,
        "Evaluation": split_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }


# ==========================================
# 6. TRAIN RANDOM FOREST
# ==========================================

def train_random_forest(
    X_train,
    X_test,
    y_train,
    y_test,
    split_name
):

    print("\n========== TRAINING RANDOM FOREST ==========")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )

    start_time = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    print(
        "Training time:",
        round(training_time, 2),
        "seconds"
    )

    # Save model
    model_filename = (
        f"random_forest_{split_name}.joblib"
    )

    model_path = os.path.join(
        MODEL_DIR,
        model_filename
    )

    joblib.dump(model, model_path)

    print("Model saved:", model_path)

    # Feature importance
    importance = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    print("\nTop 10 Important Features:")
    print(importance.head(10).to_string(index=False))

    importance_path = os.path.join(
        REPORT_DIR,
        f"random_forest_{split_name}_feature_importance.csv"
    )

    importance.to_csv(
        importance_path,
        index=False
    )

    return evaluate_model(
        model,
        X_test,
        y_test,
        "Random Forest",
        split_name
    )


# ==========================================
# 7. TRAIN SVM
# ==========================================

def train_svm(
    X_train,
    X_test,
    y_train,
    y_test,
    split_name
):

    print("\n========== TRAINING SVM ==========")

    model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "svm",
            SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale"
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

    # Save model
    model_filename = (
        f"svm_{split_name}.joblib"
    )

    model_path = os.path.join(
        MODEL_DIR,
        model_filename
    )

    joblib.dump(model, model_path)

    print("Model saved:", model_path)

    return evaluate_model(
        model,
        X_test,
        y_test,
        "SVM",
        split_name
    )


# ==========================================
# 8. RUN EXPERIMENT
# ==========================================

def run_experiment(
    X_train,
    X_test,
    y_train,
    y_test,
    split_name
):

    results = []

    # Random Forest
    rf_results = train_random_forest(
        X_train,
        X_test,
        y_train,
        y_test,
        split_name
    )

    results.append(rf_results)

    # SVM
    svm_results = train_svm(
        X_train,
        X_test,
        y_train,
        y_test,
        split_name
    )

    results.append(svm_results)

    return results


# ==========================================
# 9. MAIN FUNCTION
# ==========================================

def main():

    print("\n======================================")
    print(" KEYGUARD - MODEL TRAINING")
    print("======================================")

    # Load dataset
    X, y, groups, sessions = load_data()

    all_results = []

    # --------------------------------------
    # Experiment 1: Normal random split
    # --------------------------------------

    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )

    random_results = run_experiment(
        X_train,
        X_test,
        y_train,
        y_test,
        "random"
    )

    all_results.extend(random_results)

    # --------------------------------------
    # Experiment 2: Session-aware split
    # --------------------------------------

    X_train, X_test, y_train, y_test = session_aware_split(
        X,
        y,
        groups,
        sessions
    )

    session_results = run_experiment(
        X_train,
        X_test,
        y_train,
        y_test,
        "session_aware"
    )

    all_results.extend(session_results)

    # --------------------------------------
    # Save comparison results
    # --------------------------------------

    results_df = pd.DataFrame(all_results)

    results_path = os.path.join(
        REPORT_DIR,
        "model_comparison.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    print("\n========== FINAL MODEL COMPARISON ==========")

    print(
        results_df.to_string(index=False)
    )

    print("\nComparison saved to:", results_path)

    print("\n========== TRAINING COMPLETED ==========")


# ==========================================
# 10. PROGRAM ENTRY POINT
# ==========================================

if __name__ == "__main__":
    main()
