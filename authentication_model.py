
"""
KeyGuard - Random Forest Behavioral Authentication

Uses the trained CMU pairwise Random Forest classifier
to compare login keystrokes against enrollment samples.
"""

import joblib
import pandas as pd
import numpy as np

from pathlib import Path

from live_feature_processor import (
    CMU_FEATURES,
    build_cmu_features
)


# =========================================================
# 1. PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "authentication_model.joblib"

DATASET_PATH = (
    BASE_DIR / "dataset" / "authentication_dataset.csv"
)

METADATA_PATH = (
    BASE_DIR / "models" / "authentication_model_metadata.joblib"
)


# =========================================================
# 2. LOAD TRAINED MODEL
# =========================================================

def load_authentication_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Authentication model not found. "
            "Train the model first."
        )

    model = joblib.load(MODEL_PATH)

    if METADATA_PATH.exists():
        metadata = joblib.load(METADATA_PATH)

        if metadata.get("feature_names") != CMU_FEATURES:
            raise ValueError(
                "Model feature order does not match live features."
            )

    return model


# =========================================================
# 3. LOAD USER ENROLLMENT SAMPLES
# =========================================================

def load_enrollment_samples(user_id):

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            "Authentication dataset does not exist."
        )

    df = pd.read_csv(DATASET_PATH)

    required_columns = (
        ["user_id", "sample_type"] + CMU_FEATURES
    )

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing enrollment columns: {missing}"
        )

    user_samples = df[
        (df["user_id"].astype(str) == str(user_id))
        & (df["sample_type"] == "enrollment")
    ].copy()

    for column in CMU_FEATURES:
        user_samples[column] = pd.to_numeric(
            user_samples[column],
            errors="coerce"
        )

    user_samples = user_samples.dropna(
        subset=CMU_FEATURES
    )

    if len(user_samples) < 3:
        raise ValueError(
            "Three valid enrollment samples are required."
        )

    return user_samples[CMU_FEATURES].to_numpy(
        dtype=float
    )


# =========================================================
# 4. BUILD PAIRWISE FEATURE DIFFERENCES
# =========================================================

def build_pair_features(enrollment_samples, login_features):

    enrollment_samples = np.asarray(
        enrollment_samples,
        dtype=float
    )

    login_features = np.asarray(
        login_features,
        dtype=float
    ).reshape(-1)

    if login_features.shape[0] != len(CMU_FEATURES):
        raise ValueError(
            "Login sample must contain exactly 31 features."
        )

    if enrollment_samples.ndim != 2:
        raise ValueError("Invalid enrollment sample format.")

    if enrollment_samples.shape[1] != len(CMU_FEATURES):
        raise ValueError(
            "Enrollment samples must contain 31 features."
        )

    if not np.isfinite(login_features).all():
        raise ValueError("Login features contain invalid values.")

    if not np.isfinite(enrollment_samples).all():
        raise ValueError("Enrollment features contain invalid values.")

    # Compare login sample with every enrollment sample.
    differences = np.abs(
        enrollment_samples - login_features
    )

    return differences


# =========================================================
# 5. VERIFY USER
# =========================================================

def verify_user(user_id, events):

    """
    Verify a login attempt using the trained Random Forest.

    Returns:
        genuine: predicted genuine
        impostor: predicted impostor

    This is a preliminary decision using the CMU-trained model.
    """

    model = load_authentication_model()

    # Load user's enrollment samples.
    enrollment_samples = load_enrollment_samples(user_id)

    # Extract the 31 live CMU features.
    login_features = build_cmu_features(events)

    login_vector = np.array(
        [login_features[name] for name in CMU_FEATURES],
        dtype=float
    )

    # Compare login sample with enrollment samples.
    pair_features = build_pair_features(
        enrollment_samples,
        login_vector
    )

    # Predict each comparison.
    predictions = model.predict(pair_features)

    genuine_count = int(
        np.sum(predictions == 1)
    )

    impostor_count = int(
        np.sum(predictions == 0)
    )

    total_comparisons = len(predictions)

    # Average genuine probability if supported by the model.
    genuine_probability = None

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(
            pair_features
        )

        genuine_column = list(model.classes_).index(1)

        genuine_probability = float(
            np.mean(probabilities[:, genuine_column])
        )

    # Majority decision.
    is_genuine = (
        genuine_count > impostor_count
    )

    return {
        "user_id": str(user_id),
        "decision": (
            "genuine" if is_genuine else "impostor"
        ),
        "authenticated": bool(is_genuine),
        "genuine_comparisons": genuine_count,
        "impostor_comparisons": impostor_count,
        "total_comparisons": total_comparisons,
        "genuine_probability": genuine_probability
    }


# =========================================================
# 6. SIMPLE TEST
# =========================================================

if __name__ == "__main__":

    print("KeyGuard authentication model module loaded.")
    print("Model path:", MODEL_PATH)

    model = load_authentication_model()

    print("Random Forest model loaded successfully.")
    print("Model classes:", model.classes_)
    print("Feature count:", len(CMU_FEATURES))
