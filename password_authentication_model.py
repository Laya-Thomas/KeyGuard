
import json
import numpy as np
import pandas as pd

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = (
    BASE_DIR / "dataset" / "password_authentication_dataset.csv"
)

UNCERTAINTY_MARGIN = 0.5


def load_password_samples(user_id):
    if not DATASET_PATH.exists():
        raise ValueError("No password timing dataset found.")

    df = pd.read_csv(DATASET_PATH)

    samples = df[
        (df["user_id"].astype(str) == str(user_id)) &
        (df["sample_type"] == "enrollment")
    ]

    if len(samples) < 3:
        raise ValueError("At least 3 enrollment samples are required.")

    return samples


def build_password_profile(user_id):
    samples = load_password_samples(user_id)

    timing_vectors = []

    for _, sample in samples.iterrows():
        timings = json.loads(sample["timings_json"])

        if not isinstance(timings, list) or not timings:
            raise ValueError("Invalid enrollment timing data.")

        timing_vectors.append(np.array(timings, dtype=float))

    # Every enrollment attempt must have the same number
    # of timing measurements.
    lengths = {len(vector) for vector in timing_vectors}

    if len(lengths) != 1:
        raise ValueError(
            "Enrollment samples have different lengths. "
            "Please enroll again using the same password."
        )

    matrix = np.vstack(timing_vectors)

    if not np.all(np.isfinite(matrix)):
        raise ValueError("Invalid timing values in enrollment data.")

    mean_profile = np.mean(matrix, axis=0)

    std_profile = np.std(matrix, axis=0, ddof=1)

    # Prevent division by zero for consistent timings.
    std_profile = np.maximum(std_profile, 0.02)

    return {
        "user_id": str(user_id),
        "mean_profile": mean_profile,
        "std_profile": std_profile,
        "sample_count": len(timing_vectors),
        "feature_count": len(mean_profile)
    }


def verify_password_typing(user_id, events):
    profile = build_password_profile(user_id)

    from live_feature_processor import build_password_timing_features

    extracted = build_password_timing_features(events)

    current = np.array(extracted["timings"], dtype=float)
    expected = profile["mean_profile"]
    variation = profile["std_profile"]

    if len(current) != len(expected):
        return {
            "decision": "impostor",
            "distance": None,
            "message": "Typing pattern length does not match enrollment."
        }

    deviations = (current - expected) / variation

    distance = float(np.sqrt(np.mean(deviations ** 2)))

    # Initial provisional threshold.
    threshold = 3.0

    if distance < threshold - UNCERTAINTY_MARGIN:
        decision = "genuine"
    elif distance <= threshold + UNCERTAINTY_MARGIN:
        decision = "uncertain"
    else:
        decision = "impostor"

    return {
        "user_id": str(user_id),
        "decision": decision,
        "distance": round(distance, 4),
        "threshold": threshold,
        "sample_count": profile["sample_count"],
        "message": decision.capitalize()
    }
