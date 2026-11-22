
import pandas as pd

from authentication_model import (
    FEATURE_COLUMNS,
    verify_user,
    AUTHENTICATION_THRESHOLD,
    UNCERTAINTY_MARGIN
)


# -----------------------------------------
# CONFIGURATION
# -----------------------------------------

TARGET_USER_ID = 7

DATASET_PATH = "dataset/authentication_dataset.csv"


# -----------------------------------------
# LOAD AUTHENTICATION DATA
# -----------------------------------------

try:
    df = pd.read_csv(DATASET_PATH)

    print("\nKeyGuard Genuine vs Impostor Evaluation")
    print("========================================")

    authentication_samples = df[
        df["sample_type"] == "authentication"
    ]

    if authentication_samples.empty:
        raise ValueError("No authentication samples found.")

    # -----------------------------------------
    # GENUINE AND IMPOSTOR COMPARISON
    # -----------------------------------------

    genuine_count = 0
    impostor_count = 0
    correct_count = 0

    for _, sample in authentication_samples.iterrows():

        result = verify_user(
            TARGET_USER_ID,
            sample[FEATURE_COLUMNS].to_dict()
        )

        user_id = str(sample["user_id"])

        actual_identity = (
            "GENUINE"
            if user_id == str(TARGET_USER_ID)
            else "IMPOSTOR"
        )

        predicted_identity = result["decision"].upper()

        if actual_identity == "GENUINE":
            genuine_count += 1
        else:
            impostor_count += 1

        if actual_identity == predicted_identity:
            correct_count += 1

        print(f"\nAccount: User {user_id}")
        print("Actual identity:", actual_identity)
        print(f"Distance score: {result['distance']:.4f}")
        print(f"Threshold: {result['threshold']:.4f}")
        print("Model decision:", predicted_identity)

        if actual_identity != predicted_identity:
            print("Classification: MISMATCH")

    # -----------------------------------------
    # EVALUATION SUMMARY
    # -----------------------------------------

    total_samples = genuine_count + impostor_count

    print("\nEvaluation Summary")
    print("------------------")
    print("Genuine samples:", genuine_count)
    print("Impostor samples:", impostor_count)
    print("Correct classifications:", correct_count)

    if total_samples:
        accuracy = (correct_count / total_samples) * 100
        print(f"Observed classification accuracy: {accuracy:.2f}%")

    # -----------------------------------------
    # THRESHOLD BOUNDARY TEST
    # -----------------------------------------

    print("\nDecision Threshold Test")
    print("------------------------")

    test_distances = [
        AUTHENTICATION_THRESHOLD - UNCERTAINTY_MARGIN - 0.5,
        AUTHENTICATION_THRESHOLD,
        AUTHENTICATION_THRESHOLD + UNCERTAINTY_MARGIN + 0.5
    ]

    for distance in test_distances:

        if distance < AUTHENTICATION_THRESHOLD - UNCERTAINTY_MARGIN:
            decision = "GENUINE"

        elif distance <= AUTHENTICATION_THRESHOLD + UNCERTAINTY_MARGIN:
            decision = "UNCERTAIN"

        else:
            decision = "IMPOSTOR"

        print(f"Distance: {distance:.4f} -> {decision}")

    print("\nAuthentication testing completed.")

except Exception as error:
    print("\nAuthentication test failed:")
    print(error)
