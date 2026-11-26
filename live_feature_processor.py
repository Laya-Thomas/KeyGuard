
"""
KeyGuard - CMU Live Feature Processor

Converts browser keyboard events into the 31 timing
features used by the CMU Keystroke Dynamics dataset.

Input timestamps: milliseconds
Output timing features: seconds
"""

import csv
import math
import uuid

from datetime import datetime
from pathlib import Path


# =========================================================
# 1. PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

LIVE_DATASET_PATH = (
    BASE_DIR / "dataset" / "authentication_dataset.csv"
)


# =========================================================
# 2. CMU FEATURE ORDER
# =========================================================

CMU_FEATURES = [
    "H.period",
    "DD.period.t",
    "UD.period.t",
    "H.t",
    "DD.t.i",
    "UD.t.i",
    "H.i",
    "DD.i.e",
    "UD.i.e",
    "H.e",
    "DD.e.five",
    "UD.e.five",
    "H.five",
    "DD.five.Shift.r",
    "UD.five.Shift.r",
    "H.Shift.r",
    "DD.Shift.r.o",
    "UD.Shift.r.o",
    "H.o",
    "DD.o.a",
    "UD.o.a",
    "H.a",
    "DD.a.n",
    "UD.a.n",
    "H.n",
    "DD.n.l",
    "UD.n.l",
    "H.l",
    "DD.l.Return",
    "UD.l.Return",
    "H.Return"
]


# =========================================================
# 3. EXPECTED FIXED-PHRASE KEY SEQUENCE
# =========================================================

# The phrase is: .tie5Roanl
#
# Shift is captured as a separate physical key event.
# Either left or right Shift is accepted.

EXPECTED_KEYS_LEFT_SHIFT = [
    "Period",
    "KeyT",
    "KeyI",
    "KeyE",
    "Digit5",
    "ShiftLeft",
    "KeyR",
    "KeyO",
    "KeyA",
    "KeyN",
    "KeyL",
    "Enter"
]

EXPECTED_KEYS_RIGHT_SHIFT = [
    "Period",
    "KeyT",
    "KeyI",
    "KeyE",
    "Digit5",
    "ShiftRight",
    "KeyR",
    "KeyO",
    "KeyA",
    "KeyN",
    "KeyL",
    "Enter"
]


# =========================================================
# 4. VALIDATE AND CONVERT BROWSER EVENTS
# =========================================================

def validate_keyboard_events(events):
    """
    Validate browser keyboard events and return
    completed events.
    """

    if not isinstance(events, list):
        raise ValueError(
            "Keyboard events must be a list."
        )

    completed = []

    for event in events:

        if not isinstance(event, dict):
            raise ValueError(
                "Invalid keyboard event."
            )

        if event.get("up") is None:
            continue

        if event.get("code") is None:
            raise ValueError(
                "A keyboard event is missing its key code."
            )

        try:
            down = float(event["down"])
            up = float(event["up"])
            hold = float(event["hold"])

        except (KeyError, TypeError, ValueError):
            raise ValueError(
                "Invalid keyboard timing values."
            )

        if not all(
            math.isfinite(value)
            for value in (down, up, hold)
        ):
            raise ValueError(
                "Keyboard timings must be finite numbers."
            )

        if up < down or hold < 0:
            raise ValueError(
                "Invalid key timing values."
            )

        # Check that recorded hold time agrees
        # with the down and up timestamps.

        calculated_hold = up - down

        if abs(calculated_hold - hold) > 2.0:
            raise ValueError(
                "Inconsistent key hold time."
            )

        completed.append({
            "code": str(event["code"]),
            "down": down,
            "up": up,
            "hold": hold
        })

    # Sort events by keydown timestamp.
    completed.sort(
        key=lambda item: item["down"]
    )

    if len(completed) != 12:
        raise ValueError(
            "The phrase must contain exactly "
            "12 completed key events."
        )

    return completed


# =========================================================
# 5. BUILD THE 31 CMU FEATURES
# =========================================================

def build_cmu_features(events):
    """
    Convert browser events into the 31 CMU timing features.

    The physical Shift key is validated as a modifier.
    KeyR is used for the uppercase R feature position.
    """

    completed = validate_keyboard_events(events)

    # -----------------------------------------------------
    # IDENTIFY LEFT OR RIGHT SHIFT
    # -----------------------------------------------------

    actual_keys = [
        event["code"]
        for event in completed
    ]

    if actual_keys == EXPECTED_KEYS_LEFT_SHIFT:
        pass

    elif actual_keys == EXPECTED_KEYS_RIGHT_SHIFT:
        pass

    else:
        raise ValueError(
            "Incorrect typing sequence. "
            "Please type the phrase exactly as displayed. "
            f"Received keys: {actual_keys}"
        )

    # -----------------------------------------------------
    # SELECT THE 11 EVENTS REPRESENTED IN CMU FEATURES
    # -----------------------------------------------------

    # KeyR represents the uppercase R position.
    # The separate physical Shift event is not selected.

    feature_codes = [
        "Period",
        "KeyT",
        "KeyI",
        "KeyE",
        "Digit5",
        "KeyR",
        "KeyO",
        "KeyA",
        "KeyN",
        "KeyL",
        "Enter"
    ]

    event_lookup = {
        event["code"]: event
        for event in completed
    }

    feature_events = [
        event_lookup[code]
        for code in feature_codes
    ]

    features = {}

    # -----------------------------------------------------
    # HOLD-TIME FEATURES
    # -----------------------------------------------------

    hold_names = [
        "H.period",
        "H.t",
        "H.i",
        "H.e",
        "H.five",
        "H.Shift.r",
        "H.o",
        "H.a",
        "H.n",
        "H.l",
        "H.Return"
    ]

    for name, event in zip(
        hold_names,
        feature_events
    ):

        features[name] = (
            event["hold"] / 1000.0
        )

    # -----------------------------------------------------
    # DOWN-DOWN AND UP-DOWN TRANSITION FEATURES
    # -----------------------------------------------------

    transitions = [
        ("period", "t"),
        ("t", "i"),
        ("i", "e"),
        ("e", "five"),
        ("five", "Shift.r"),
        ("Shift.r", "o"),
        ("o", "a"),
        ("a", "n"),
        ("n", "l"),
        ("l", "Return")
    ]

    for index, (first, second) in enumerate(
        transitions
    ):

        first_event = feature_events[index]
        second_event = feature_events[index + 1]

        # Down-Down time:
        # Difference between two keydown timestamps.

        dd = (
            second_event["down"]
            - first_event["down"]
        )

        # Up-Down time:
        # Difference between the first keyup
        # and the next keydown.

        ud = (
            second_event["down"]
            - first_event["up"]
        )

        features[
            f"DD.{first}.{second}"
        ] = dd / 1000.0

        features[
            f"UD.{first}.{second}"
        ] = ud / 1000.0

    # -----------------------------------------------------
    # RETURN FEATURES IN EXACT CMU ORDER
    # -----------------------------------------------------

    ordered_features = {
        name: features[name]
        for name in CMU_FEATURES
    }

    if len(ordered_features) != 31:
        raise ValueError(
            "Expected exactly 31 CMU features."
        )

    if not all(
        math.isfinite(value)
        for value in ordered_features.values()
    ):
        raise ValueError(
            "Generated features contain invalid values."
        )

    return ordered_features


# =========================================================
# 6. GET MODEL-READY FEATURE VECTOR
# =========================================================

def get_feature_vector(events):
    """
    Return the 31 CMU features as an ordered list.
    """

    features = build_cmu_features(events)

    return [
        features[name]
        for name in CMU_FEATURES
    ]


# =========================================================
# 7. SAVE LIVE KEYSTROKE FEATURES
# =========================================================

def save_live_features(
    user_id,
    events,
    sample_type="test"
):
    """
    Save a live keystroke sample to authentication_dataset.csv.

    Allowed sample types:
    - enrollment
    - authentication
    - test
    """

    if user_id is None or str(user_id).strip() == "":
        raise ValueError(
            "A valid user ID is required."
        )

    allowed_types = {
        "enrollment",
        "authentication",
        "test"
    }

    if sample_type not in allowed_types:
        raise ValueError(
            "Invalid sample type."
        )

    # Extract the 31 CMU features.
    features = build_cmu_features(events)

    # Ensure dataset directory exists.
    LIVE_DATASET_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    row = {
        "user_id": str(user_id),
        "sample_id": str(uuid.uuid4()),
        "sample_type": sample_type,
        "created_at": datetime.now().isoformat(
            timespec="seconds"
        )
    }

    row.update(features)

    fieldnames = [
        "user_id",
        "sample_id",
        "sample_type",
        "created_at"
    ] + CMU_FEATURES

    file_exists = (
        LIVE_DATASET_PATH.exists()
        and LIVE_DATASET_PATH.stat().st_size > 0
    )

    with open(
        LIVE_DATASET_PATH,
        mode="a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

    print(
        "Live keystroke sample saved successfully."
    )

    print("User ID:", user_id)
    print("Sample type:", sample_type)
    print("Feature count:", len(features))
    print("Saved location:", LIVE_DATASET_PATH)

    return row
