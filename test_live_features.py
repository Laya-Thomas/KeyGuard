


from live_feature_processor import (
    build_cmu_features,
    get_feature_vector,
    CMU_FEATURES
)


# Create sample keyboard events for testing.
# These are synthetic timestamps, not real user measurements.

events = [
    {"code": "Period", "down": 1000, "up": 1090, "hold": 90},
    {"code": "KeyT", "down": 1200, "up": 1280, "hold": 80},
    {"code": "KeyI", "down": 1400, "up": 1485, "hold": 85},
    {"code": "KeyE", "down": 1600, "up": 1680, "hold": 80},
    {"code": "Digit5", "down": 1800, "up": 1890, "hold": 90},
    {"code": "ShiftLeft", "down": 1950, "up": 2050, "hold": 100},
    {"code": "KeyR", "down": 2000, "up": 2080, "hold": 80},
    {"code": "KeyO", "down": 2200, "up": 2290, "hold": 90},
    {"code": "KeyA", "down": 2400, "up": 2480, "hold": 80},
    {"code": "KeyN", "down": 2600, "up": 2690, "hold": 90},
    {"code": "KeyL", "down": 2800, "up": 2880, "hold": 80},
    {"code": "Enter", "down": 3000, "up": 3090, "hold": 90}
]


def main():

    print("\n========== LIVE FEATURE TEST ==========")

    # Calculate 31 CMU features
    features = build_cmu_features(events)

    # Generate model-ready feature vector
    feature_vector = get_feature_vector(events)

    print("\nNumber of generated features:", len(features))
    print("Feature vector length:", len(feature_vector))

    print("\nGenerated CMU features:")

    for name in CMU_FEATURES:
        print(f"{name}: {features[name]:.4f}")

    # Verify feature count and ordering
    assert len(features) == 31
    assert len(feature_vector) == 31
    assert list(features.keys()) == CMU_FEATURES

    print("\nAll 31 features generated successfully!")
    print("Feature order verified!")
    print("Live feature processor test passed!")


if __name__ == "__main__":
    main()

