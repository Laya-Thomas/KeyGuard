
from live_feature_processor import save_live_features

# Synthetic test events (milliseconds)
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

# Test with a clearly labelled synthetic user ID
saved_row = save_live_features(
    user_id="TEST_USER",
    events=events
)

print("\nStorage test completed!")
print("Saved user:", saved_row["user_id"])
print("Number of timing features:", len(saved_row) - 3)
