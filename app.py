
import pandas as pd
import os
import joblib

from functools import wraps
from authentication_model import verify_user
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash
)

from database import (
    init_db,
    create_user,
    get_user_by_username,
    verify_password,
    get_user_by_id,
    get_all_users,
    record_enrollment_sample,
    get_enrollment_status
)
from live_feature_processor import save_live_features

# ==========================================
# 1. INITIALIZE FLASK APPLICATION
# ==========================================

app = Flask(__name__)

app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY",
    "keyguard-development-secret-change-this"
)


# ==========================================
# 2. PROJECT PATHS
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "authentication_model.joblib"
)
DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "cleaned_dataset.csv"
)

# ==========================================
# 3. LOAD MACHINE LEARNING MODEL
# ==========================================

model = None

try:

    if os.path.exists(MODEL_PATH):

        model = joblib.load(MODEL_PATH)

        print("Random Forest model loaded successfully.")

    else:

        print("WARNING: Trained model file was not found.")

except Exception as error:

    print("Error loading model:", error)


# ==========================================
# 4. ROLE-BASED ACCESS DECORATORS
# ==========================================

def login_required(function):

    @wraps(function)
    def wrapper(*args, **kwargs):

        if "user_id" not in session:

            flash("Please log in first.", "error")

            return redirect(url_for("login"))

        return function(*args, **kwargs)

    return wrapper


def admin_required(function):

    @wraps(function)
    def wrapper(*args, **kwargs):

        if "user_id" not in session:

            flash("Please log in first.", "error")

            return redirect(url_for("login"))

        if session.get("role") != "admin":

            flash(
                "Access denied. Admin permission required.",
                "error"
            )

            return redirect(url_for("user_dashboard"))

        return function(*args, **kwargs)

    return wrapper


def user_required(function):
    @wraps(function)
    def wrapper(*args, **kwargs):

        if "user_id" not in session:
            flash("Please log in first.", "error")
            return redirect(url_for("login"))

        if session.get("role") != "user":
            flash("This page is for normal users only.", "error")
            return redirect(url_for("admin_dashboard"))

        if not session.get("behavior_verified"):
            flash(
                "Please complete behavioral authentication first.",
                "warning"
            )
            return redirect(url_for("authentication"))

        return function(*args, **kwargs)

    return wrapper
# ==========================================
# 5. HOME ROUTE
# ==========================================

@app.route("/")
def home():

    if "user_id" in session:

        return redirect(url_for("dashboard"))

    return redirect(url_for("login"))


# ==========================================
# 6. REGISTRATION ROUTE
# ==========================================

@app.route("/register", methods=["GET", "POST"])
def register():

    if "user_id" in session:

        return redirect(url_for("dashboard"))

    if request.method == "POST":

        username = request.form.get(
            "username", ""
        ).strip()

        password = request.form.get(
            "password", ""
        )

        confirm_password = request.form.get(
            "confirm_password", ""
        )

        if not username or not password or not confirm_password:

            flash("Please fill in all fields.", "error")

            return render_template("register.html")

        if len(username) < 3:

            flash(
                "Username must contain at least 3 characters.",
                "error"
            )

            return render_template("register.html")

        if len(password) < 8:

            flash(
                "Password must contain at least 8 characters.",
                "error"
            )

            return render_template("register.html")

        if password != confirm_password:

            flash(
                "Passwords do not match.",
                "error"
            )

            return render_template("register.html")

        success, message = create_user(
            username,
            password
        )

        if success:

            flash(
                "Account created successfully. Please log in.",
                "success"
            )

            return redirect(url_for("login"))

        flash(message, "error")

    return render_template("register.html")


# ==========================================
# 7. LOGIN ROUTE
# ==========================================


@app.route("/login", methods=["GET", "POST"])
def login():

    if "user_id" in session:
        if session.get("role") == "admin":
            return redirect(url_for("admin_dashboard"))

        if session.get("behavior_verified"):
            return redirect(url_for("user_dashboard"))

        if session.get("role") == "user":
            status = get_enrollment_status(session["user_id"])

            if status and status["enrollment_completed"]:
                return redirect(url_for("authentication"))

            return redirect(url_for("enrollment"))

    if request.method == "POST":

        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        user = get_user_by_username(username)

        if user and verify_password(user, password):

            session.clear()

            session["user_id"] = user["id"]
            session["username"] = user["username"]
            session["role"] = user["role"]

            flash("Password verified successfully.", "success")

            if user["role"] == "admin":
                return redirect(url_for("admin_dashboard"))

            enrollment_status = get_enrollment_status(user["id"])

            if (
                enrollment_status is None
                or not enrollment_status["enrollment_completed"]
            ):
                return redirect(url_for("enrollment"))

            return redirect(url_for("authentication"))

        flash("Invalid username or password.", "error")

    return render_template("login.html")


# ==========================================
# 8. COMMON DASHBOARD REDIRECTION
# ==========================================

@app.route("/dashboard")
@login_required
def dashboard():

    if session.get("role") == "admin":

        return redirect(url_for("admin_dashboard"))

    return redirect(url_for("user_dashboard"))


# ==========================================
# 9. ADMIN DASHBOARD
# ==========================================

@app.route("/admin/dashboard")
@admin_required
def admin_dashboard():

    users = get_all_users()

    model_status = (
        "Loaded successfully"
        if model is not None
        else "Model not found"
    )

    return render_template(
        "admin_dashboard.html",
        username=session["username"],
        model_status=model_status,
        total_users=len(users)
    )





# ==========================================
# 10. NORMAL USER DASHBOARD
# ==========================================


@app.route("/user/dashboard")
@user_required
def user_dashboard():

    user_id = session["user_id"]

    enrollment_status = get_enrollment_status(user_id)

    # Redirect incomplete users to enrollment
    if (
        enrollment_status is None
        or not enrollment_status["enrollment_completed"]
    ):
        flash(
            "Please complete your behavioral enrollment first.",
            "warning"
        )
        return redirect(url_for("enrollment"))

    return render_template(
        "user_dashboard.html",
        username=session["username"],
        enrollment_count=enrollment_status["enrollment_count"],
        enrollment_completed=enrollment_status["enrollment_completed"]
    )


# ==========================================
# 11. ADMIN USER MANAGEMENT
# ==========================================

@app.route("/admin/users")
@admin_required
def admin_users():

    users = get_all_users()

    return render_template(
        "users.html",
        users=users,
        total_users=len(users)
    )

# ==========================================
# ADMIN DATASET MANAGEMENT
# ==========================================

@app.route("/admin/dataset")
@admin_required
def admin_dataset():

    # Check whether the dataset exists
    if not os.path.exists(DATASET_PATH):

        flash(
            "Cleaned dataset file was not found.",
            "error"
        )

        return redirect(url_for("admin_dashboard"))

    try:

        # Load the actual CSV dataset
        df = pd.read_csv(DATASET_PATH)

        # Calculate dataset statistics
        total_samples = len(df)

        total_columns = len(df.columns)

        total_subjects = (
            df["subject"].nunique()
            if "subject" in df.columns
            else 0
        )

        total_sessions = (
            df["sessionIndex"].nunique()
            if "sessionIndex" in df.columns
            else 0
        )

        # Exclude metadata columns when counting features
        metadata_columns = [
            "subject",
            "sessionIndex",
            "rep"
        ]

        feature_columns = [
            column
            for column in df.columns
            if column not in metadata_columns
        ]

        total_features = len(feature_columns)

        # Get first 10 rows for preview
        preview = (
            df.head(10)
            .fillna("")
            .astype(str)
        )

        rows = preview.values.tolist()

        columns = df.columns.tolist()

        # Get number of samples for each subject
        if "subject" in df.columns:

            distribution = (
                df["subject"]
                .value_counts()
                .sort_index()
            )

            subject_distribution = [
                {
                    "subject": str(subject),
                    "count": int(count)
                }
                for subject, count in distribution.items()
            ]

        else:

            subject_distribution = []

        # Display dataset page
        return render_template(
            "dataset.html",
            dataset_name="DSL-StrongPasswordData.csv",
            dataset_status="Loaded successfully",
            total_samples=total_samples,
            total_subjects=total_subjects,
            total_features=total_features,
            total_sessions=total_sessions,
            total_columns=total_columns,
            columns=columns,
            rows=rows,
            subject_distribution=subject_distribution
        )

    except Exception as error:

        print("Dataset loading error:", error)

        flash(
            "An error occurred while loading the dataset.",
            "error"
        )

        return redirect(url_for("admin_dashboard"))

    
@app.route("/admin/analysis")
@admin_required
def admin_analysis():
    return render_template("analysis.html")

# ==========================================
# 12. LOGOUT ROUTE
# ==========================================

@app.route("/logout")
@login_required
def logout():

    session.clear()

    flash("You have been logged out.", "success")

    return redirect(url_for("login"))


# ==========================================
# 13. HEALTH TEST ROUTE
# ==========================================

@app.route("/health")
def health():

    return "KEYGUARD FLASK SERVER IS WORKING!"




@app.route("/keystroke-test")
@login_required
def keystroke_test():
    return render_template("keystroke_test.html")

@app.route("/enrollment")
@login_required
def enrollment():

    user_id = session["user_id"]

    enrollment_status = get_enrollment_status(user_id)

    if (
        enrollment_status is not None
        and enrollment_status["enrollment_completed"]
    ):
        return redirect(url_for("user_dashboard"))

    return render_template("enrollment.html")

@app.route("/authentication")
@login_required
def authentication():

    if session.get("role") != "user":
        return redirect(url_for("admin_dashboard"))

    enrollment_status = get_enrollment_status(
        session["user_id"]
    )

    if (
        enrollment_status is None
        or not enrollment_status["enrollment_completed"]
    ):
        flash(
            "Please complete behavioral enrollment first.",
            "warning"
        )
        return redirect(url_for("enrollment"))

    return render_template("authentication.html")
# ==========================================
# SAVE LIVE KEYSTROKE SAMPLE
# ==========================================

@app.route("/api/save-keystroke-sample", methods=["POST"])
@login_required
def save_keystroke_sample():

    try:
        if session.get("role") != "user":
            return {
                "success": False,
                "message": "Only normal users can perform this action."
            }, 403

        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return {
                "success": False,
                "message": "Invalid request data."
            }, 400

        events = data.get("events")

        if not isinstance(events, list):
            return {
                "success": False,
                "message": "Invalid keyboard events."
            }, 400

        user_id = session.get("user_id")
        print("DEBUG - Logged-in user ID:", user_id)
        enrollment_status = get_enrollment_status(user_id)

        if enrollment_status is None:
            return {
                "success": False,
                "message": "User account not found."
            }, 404

        is_enrollment = data.get("enrollment") is True
        is_authentication = data.get("authentication") is True

        if is_enrollment and is_authentication:
            return {
                "success": False,
                "message": "Invalid sample request."
            }, 400

        # ---------------------------------------------
        # ENROLLMENT
        # ---------------------------------------------

        if is_enrollment:

            if enrollment_status["enrollment_completed"]:
                return {
                    "success": False,
                    "message": "Enrollment is already completed."
                }, 400

            saved_sample = save_live_features(
                user_id=user_id,
                events=events,
                sample_type="enrollment"
            )

            updated_status = record_enrollment_sample(user_id)

            return {
                "success": True,
                "message": "Enrollment sample saved successfully.",
                "sample_id": saved_sample["sample_id"],
                "enrollment_count": updated_status["enrollment_count"],
                "enrollment_completed": bool(
                    updated_status["enrollment_completed"]
                )
            }, 200

        # ---------------------------------------------
        # BEHAVIORAL AUTHENTICATION
        # ---------------------------------------------

        if is_authentication:

            if not enrollment_status["enrollment_completed"]:
                return {
                    "success": False,
                    "message": "Complete enrollment before authentication."
                }, 400

            # Run the trained Random Forest model.
            result = verify_user(user_id, events)
            print("DEBUG - Starting Random Forest verification for:", user_id)
            if result["authenticated"]:

                save_live_features(
                    user_id=user_id,
                    events=events,
                    sample_type="authentication"
                )

                session["behavior_verified"] = True

                return {
                    "success": True,
                    "authenticated": True,
                    "decision": result["decision"],
                    "message": "Behavioral authentication successful.",
                    "genuine_comparisons": result["genuine_comparisons"],
                    "impostor_comparisons": result["impostor_comparisons"],
                    "genuine_probability": result["genuine_probability"]
                }, 200

            session.pop("behavior_verified", None)

            return {
                "success": True,
                "authenticated": False,
                "decision": result["decision"],
                "message": "Behavioral verification failed. Please try again.",
                "genuine_comparisons": result["genuine_comparisons"],
                "impostor_comparisons": result["impostor_comparisons"],
                "genuine_probability": result["genuine_probability"]
            }, 200

        return {
            "success": False,
            "message": "Specify enrollment or authentication."
        }, 400

    except ValueError as error:

        return {
            "success": False,
            "message": str(error)
        }, 400

    except Exception as error:

        app.logger.exception(
            "Keystroke processing failed."
        )

        return {
            "success": False,
            "message": "An internal authentication error occurred."
        }, 500
# ==========================================
# 14. RUN FLASK APPLICATION
# ==========================================

if __name__ == "__main__":

    init_db()

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )
