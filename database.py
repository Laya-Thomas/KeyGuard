
import os
import sqlite3

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)


# ==========================================
# 1. DATABASE CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATABASE_PATH = os.path.join(
    BASE_DIR,
    "keyguard.db"
)


# ==========================================
# 2. CONNECT TO DATABASE
# ==========================================

def get_db_connection():

    connection = sqlite3.connect(DATABASE_PATH)

    connection.row_factory = sqlite3.Row

    return connection


# ==========================================
# 3. INITIALIZE DATABASE
# ==========================================

def init_db():

    connection = get_db_connection()

    try:

        connection.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL
                    CHECK(role IN ('admin', 'user')),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        columns = [
            column["name"]
            for column in connection.execute(
                "PRAGMA table_info(users)"
            ).fetchall()
        ]

        if "enrollment_count" not in columns:
            connection.execute("""
                ALTER TABLE users
                ADD COLUMN enrollment_count INTEGER DEFAULT 0
            """)

        if "enrollment_completed" not in columns:
            connection.execute("""
                ALTER TABLE users
                ADD COLUMN enrollment_completed INTEGER DEFAULT 0
            """)

        connection.commit()

        print("Users table created successfully.")

    finally:

        connection.close()

    create_default_admin()

# ==========================================
# 4. CREATE DEFAULT ADMIN ACCOUNT
# ==========================================

def create_default_admin():

    connection = get_db_connection()

    try:

        existing_admin = connection.execute(
            """
            SELECT id
            FROM users
            WHERE username = ?
            """,
            ("admin",)
        ).fetchone()

        if existing_admin is None:

            password_hash = generate_password_hash(
                "admin123"
            )

            connection.execute(
                """
                INSERT INTO users
                (username, password_hash, role)
                VALUES (?, ?, ?)
                """,
                ("admin", password_hash, "admin")
            )

            connection.commit()

            print("Default admin account created.")

        else:

            print("Admin account already exists.")

    finally:

        connection.close()


# ==========================================
# 5. REGISTER NEW USER
# ==========================================

def create_user(username, password):

    username = username.strip()

    if not username or not password:
        return False, "Username and password are required."

    if len(username) < 3:
        return False, "Username must contain at least 3 characters."

    if len(password) < 8:
        return False, "Password must contain at least 8 characters."

    password_hash = generate_password_hash(password)

    connection = get_db_connection()

    try:

        connection.execute(
            """
            INSERT INTO users
            (username, password_hash, role)
            VALUES (?, ?, ?)
            """,
            (username, password_hash, "user")
        )

        connection.commit()

        return True, "Account created successfully."

    except sqlite3.IntegrityError:

        return False, "Username already exists."

    finally:

        connection.close()


# ==========================================
# 6. GET USER BY USERNAME
# ==========================================

def get_user_by_username(username):

    connection = get_db_connection()

    try:

        user = connection.execute(
            """
            SELECT *
            FROM users
            WHERE username = ?
            """,
            (username,)
        ).fetchone()

        return dict(user) if user else None

    finally:

        connection.close()


# ==========================================
# 7. VERIFY USER PASSWORD
# ==========================================

def verify_password(user, password):

    if user is None:
        return False

    return check_password_hash(
        user["password_hash"],
        password
    )


# ==========================================
# 8. GET USER BY ID
# ==========================================

def get_user_by_id(user_id):

    connection = get_db_connection()

    try:

        user = connection.execute(
            """
            SELECT id, username, role, created_at
            FROM users
            WHERE id = ?
            """,
            (user_id,)
        ).fetchone()

        return dict(user) if user else None

    finally:

        connection.close()


# ==========================================
# 9. GET ALL USERS (ADMIN USE)
# ==========================================

def get_all_users():

    connection = get_db_connection()

    try:

        users = connection.execute(
            """
            SELECT id, username, role, created_at
            FROM users
            ORDER BY id DESC
            """
        ).fetchall()

        return [dict(user) for user in users]

    finally:

        connection.close()


# ==========================================
# 10. ENROLLMENT TRACKING
# ==========================================

def record_enrollment_sample(user_id):

    connection = get_db_connection()

    try:
        connection.execute("""
            UPDATE users
            SET enrollment_count = enrollment_count + 1,
                enrollment_completed =
                    CASE
                        WHEN enrollment_count + 1 >= 3 THEN 1
                        ELSE 0
                    END
            WHERE id = ?
              AND enrollment_count < 3
              AND role = 'user'
        """, (user_id,))

        connection.commit()

        user = connection.execute("""
            SELECT enrollment_count, enrollment_completed
            FROM users
            WHERE id = ?
        """, (user_id,)).fetchone()

        if user is None:
            raise ValueError("User not found.")

        return dict(user)

    finally:
        connection.close()


def get_enrollment_status(user_id):

    connection = get_db_connection()

    try:
        user = connection.execute("""
            SELECT enrollment_count, enrollment_completed
            FROM users
            WHERE id = ?
        """, (user_id,)).fetchone()

        if user is None:
            return None

        return dict(user)

    finally:
        connection.close()

# ==========================================
# 10. RUN DATABASE INITIALIZATION
# ==========================================

if __name__ == "__main__":

    init_db()

    print("\nKeyGuard database setup completed!")

    print("Database location:", DATABASE_PATH)
