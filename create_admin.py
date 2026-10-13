"""
EcoSort AI - Admin Management Utility
CLI tool to safely create or promote an admin account in MySQL (ecosort_ai).
Usage:
  Interactive:
    python create_admin.py

  Direct creation:
    python create_admin.py --name "System Admin" --email "admin@ecosort.ai" --password "AdminPass123!"

  Promote existing user:
    python create_admin.py --promote "user@ecosort.ai"
"""

import sys
import argparse
import re
from pathlib import Path
from werkzeug.security import generate_password_hash

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from database.db_connection import get_db_connection
from config import Config

EMAIL_REGEX = r'^[\w\.-]+@[\w\.-]+\.\w+$'


def validate_email(email):
    return bool(re.match(EMAIL_REGEX, email))


def promote_user(email):
    """Promotes an existing registered user to admin role."""
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT id, name, email, role FROM users WHERE email = %s", (email,))
            user = cursor.fetchone()
            if not user:
                print(f"[ERROR] No user found with email '{email}'.")
                conn.close()
                return False

            if user['role'] == 'admin':
                print(f"[INFO] User '{user['name']}' ({email}) is already an administrator.")
                conn.close()
                return True

            cursor.execute("UPDATE users SET role = 'admin' WHERE id = %s", (user['id'],))
            print(f"[SUCCESS] User '{user['name']}' ({email}) has been promoted to 'admin' role.")
            conn.close()
            return True
    except Exception as e:
        print(f"[DATABASE ERROR] Could not promote user: {e}")
        return False


def create_or_update_admin(name, email, password):
    """Creates a new admin or updates existing user password and role to admin."""
    if not validate_email(email):
        print(f"[ERROR] Invalid email format: {email}")
        return False

    if len(password) < 6:
        print("[ERROR] Password must be at least 6 characters long.")
        return False

    password_hash = generate_password_hash(password)

    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT id, name, role FROM users WHERE email = %s", (email,))
            existing = cursor.fetchone()

            if existing:
                cursor.execute(
                    "UPDATE users SET name = %s, password_hash = %s, role = 'admin' WHERE id = %s",
                    (name, password_hash, existing['id'])
                )
                print(f"[SUCCESS] Updated existing account '{email}' to administrator with new credentials.")
            else:
                cursor.execute(
                    "INSERT INTO users (name, email, password_hash, role) VALUES (%s, %s, %s, 'admin')",
                    (name, email, password_hash)
                )
                print(f"[SUCCESS] Created new administrator account '{name}' ({email}).")

        conn.close()
        return True
    except Exception as e:
        print(f"[DATABASE ERROR] Failed to create admin: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="EcoSort AI - Admin Management Utility")
    parser.add_argument("--name", help="Admin Full Name")
    parser.add_argument("--email", help="Admin Email Address")
    parser.add_argument("--password", help="Admin Password")
    parser.add_argument("--promote", help="Email of existing user to promote to admin")

    args = parser.parse_args()

    if args.promote:
        promote_user(args.promote.strip())
        return

    if args.email and args.password:
        name = args.name.strip() if args.name else "Administrator"
        create_or_update_admin(name, args.email.strip(), args.password)
        return

    # Interactive mode if no CLI arguments supplied
    print("=" * 60)
    print("  EcoSort AI - Local Administrator Setup Utility")
    print(f"  Target Database: {Config.DB_NAME}")
    print("=" * 60)
    mode = input("Choose action: [1] Create new admin, [2] Promote existing user (1/2): ").strip()

    if mode == '2':
        email = input("Enter email of user to promote: ").strip()
        promote_user(email)
    else:
        name = input("Enter admin name: ").strip() or "Administrator"
        email = input("Enter admin email: ").strip()
        password = input("Enter admin password (min 6 chars): ").strip()
        create_or_update_admin(name, email, password)


if __name__ == '__main__':
    main()
