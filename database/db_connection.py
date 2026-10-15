"""
EcoSort AI - Database Connection Helper
Handles MySQL connections using PyMySQL with the ecosort_ai database.
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path when script is executed directly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pymysql
from pymysql.cursors import DictCursor
from config import Config


def get_db_connection(use_database=True):
    """
    Establishes and returns a new connection to the MySQL database.
    
    :param use_database: If True, connects directly to Config.DB_NAME ('ecosort_ai').
                         If False, connects without selecting a database (useful for init_db).
    :return: pymysql.Connection object
    """
    params = {
        'host': Config.DB_HOST,
        'port': Config.DB_PORT,
        'user': Config.DB_USER,
        'password': Config.DB_PASSWORD,
        'charset': 'utf8mb4',
        'cursorclass': DictCursor,
        'autocommit': True,
    }
    # Enable certificate-verified SSL for remote MySQL (such as Aiven).
    # Keep local XAMPP connections unchanged.
    if Config.DB_HOST not in ('localhost', '127.0.0.1'):
        import ssl

        ca_path = os.path.join(Config.BASE_DIR, 'certs', 'ca.pem')
        ssl_context = ssl.create_default_context(cafile=ca_path)
        ssl_context.check_hostname = True
        ssl_context.verify_mode = ssl.CERT_REQUIRED

        # Preserve certificate and hostname verification while allowing
        # certificate chains that fail Python's extra strict X.509 checks.
        if hasattr(ssl, 'VERIFY_X509_STRICT'):
            ssl_context.verify_flags &= ~ssl.VERIFY_X509_STRICT

        params['ssl'] = ssl_context

    if use_database:
        params['database'] = Config.DB_NAME

    return pymysql.connect(**params)


def test_connection():
    """
    Tests connectivity to MySQL and verifies if the ecosort_ai database and tables exist.
    
    :return: dict with 'connected' (bool), 'database_exists' (bool),
             'tables' (list of str), and 'message' (str).
    """
    result = {
        'connected': False,
        'database_exists': False,
        'tables': [],
        'categories_count': 0,
        'message': ''
    }

    try:
        # First test raw connection to MySQL server
        conn = get_db_connection(use_database=False)
        result['connected'] = True

        with conn.cursor() as cursor:
            # Check if database 'ecosort_ai' exists
            cursor.execute("SHOW DATABASES LIKE %s", (Config.DB_NAME,))
            db_row = cursor.fetchone()
            if db_row:
                result['database_exists'] = True

        conn.close()

        # If database exists, check tables inside it
        if result['database_exists']:
            conn = get_db_connection(use_database=True)
            with conn.cursor() as cursor:
                cursor.execute("SHOW TABLES;")
                tables = [list(row.values())[0] for row in cursor.fetchall()]
                result['tables'] = tables

                # Check seed categories
                if 'waste_categories' in tables:
                    cursor.execute("SELECT COUNT(*) AS total FROM waste_categories;")
                    count_row = cursor.fetchone()
                    result['categories_count'] = count_row['total'] if count_row else 0

            conn.close()
            result['message'] = (
                f"Successfully connected to MySQL database '{Config.DB_NAME}'. "
                f"Tables: {', '.join(result['tables'])}. "
                f"Waste categories loaded: {result['categories_count']}."
            )
        else:
            result['message'] = (
                f"MySQL server is running, but database '{Config.DB_NAME}' does not exist yet. "
                f"Import database/schema.sql using phpMyAdmin or execute init_db()."
            )

    except pymysql.err.OperationalError as e:
        result['message'] = (
            f"Cannot connect to MySQL server at {Config.DB_HOST}:{Config.DB_PORT}. "
            f"Ensure XAMPP MySQL service is running. Error details: {e}"
        )
    except Exception as e:
        result['message'] = f"Database test error: {e}"

    return result


def init_db(schema_path=None):
    """
    Initializes the database by executing the database/schema.sql script.
    
    :param schema_path: Path to schema.sql (defaults to database/schema.sql)
    :return: tuple (bool success, str message)
    """
    if schema_path is None:
        schema_path = os.path.join(Config.BASE_DIR, 'database', 'schema.sql')

    if not os.path.exists(schema_path):
        return False, f"Schema file not found at: {schema_path}"

    try:
        # Connect to server without database selected to run CREATE DATABASE
        conn = get_db_connection(use_database=False)
        with open(schema_path, 'r', encoding='utf-8') as f:
            sql_script = f.read()

        # Split commands by semicolon (ignoring comments and empty lines)
        statements = []
        delimiter = ';'
        current_statement = []

        for line in sql_script.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith('--') or stripped.startswith('/*'):
                continue
            current_statement.append(line)
            if stripped.endswith(';'):
                statements.append("\n".join(current_statement))
                current_statement = []

        with conn.cursor() as cursor:
            for statement in statements:
                stmt = statement.strip()
                if stmt:
                    cursor.execute(stmt)

        conn.close()
        return True, f"Database '{Config.DB_NAME}' initialized successfully from {schema_path}."

    except Exception as e:
        return False, f"Failed to initialize database: {e}"


if __name__ == '__main__':
    print("Testing connection to MySQL...")
    res = test_connection()
    print(f"Connected to MySQL: {res['connected']}")
    print(f"Database exists:   {res['database_exists']}")
    print(f"Tables:            {res['tables']}")
    print(f"Message:           {res['message']}")
