"""
EcoSort AI - Phase 2 Authentication & Route Protection Tests
Tests for registration, password hashing, login, logout, session management,
role-based access control (@login_required, @admin_required), and health check.
"""

import sys
import unittest
from pathlib import Path
from werkzeug.security import check_password_hash, generate_password_hash

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app import app
from database.db_connection import get_db_connection


class AuthenticationTestCase(unittest.TestCase):
    """Test suite for user and admin authentication flows."""

    def setUp(self):
        """Set up Flask test client and ensure test database records are clean."""
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        self.client = app.test_client()

        # Clean up any residual test accounts before each test
        self._cleanup_test_users()

    def tearDown(self):
        """Clean up test accounts created during tests."""
        self._cleanup_test_users()

    def _cleanup_test_users(self):
        """Helper to remove all @test.com accounts from MySQL users table."""
        try:
            conn = get_db_connection()
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM users WHERE email LIKE '%@test.com'")
            conn.close()
        except Exception as e:
            print(f"Warning: Cleanup failed ({e})")

    def test_01_registration_success(self):
        """Test successful registration with valid inputs."""
        response = self.client.post('/register', data={
            'name': 'Test Student',
            'email': 'student@test.com',
            'password': 'SecurePassword123',
            'confirm_password': 'SecurePassword123'
        }, follow_redirects=False)

        # Should redirect to login page
        self.assertEqual(response.status_code, 302)
        self.assertTrue('/login' in response.headers['Location'])

        # Check user in database
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT name, email, role FROM users WHERE email = 'student@test.com'")
            user = cursor.fetchone()
        conn.close()

        self.assertIsNotNone(user)
        self.assertEqual(user['name'], 'Test Student')
        self.assertEqual(user['email'], 'student@test.com')
        self.assertEqual(user['role'], 'user')  # Must be 'user'

    def test_02_registration_duplicate_email(self):
        """Test that registering with an existing email is rejected."""
        # First registration
        self.client.post('/register', data={
            'name': 'Original User',
            'email': 'duplicate@test.com',
            'password': 'Password123',
            'confirm_password': 'Password123'
        })

        # Second registration with same email
        response = self.client.post('/register', data={
            'name': 'Imposter User',
            'email': 'duplicate@test.com',
            'password': 'AnotherPassword123',
            'confirm_password': 'AnotherPassword123'
        }, follow_redirects=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"already exists", response.data)

    def test_03_password_is_properly_hashed(self):
        """Test that passwords are never stored in plain text and use werkzeug hashing."""
        raw_password = 'MySecretPlaintextPassword!99'
        self.client.post('/register', data={
            'name': 'Security Check',
            'email': 'security@test.com',
            'password': raw_password,
            'confirm_password': raw_password
        })

        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT password_hash FROM users WHERE email = 'security@test.com'")
            user = cursor.fetchone()
        conn.close()

        self.assertIsNotNone(user)
        stored_hash = user['password_hash']
        # Hash must not be plain text
        self.assertNotEqual(stored_hash, raw_password)
        # Hash must verify correctly with werkzeug
        self.assertTrue(check_password_hash(stored_hash, raw_password))
        # Hash must fail for incorrect password
        self.assertFalse(check_password_hash(stored_hash, 'WrongPassword'))

    def test_04_login_success(self):
        """Test successful login and session population."""
        # Create user
        self.client.post('/register', data={
            'name': 'Login User',
            'email': 'login@test.com',
            'password': 'LoginPass123',
            'confirm_password': 'LoginPass123'
        })

        # Perform login
        with self.client as c:
            response = c.post('/login', data={
                'email': 'login@test.com',
                'password': 'LoginPass123'
            }, follow_redirects=False)

            self.assertEqual(response.status_code, 302)
            self.assertTrue('/dashboard' in response.headers['Location'])

            # Verify session variables
            with c.session_transaction() as sess:
                self.assertIn('user_id', sess)
                self.assertEqual(sess['name'], 'Login User')
                self.assertEqual(sess['email'], 'login@test.com')
                self.assertEqual(sess['role'], 'user')

    def test_05_login_failure_wrong_password(self):
        """Test login fails with incorrect password."""
        self.client.post('/register', data={
            'name': 'Fail User',
            'email': 'fail@test.com',
            'password': 'CorrectPassword1',
            'confirm_password': 'CorrectPassword1'
        })

        with self.client as c:
            response = c.post('/login', data={
                'email': 'fail@test.com',
                'password': 'WrongPassword99'
            }, follow_redirects=True)

            self.assertEqual(response.status_code, 200)
            self.assertIn(b"Invalid email or password", response.data)

            with c.session_transaction() as sess:
                self.assertNotIn('user_id', sess)

    def test_06_login_failure_unregistered_email(self):
        """Test login fails with non-existent email."""
        with self.client as c:
            response = c.post('/login', data={
                'email': 'nobody@test.com',
                'password': 'SomePassword123'
            }, follow_redirects=True)

            self.assertEqual(response.status_code, 200)
            self.assertIn(b"Invalid email or password", response.data)

            with c.session_transaction() as sess:
                self.assertNotIn('user_id', sess)

    def test_07_logout_clears_session(self):
        """Test that logging out removes all session data."""
        self.client.post('/register', data={
            'name': 'Logout User',
            'email': 'logout@test.com',
            'password': 'Password123',
            'confirm_password': 'Password123'
        })

        with self.client as c:
            c.post('/login', data={'email': 'logout@test.com', 'password': 'Password123'})
            logout_resp = c.get('/logout', follow_redirects=False)

            self.assertEqual(logout_resp.status_code, 302)
            self.assertTrue('/login' in logout_resp.headers['Location'])

            # Check session is cleared
            with c.session_transaction() as sess:
                self.assertNotIn('user_id', sess)

    def test_08_protected_dashboard_requires_login(self):
        """Test that unauthenticated users cannot access /dashboard."""
        response = self.client.get('/dashboard', follow_redirects=False)
        self.assertEqual(response.status_code, 302)
        self.assertTrue('/login' in response.headers['Location'])

    def test_09_protected_history_requires_login(self):
        """Test that unauthenticated users cannot access /history."""
        response = self.client.get('/history', follow_redirects=False)
        self.assertEqual(response.status_code, 302)
        self.assertTrue('/login' in response.headers['Location'])

    def test_10_admin_route_requires_login(self):
        """Test that unauthenticated users are redirected to login when visiting /admin."""
        response = self.client.get('/admin', follow_redirects=False)
        self.assertEqual(response.status_code, 302)
        self.assertTrue('/login' in response.headers['Location'])

    def test_11_normal_user_denied_from_admin_route(self):
        """Test that normal users ('user' role) receive 403 Forbidden when visiting /admin."""
        self.client.post('/register', data={
            'name': 'Normal Student',
            'email': 'normal@test.com',
            'password': 'Password123',
            'confirm_password': 'Password123'
        })

        with self.client as c:
            c.post('/login', data={'email': 'normal@test.com', 'password': 'Password123'})
            response = c.get('/admin')

            self.assertEqual(response.status_code, 403)
            self.assertIn(b"403 - Access Forbidden", response.data)

    def test_12_admin_user_can_access_admin_dashboard(self):
        """Test that an administrator can access /admin."""
        # Create an admin user directly
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO users (name, email, password_hash, role) VALUES (%s, %s, %s, 'admin')",
                ('Admin Test', 'admintest@test.com', generate_password_hash('AdminPass123'))
            )
        conn.close()

        with self.client as c:
            c.post('/login', data={'email': 'admintest@test.com', 'password': 'AdminPass123'})
            response = c.get('/admin')

            self.assertEqual(response.status_code, 200)
            self.assertIn(b"Administrator Control Panel", response.data)

    def test_13_api_health_check(self):
        """Test that /api/health works properly."""
        response = self.client.get('/api/health')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        self.assertEqual(data['status'], 'healthy')
        self.assertIn(data['phase'], [2, 3, 7])
        self.assertEqual(data['database']['target_db'], 'ecosort_ai')
        self.assertEqual(data['database']['connected'], True)
        self.assertTrue(data['ai_engine']['is_trained'])

    def test_14_login_with_next_redirect(self):
        """Test that logging in with a safe next URL redirects directly to that URL."""
        self.client.post('/register', data={
            'name': 'Next User',
            'email': 'next@test.com',
            'password': 'Password123',
            'confirm_password': 'Password123'
        })

        with self.client as c:
            response = c.post('/login', data={
                'email': 'next@test.com',
                'password': 'Password123',
                'next': '/analyze'
            }, follow_redirects=False)

            self.assertEqual(response.status_code, 302)
            self.assertEqual(response.headers['Location'], '/analyze')


if __name__ == '__main__':
    unittest.main()
