"""
EcoSort AI - Phase 5 History, Feedback & Dashboard Unit and Integration Tests
Tests user history, privacy isolation (cannot view others' predictions),
prediction detail, feedback submission, duplicate feedback handling,
and dashboard statistics derived strictly from MySQL ecosort_ai.
"""

import io
import sys
import unittest
from pathlib import Path
from werkzeug.security import generate_password_hash

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app import app
from database.db_connection import get_db_connection


class HistoryFeedbackTestCase(unittest.TestCase):
    """Test suite for prediction history, ownership verification, and feedback."""

    def setUp(self):
        """Set up test users and test predictions in MySQL ecosort_ai."""
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        self.client = app.test_client()

        self._cleanup()

        # Create two test users and one admin
        conn = get_db_connection()
        with conn.cursor() as cursor:
            # User 1
            cursor.execute(
                "INSERT INTO users (name, email, password_hash, role) VALUES (%s, %s, %s, 'user')",
                ('Alice User', 'alice@phase5-test.com', generate_password_hash('Password123'))
            )
            self.user1_id = cursor.lastrowid

            # User 2
            cursor.execute(
                "INSERT INTO users (name, email, password_hash, role) VALUES (%s, %s, %s, 'user')",
                ('Bob User', 'bob@phase5-test.com', generate_password_hash('Password123'))
            )
            self.user2_id = cursor.lastrowid

            # Admin User
            cursor.execute(
                "INSERT INTO users (name, email, password_hash, role) VALUES (%s, %s, %s, 'admin')",
                ('Super Admin', 'admin@phase5-test.com', generate_password_hash('AdminPass123'))
            )
            self.admin_id = cursor.lastrowid

            # Seed a test prediction for User 1 (Wet Waste, id=1)
            cursor.execute(
                """
                INSERT INTO predictions (
                    user_id, image_filename, predicted_category_id,
                    confidence_score, model_version, user_feedback
                ) VALUES (%s, %s, 1, 91.50, 'MobileNetV2-v1', 'unreviewed')
                """,
                (self.user1_id, 'test_apple_peel.jpg')
            )
            self.user1_pred_id = cursor.lastrowid

            # Seed a test prediction for User 2 (E-Waste, id=4)
            cursor.execute(
                """
                INSERT INTO predictions (
                    user_id, image_filename, predicted_category_id,
                    confidence_score, model_version, user_feedback
                ) VALUES (%s, %s, 4, 88.00, 'MobileNetV2-v1', 'unreviewed')
                """,
                (self.user2_id, 'test_charger.jpg')
            )
            self.user2_pred_id = cursor.lastrowid

        conn.close()

    def tearDown(self):
        self._cleanup()

    def _cleanup(self):
        """Remove test users and their cascade-associated predictions/feedback."""
        try:
            conn = get_db_connection()
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM users WHERE email LIKE '%@phase5-test.com'")
                cursor.execute("DELETE FROM predictions WHERE image_filename LIKE 'test_%.jpg'")
            conn.close()
        except Exception:
            pass

    def test_01_user_can_view_own_prediction_history(self):
        """Verify User 1 can view their own prediction in /history."""
        with self.client as c:
            c.post('/login', data={'email': 'alice@phase5-test.com', 'password': 'Password123'})
            response = c.get('/history')
            self.assertEqual(response.status_code, 200)
            content = response.data.decode('utf-8')

            self.assertIn("Wet Waste", content)
            self.assertIn("91.5", content)
            self.assertIn(f"/prediction/{self.user1_pred_id}", content)

    def test_02_user_cannot_see_another_users_predictions_in_history(self):
        """Verify User 1 does NOT see User 2's prediction in /history."""
        with self.client as c:
            c.post('/login', data={'email': 'alice@phase5-test.com', 'password': 'Password123'})
            response = c.get('/history')
            self.assertEqual(response.status_code, 200)
            content = response.data.decode('utf-8')

            # User 1 should see Wet Waste (their own), but NOT E-Waste (User 2's)
            self.assertIn("Wet Waste", content)
            self.assertNotIn(f"/prediction/{self.user2_pred_id}", content)

    def test_03_empty_history_displays_honest_empty_message(self):
        """Verify user with 0 predictions sees 'No waste analyses yet.'."""
        # Create user with 0 scans
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO users (name, email, password_hash, role) VALUES (%s, %s, %s, 'user')",
                ('Zero Scans', 'zero@phase5-test.com', generate_password_hash('Password123'))
            )
        conn.close()

        with self.client as c:
            c.post('/login', data={'email': 'zero@phase5-test.com', 'password': 'Password123'})
            response = c.get('/history')
            self.assertEqual(response.status_code, 200)
            content = response.data.decode('utf-8')
            self.assertIn("No waste analyses yet.", content)

    def test_04_prediction_detail_owner_access(self):
        """Verify owner can access full prediction detail page."""
        with self.client as c:
            c.post('/login', data={'email': 'alice@phase5-test.com', 'password': 'Password123'})
            response = c.get(f'/prediction/{self.user1_pred_id}')
            self.assertEqual(response.status_code, 200)
            content = response.data.decode('utf-8')

            self.assertIn("Waste Analysis #", content)
            self.assertIn("Wet Waste", content)
            self.assertIn("91.5", content)
            self.assertIn("Green Bin", content)
            self.assertIn("Classification Feedback", content)

    def test_05_prediction_detail_unauthorized_user_denied(self):
        """Verify User 2 is denied (403 Forbidden) when accessing User 1's prediction."""
        with self.client as c:
            c.post('/login', data={'email': 'bob@phase5-test.com', 'password': 'Password123'})
            response = c.get(f'/prediction/{self.user1_pred_id}')
            # Must return 403 Forbidden
            self.assertEqual(response.status_code, 403)

    def test_06_admin_can_view_any_prediction_detail(self):
        """Verify administrator can inspect any user's prediction detail."""
        with self.client as c:
            c.post('/login', data={'email': 'admin@phase5-test.com', 'password': 'AdminPass123'})
            response = c.get(f'/prediction/{self.user1_pred_id}')
            self.assertEqual(response.status_code, 200)

    def test_07_feedback_submission_success(self):
        """Verify submitting feedback records into prediction_feedback and updates prediction status."""
        with self.client as c:
            c.post('/login', data={'email': 'alice@phase5-test.com', 'password': 'Password123'})
            response = c.post(
                f'/prediction/{self.user1_pred_id}/feedback',
                data={'helpful': 'yes', 'comment': 'Correctly identified organic scrap.'},
                follow_redirects=True
            )
            self.assertEqual(response.status_code, 200)
            content = response.data.decode('utf-8')
            self.assertIn("Thank you for your feedback", content)

        # Check in MySQL
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT feedback_type, user_notes FROM prediction_feedback WHERE prediction_id = %s",
                (self.user1_pred_id,)
            )
            fb = cursor.fetchone()
            self.assertIsNotNone(fb)
            self.assertEqual(fb['feedback_type'], 'confirm')
            self.assertEqual(fb['user_notes'], 'Correctly identified organic scrap.')

            cursor.execute("SELECT user_feedback FROM predictions WHERE id = %s", (self.user1_pred_id,))
            pred = cursor.fetchone()
            self.assertEqual(pred['user_feedback'], 'correct')
        conn.close()

    def test_08_duplicate_feedback_updates_safely(self):
        """Verify submitting feedback again on the same prediction updates the existing record."""
        with self.client as c:
            c.post('/login', data={'email': 'alice@phase5-test.com', 'password': 'Password123'})

            # First feedback
            c.post(
                f'/prediction/{self.user1_pred_id}/feedback',
                data={'helpful': 'yes', 'comment': 'First comment'}
            )

            # Updated feedback
            c.post(
                f'/prediction/{self.user1_pred_id}/feedback',
                data={'helpful': 'no', 'comment': 'Actually it was a dry wrapper'}
            )

        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(*) AS total, feedback_type, user_notes FROM prediction_feedback WHERE prediction_id = %s",
                (self.user1_pred_id,)
            )
            fb = cursor.fetchone()
            # Must only have 1 feedback record, safely updated
            self.assertEqual(fb['total'], 1)
            self.assertEqual(fb['feedback_type'], 'correction')
            self.assertEqual(fb['user_notes'], 'Actually it was a dry wrapper')

            cursor.execute("SELECT user_feedback FROM predictions WHERE id = %s", (self.user1_pred_id,))
            pred = cursor.fetchone()
            self.assertEqual(pred['user_feedback'], 'incorrect')
        conn.close()

    def test_09_unauthorized_user_cannot_submit_feedback(self):
        """Verify User 2 cannot submit feedback on User 1's prediction."""
        with self.client as c:
            c.post('/login', data={'email': 'bob@phase5-test.com', 'password': 'Password123'})
            response = c.post(
                f'/prediction/{self.user1_pred_id}/feedback',
                data={'helpful': 'yes', 'comment': 'Hacking feedback'}
            )
            self.assertEqual(response.status_code, 403)

    def test_10_api_history_and_prediction_endpoints(self):
        """Verify JSON APIs for history and prediction details with ownership check."""
        with self.client as c:
            c.post('/login', data={'email': 'alice@phase5-test.com', 'password': 'Password123'})

            # 1. API History
            res_hist = c.get('/api/history')
            self.assertEqual(res_hist.status_code, 200)
            data_hist = res_hist.get_json()
            self.assertTrue(data_hist['success'])
            self.assertEqual(len(data_hist['predictions']), 1)
            self.assertEqual(data_hist['predictions'][0]['category_name'], 'Wet Waste')

            # 2. API Prediction Detail (Owner)
            res_detail = c.get(f'/api/prediction/{self.user1_pred_id}')
            self.assertEqual(res_detail.status_code, 200)
            data_detail = res_detail.get_json()
            self.assertTrue(data_detail['success'])
            self.assertEqual(data_detail['prediction']['category_name'], 'Wet Waste')

            # 3. API Prediction Detail (Unauthorized)
            res_unauth = c.get(f'/api/prediction/{self.user2_pred_id}')
            self.assertEqual(res_unauth.status_code, 403)

    def test_11_dashboard_accurate_counts_from_database(self):
        """Verify dashboard displays exact real counts from MySQL (1 Wet, 0 Dry, 0 Recyclable, 0 E-Waste)."""
        with self.client as c:
            c.post('/login', data={'email': 'alice@phase5-test.com', 'password': 'Password123'})
            response = c.get('/dashboard')
            self.assertEqual(response.status_code, 200)
            content = response.data.decode('utf-8')

            # Alice has 1 Wet Waste analysis
            self.assertIn("Personal Waste Category Distribution", content)
            self.assertIn("1 items (100.0%)", content)
            # Ensure no hardcoded or fake statistics
            self.assertNotIn("125 scans", content)
            self.assertNotIn("94%", content)

    def test_12_admin_dashboard_metrics(self):
        """Verify admin dashboard displays real platform metrics from MySQL."""
        with self.client as c:
            c.post('/login', data={'email': 'admin@phase5-test.com', 'password': 'AdminPass123'})
            response = c.get('/admin')
            self.assertEqual(response.status_code, 200)
            content = response.data.decode('utf-8')

            self.assertIn("Administrator Control Panel", content)
            self.assertIn("Platform-Wide Waste Category Distribution", content)
            self.assertIn("Total Analyses", content)


if __name__ == '__main__':
    unittest.main()
