"""
EcoSort AI - Phase 4 Frontend Classifier UI & Integration Tests
Tests routes (/analyze, /awareness, /), navigation, dashboard real stats,
file upload validation, and truthful untrained model response.
"""

import io
import sys
import unittest
from unittest.mock import patch
from pathlib import Path
from PIL import Image

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app import app
from database.db_connection import get_db_connection


class FrontendIntegrationTestCase(unittest.TestCase):
    """Test suite for Phase 4 UI routes, analyzer integration, and templates."""

    def setUp(self):
        """Configure test client and in-memory test image."""
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        self.client = app.test_client()

        # In-memory test JPEG image
        img = Image.new('RGB', (120, 120), color=(50, 150, 200))
        self.img_bytes = io.BytesIO()
        img.save(self.img_bytes, format='JPEG')
        self.img_bytes.seek(0)

        self._cleanup_test_users()

    def tearDown(self):
        self._cleanup_test_users()

    def _cleanup_test_users(self):
        try:
            conn = get_db_connection()
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM users WHERE email LIKE '%@fe-test.com'")
            conn.close()
        except Exception:
            pass

    def test_01_analyze_page_allows_anonymous_and_auth_users(self):
        """Verify /analyze returns 200 OK for anonymous users as well as logged-in users."""
        # 1. Unauthenticated request must return 200 OK with dropzone, preview, and category reference
        anon_resp = self.client.get('/analyze', follow_redirects=False)
        self.assertEqual(anon_resp.status_code, 200)
        anon_content = anon_resp.data.decode('utf-8')
        self.assertIn("Analyze Your Waste", anon_content)
        self.assertIn("upload-dropzone", anon_content)
        self.assertIn("waste-image-input", anon_content)
        self.assertIn("preview-container", anon_content)
        self.assertIn("analyzer.js", anon_content)
        self.assertIn("Wet Waste", anon_content)
        self.assertIn("Dry Waste", anon_content)
        self.assertIn("Recyclable Waste", anon_content)
        self.assertIn("E-Waste", anon_content)
        self.assertIn("isLoggedIn: false", anon_content)
        self.assertIn("loginUrl:", anon_content)
        self.assertIn('/login?next=/analyze', anon_content)

        # Verify static/js/analyzer.js does not contain the old login-required warning panel
        js_path = ROOT_DIR / "static" / "js" / "analyzer.js"
        with open(js_path, "r", encoding="utf-8") as f:
            js_content = f.read()
        self.assertNotIn("Please log in to analyze this waste image.", js_content)
        self.assertNotIn("Login is required to analyze waste images with AI", js_content)
        self.assertNotIn("renderLoginRequired", js_content)
        self.assertIn("window.location.href = loginUrl", js_content)

        # 2. Authenticated request also renders the full analyzer interface with isLoggedIn: true
        with self.client.session_transaction() as sess:
            sess['user_id'] = 99
            sess['name'] = 'Auth User'
            sess['email'] = 'auth@fe-test.com'
            sess['role'] = 'user'

        response = self.client.get('/analyze')
        self.assertEqual(response.status_code, 200)
        content = response.data.decode('utf-8')

        self.assertIn("Analyze Your Waste", content)
        self.assertIn("upload-dropzone", content)
        self.assertIn("waste-image-input", content)
        self.assertIn("preview-container", content)
        self.assertIn("analyzer.js", content)
        self.assertIn("isLoggedIn: true", content)
        self.assertIn("Wet Waste", content)
        self.assertIn("Dry Waste", content)
        self.assertIn("Recyclable Waste", content)
        self.assertIn("E-Waste", content)

    def test_02_awareness_page_loads_with_categories_and_no_sdg12(self):
        """Verify /awareness renders 200 OK with environmental context and NO SDG 12."""
        response = self.client.get('/awareness')
        self.assertEqual(response.status_code, 200)
        content = response.data.decode('utf-8')

        self.assertIn("Understanding Waste Segregation", content)
        self.assertIn("Environmental Education", content)
        self.assertNotIn("SDG 12", content)
        self.assertIn("Wet Waste", content)
        self.assertIn("Dry Waste", content)
        self.assertIn("Recyclable Waste", content)
        self.assertIn("E-Waste", content)

    def test_03_home_page_includes_required_tagline_and_buttons(self):
        """Verify home page has required slogan, buttons, updated bin labels, and no SDG 12 or AICTE badges."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        content = response.data.decode('utf-8')

        self.assertIn("EcoSort AI", content)
        self.assertIn("Smart Waste Segregation System", content)
        self.assertIn("Identify your waste. Choose the right disposal category. Protect the environment.", content)
        self.assertIn("Analyze Waste", content)
        self.assertIn("Learn About Waste Segregation", content)
        self.assertIn("How EcoSort AI Helps", content)
        self.assertIn("How the System Works", content)
        self.assertIn("Green Bin", content)
        self.assertIn("Blue Bin", content)
        self.assertIn("Yellow Bin", content)
        self.assertIn("Red Bin", content)
        self.assertNotIn("Yellow / Blue Bin", content)
        self.assertNotIn("Red / E-Waste Bin", content)
        self.assertNotIn("SDG 12", content)
        self.assertNotIn("AICTE B.Tech Project", content)
        self.assertNotIn("Phase 6 Pipeline Ready (Untrained)", content)

    def test_04_navbar_unauthenticated_links(self):
        """Verify navbar shows Analyze Waste, Awareness, Login, Register when logged out."""
        response = self.client.get('/')
        content = response.data.decode('utf-8')

        self.assertIn("/analyze", content)
        self.assertIn("/awareness", content)
        self.assertIn("/login", content)
        self.assertIn("/register", content)
        self.assertNotIn("/logout", content)

    def test_05_navbar_authenticated_links(self):
        """Verify navbar shows Dashboard, History, Logout when logged in."""
        self.client.post('/register', data={
            'name': 'FE User',
            'email': 'user@fe-test.com',
            'password': 'Password123',
            'confirm_password': 'Password123'
        })

        with self.client as c:
            c.post('/login', data={'email': 'user@fe-test.com', 'password': 'Password123'})
            resp = c.get('/')
            content = resp.data.decode('utf-8')

            self.assertIn("/analyze", content)
            self.assertIn("/awareness", content)
            self.assertIn("/dashboard", content)
            self.assertIn("/history", content)
            self.assertIn("/logout", content)
            self.assertIn("FE User", content)

    def test_06_dashboard_shows_no_fake_statistics(self):
        """Verify dashboard for new user honestly displays 'No waste analyses yet.' and scan count 0."""
        self.client.post('/register', data={
            'name': 'Clean Slate',
            'email': 'clean@fe-test.com',
            'password': 'Password123',
            'confirm_password': 'Password123'
        })

        with self.client as c:
            c.post('/login', data={'email': 'clean@fe-test.com', 'password': 'Password123'})
            resp = c.get('/dashboard')
            self.assertEqual(resp.status_code, 200)
            content = resp.data.decode('utf-8')

            self.assertIn("No waste analyses yet.", content)
            self.assertIn("Analyze Waste Now", content)
            # Ensure no fake counts or fabricated accuracies
            self.assertNotIn("94%", content)
            self.assertNotIn("125 scans", content)
            # Verify system diagnostics section is removed
            self.assertNotIn("System Diagnostics", content)

    def test_07_history_shows_empty_state_honestly(self):
        """Verify history page for user with no scans displays 'No waste analyses yet.'."""
        self.client.post('/register', data={
            'name': 'History User',
            'email': 'hist@fe-test.com',
            'password': 'Password123',
            'confirm_password': 'Password123'
        })

        with self.client as c:
            c.post('/login', data={'email': 'hist@fe-test.com', 'password': 'Password123'})
            resp = c.get('/history')
            self.assertEqual(resp.status_code, 200)
            content = resp.data.decode('utf-8')

            self.assertIn("No waste analyses yet.", content)

    @patch('ai_engine.model_service.is_model_available', return_value=False)
    def test_08_api_predict_untrained_model_returns_honest_json(self, mock_avail):
        """Verify POST /api/predict handles upload and returns honest untrained response for auth user."""
        with self.client.session_transaction() as sess:
            sess['user_id'] = 99
        data = {
            'image': (io.BytesIO(self.img_bytes.getvalue()), 'waste_sample.jpg')
        }
        response = self.client.post('/api/predict', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)
        res_json = response.get_json()

        self.assertFalse(res_json['success'])
        self.assertEqual(res_json['status'], 'model_not_trained')
        self.assertFalse(res_json['is_trained'])
        self.assertIsNone(res_json['category'])
        self.assertIsNone(res_json['confidence'])
        self.assertIn("currently being prepared", res_json['message'].lower())

    def test_08b_api_predict_trained_model_succeeds(self):
        """Verify POST /api/predict with real trained model returns successful classification for auth user."""
        with self.client.session_transaction() as sess:
            sess['user_id'] = 99
        data = {
            'image': (io.BytesIO(self.img_bytes.getvalue()), 'waste_sample.jpg')
        }
        response = self.client.post('/api/predict', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)
        res_json = response.get_json()

        self.assertTrue(res_json['success'])
        self.assertTrue(res_json['is_trained'])
        self.assertIsNotNone(res_json['category_name'])
        self.assertIsNotNone(res_json['bin_name'])

    def test_09_api_predict_anonymous_user_returns_401(self):
        """Verify POST /api/predict returns 401 Unauthorized for unauthenticated requests."""
        data = {
            'image': (io.BytesIO(self.img_bytes.getvalue()), 'waste_sample.jpg')
        }
        response = self.client.post('/api/predict', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 401)
        res_json = response.get_json()
        self.assertFalse(res_json['success'])
        self.assertEqual(res_json['status'], 'unauthorized')
        self.assertIn("Authentication required", res_json['message'])

    def test_10_global_bin_labels_consistency(self):
        """Verify global bin labels on awareness and analyze pages use updated wording."""
        # Awareness page
        res_aware = self.client.get('/awareness')
        content_aware = res_aware.data.decode('utf-8')
        self.assertIn("Yellow Bin (Clean Recyclables)", content_aware)
        self.assertIn("Red Bin (Specialized E-Waste Bin)", content_aware)
        self.assertNotIn("Yellow / Blue Bin", content_aware)
        self.assertNotIn("Red / Specialized E-Waste Bin", content_aware)

        # Analyze page
        res_analyze = self.client.get('/analyze')
        content_analyze = res_analyze.data.decode('utf-8')
        self.assertIn("Yellow Bin (Clean Recyclables)", content_analyze)
        self.assertIn("Red Bin (Specialized E-Waste Bin)", content_analyze)
        self.assertNotIn("Yellow / Blue Bin", content_analyze)
        self.assertNotIn("Red / Specialized E-Waste Bin", content_analyze)

        # Categories API
        res_cat = self.client.get('/api/categories')
        cats = res_cat.get_json()['categories']
        cat_map = {c['name']: c['bin_name'] for c in cats}
        self.assertEqual(cat_map['Recyclable Waste'], "Yellow Bin (Clean Recyclables)")
        self.assertEqual(cat_map['E-Waste'], "Red Bin (Specialized E-Waste Bin)")


if __name__ == '__main__':
    unittest.main()
