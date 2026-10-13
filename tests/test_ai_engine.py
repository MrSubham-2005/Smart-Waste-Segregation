"""
EcoSort AI - Phase 3 AI Engine & Model Service Unit Tests
Tests category mappings, image preprocessing pipeline, confidence thresholding,
graceful untrained model handling, and API integration.
"""

import io
import sys
import unittest
from unittest.mock import patch
from pathlib import Path
from PIL import Image
import numpy as np

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app import app
from ai_engine import (
    WASTE_CATEGORIES,
    CATEGORY_NAMES,
    get_category_by_index,
    get_category_by_name,
    get_category_id_by_name,
    get_all_categories,
    is_valid_image,
    preprocess_image,
    TARGET_IMAGE_SIZE,
    is_model_available,
    get_model_info,
    predict_waste,
    DEFAULT_CONFIDENCE_THRESHOLD
)


class AIEngineTestCase(unittest.TestCase):
    """Test suite for the AI engine modules and model service."""

    def setUp(self):
        """Set up Flask test client and create in-memory test images."""
        app.config['TESTING'] = True
        self.client = app.test_client()

        # Authenticate client for API tests
        with self.client.session_transaction() as sess:
            sess['user_id'] = 1
            sess['name'] = 'AI Engine Test'
            sess['email'] = 'test@ai-engine.com'
            sess['role'] = 'user'

        # In-memory RGB Image (100x150)
        rgb_img = Image.new('RGB', (100, 150), color=(73, 109, 137))
        self.rgb_bytes_io = io.BytesIO()
        rgb_img.save(self.rgb_bytes_io, format='JPEG')
        self.rgb_bytes = self.rgb_bytes_io.getvalue()
        self.rgb_bytes_io.seek(0)

        # In-memory RGBA Image with transparency (200x200)
        rgba_img = Image.new('RGBA', (200, 200), color=(255, 0, 0, 128))
        self.rgba_bytes_io = io.BytesIO()
        rgba_img.save(self.rgba_bytes_io, format='PNG')
        self.rgba_bytes = self.rgba_bytes_io.getvalue()
        self.rgba_bytes_io.seek(0)

        # In-memory Grayscale Image (50x50)
        gray_img = Image.new('L', (50, 50), color=128)
        self.gray_bytes_io = io.BytesIO()
        gray_img.save(self.gray_bytes_io, format='PNG')
        self.gray_bytes = self.gray_bytes_io.getvalue()
        self.gray_bytes_io.seek(0)

    # =================================================================
    # 1. Category Mapping & Rule Engine Tests
    # =================================================================
    def test_01_core_categories_count_and_names(self):
        """Verify the 4 required categories exist with exact names."""
        self.assertEqual(len(WASTE_CATEGORIES), 4)
        expected_names = ['Wet Waste', 'Dry Waste', 'Recyclable Waste', 'E-Waste']
        self.assertEqual(CATEGORY_NAMES, expected_names)

    def test_02_category_metadata_fields(self):
        """Verify each category contains required SDG 12 attributes and bin rules."""
        required_keys = {
            'id', 'index', 'name', 'key', 'bin_name', 'bin_color_code',
            'description', 'disposal_instructions', 'examples', 'sdg_impact'
        }
        for idx, cat in WASTE_CATEGORIES.items():
            self.assertTrue(required_keys.issubset(cat.keys()), f"Missing keys in category index {idx}")
            self.assertTrue(cat['bin_color_code'].startswith('#'))
            self.assertGreater(len(cat['disposal_instructions']), 20)

    def test_03_category_lookups(self):
        """Verify category lookup functions by index and by name."""
        wet = get_category_by_index(0)
        self.assertEqual(wet['name'], 'Wet Waste')
        self.assertEqual(wet['id'], 1)

        ewaste = get_category_by_name('E-Waste')
        self.assertEqual(ewaste['index'], 3)
        self.assertEqual(ewaste['id'], 4)

        self.assertEqual(get_category_id_by_name('Recyclable Waste'), 3)
        self.assertEqual(get_category_id_by_name('Dry Waste'), 2)
        self.assertIsNone(get_category_by_name('NonExistentCategory'))

    # =================================================================
    # 2. Image Validation & Preprocessing Pipeline Tests
    # =================================================================
    def test_04_image_validator_accepts_valid_images(self):
        """Verify validator accepts JPEG, PNG, RGBA, and Grayscale images."""
        valid_rgb, err1 = is_valid_image(io.BytesIO(self.rgb_bytes))
        self.assertTrue(valid_rgb, f"RGB failed: {err1}")

        valid_rgba, err2 = is_valid_image(io.BytesIO(self.rgba_bytes))
        self.assertTrue(valid_rgba, f"RGBA failed: {err2}")

        valid_gray, err3 = is_valid_image(io.BytesIO(self.gray_bytes))
        self.assertTrue(valid_gray, f"Grayscale failed: {err3}")

    def test_05_image_validator_rejects_corrupted_data(self):
        """Verify validator rejects non-image or corrupt binary bytes."""
        corrupt_bytes = b"This is definitely not an image file content."
        valid, err = is_valid_image(io.BytesIO(corrupt_bytes))
        self.assertFalse(valid)
        self.assertIn("Invalid or corrupted", err)

        valid_empty, err_empty = is_valid_image(io.BytesIO(b""))
        self.assertFalse(valid_empty)

    def test_06_preprocessor_tensor_dimensions_and_type(self):
        """Verify preprocessor produces correct MobileNetV2 input shape (1, 224, 224, 3) and float32."""
        tensor, err = preprocess_image(io.BytesIO(self.rgb_bytes))
        self.assertIsNone(err)
        self.assertIsNotNone(tensor)
        self.assertEqual(tensor.shape, (1, 224, 224, 3))
        self.assertEqual(tensor.dtype, np.float32)

    def test_07_preprocessor_pixel_normalization(self):
        """Verify MobileNetV2 normalization [-1.0, 1.0]."""
        tensor, _ = preprocess_image(io.BytesIO(self.rgb_bytes))
        min_val = float(np.min(tensor))
        max_val = float(np.max(tensor))
        self.assertGreaterEqual(min_val, -1.0)
        self.assertLessEqual(max_val, 1.0)

    def test_08_preprocessor_handles_rgba_and_grayscale(self):
        """Verify RGBA and Grayscale images are properly converted to 3-channel RGB."""
        tensor_rgba, err1 = preprocess_image(io.BytesIO(self.rgba_bytes))
        self.assertIsNone(err1)
        self.assertEqual(tensor_rgba.shape, (1, 224, 224, 3))

        tensor_gray, err2 = preprocess_image(io.BytesIO(self.gray_bytes))
        self.assertIsNone(err2)
        self.assertEqual(tensor_gray.shape, (1, 224, 224, 3))

    # =================================================================
    # 3. Model Service & Untrained Model Handling Tests
    # =================================================================
    def test_09_model_available_flag_reflects_reality(self):
        """Ensure is_model_available correctly reports presence of trained model file."""
        available = is_model_available()
        self.assertTrue(available, "Trained MobileNetV2 model file should exist.")

    def test_10_get_model_info_truthfulness(self):
        """Verify get_model_info transparently indicates model is trained and ready."""
        info = get_model_info()
        self.assertTrue(info['is_trained'])
        self.assertEqual(info['confidence_threshold'], DEFAULT_CONFIDENCE_THRESHOLD)
        self.assertIn("ready for inference", info['status'].lower())
        self.assertEqual(info['target_classes'], CATEGORY_NAMES)

    @patch('ai_engine.model_service.is_model_available', return_value=False)
    def test_11_predict_waste_when_untrained_never_fakes_predictions(self, mock_avail):
        """Crucial academic requirement: predict_waste must NEVER return fake predictions when untrained."""
        result = predict_waste(io.BytesIO(self.rgb_bytes))
        self.assertFalse(result['success'])
        self.assertEqual(result['status'], 'model_not_trained')
        self.assertFalse(result['is_trained'])
        self.assertIsNone(result['prediction'])
        self.assertIsNone(result['category'])
        self.assertIsNone(result['confidence'])
        self.assertIn("has not been trained yet", result['message'])

    def test_11b_predict_waste_when_trained_returns_real_prediction(self):
        """Verify predict_waste returns genuine prediction when trained model is present."""
        result = predict_waste(io.BytesIO(self.rgb_bytes))
        self.assertTrue(result['success'])
        self.assertTrue(result['is_trained'])
        self.assertIn(result['category_name'], CATEGORY_NAMES)
        self.assertGreaterEqual(result['confidence_percentage'], 0.0)
        self.assertIsNotNone(result['bin_name'])

    # =================================================================
    # 4. API Endpoints Integration Tests
    # =================================================================
    def test_12_api_categories_endpoint(self):
        """Verify GET /api/categories returns HTTP 200 and all 4 categories."""
        response = self.client.get('/api/categories')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data['success'])
        self.assertEqual(len(data['categories']), 4)

    def test_12b_api_predict_unauthenticated_returns_401(self):
        """Verify POST /api/predict without authentication returns 401 Unauthorized."""
        anon_client = app.test_client()
        response = anon_client.post('/api/predict', data={})
        self.assertEqual(response.status_code, 401)
        data = response.get_json()
        self.assertFalse(data['success'])
        self.assertEqual(data['status'], 'unauthorized')

    def test_13_api_predict_without_file_returns_400(self):
        """Verify POST /api/predict without an image file returns 400 Bad Request."""
        response = self.client.post('/api/predict', data={})
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data['success'])
        self.assertEqual(data['status'], 'no_file')

    @patch('ai_engine.model_service.is_model_available', return_value=False)
    def test_14_api_predict_with_untrained_model_reports_honestly(self, mock_avail):
        """Verify POST /api/predict with untrained model reports status 'model_not_trained'."""
        data = {
            'image': (io.BytesIO(self.rgb_bytes), 'test_sample.jpg')
        }
        response = self.client.post(
            '/api/predict',
            data=data,
            content_type='multipart/form-data'
        )
        self.assertEqual(response.status_code, 200)
        res_json = response.get_json()
        self.assertEqual(res_json['status'], 'model_not_trained')
        self.assertFalse(res_json['is_trained'])
        self.assertIsNone(res_json['category'])

    def test_14b_api_predict_with_trained_model_succeeds(self):
        """Verify POST /api/predict with real trained model succeeds and returns predictions."""
        data = {
            'image': (io.BytesIO(self.rgb_bytes), 'test_sample.jpg')
        }
        response = self.client.post(
            '/api/predict',
            data=data,
            content_type='multipart/form-data'
        )
        self.assertEqual(response.status_code, 200)
        res_json = response.get_json()
        self.assertTrue(res_json['success'])
        self.assertTrue(res_json['is_trained'])
        self.assertIn(res_json['category_name'], CATEGORY_NAMES)

    def test_15_api_health_endpoint_phase_3(self):
        """Verify GET /api/health reflects system health with full AI engine info."""
        response = self.client.get('/api/health')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn(data['phase'], [3, 7])
        self.assertIn('ai_engine', data)
        self.assertTrue(data['ai_engine']['is_trained'])
        self.assertEqual(data['ai_engine']['model_architecture'], 'MobileNetV2 (Transfer Learning)')


if __name__ == '__main__':
    unittest.main()
