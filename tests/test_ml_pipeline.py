"""
EcoSort AI - ML Training Pipeline Unit Tests
Verifies dataset validation, class alignment, metrics computation, and pre-flight guards.
"""

import os
import json
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np

from ml_training.config import (
    CLASS_NAMES_DIR,
    CLASS_DISPLAY_NAMES,
    CLASS_TO_INDEX,
    INDEX_TO_DISPLAY,
    MODEL_OUTPUT_PATH,
    CLASS_INDICES_PATH,
    IMAGE_SIZE
)
from ml_training.validate_dataset import validate_dataset, count_and_validate_split
from ml_training.train_model import save_class_indices, train_model
from ml_training.evaluate_model import (
    compute_metrics_numpy,
    format_classification_report,
    format_confusion_matrix_ascii,
    evaluate_model
)
from ai_engine.categories import WASTE_CATEGORIES, CATEGORY_NAMES


class TestMLPipeline(unittest.TestCase):
    """Test suite for ML training and evaluation utilities."""

    def test_class_alignment_with_ai_engine(self):
        """Ensures ML training class definitions match ai_engine/categories.py exactly."""
        for idx, cat_key in enumerate(CLASS_NAMES_DIR):
            self.assertEqual(WASTE_CATEGORIES[idx]['key'], cat_key)
            self.assertEqual(WASTE_CATEGORIES[idx]['name'], INDEX_TO_DISPLAY[idx])
            self.assertEqual(CLASS_TO_INDEX[cat_key], idx)

    def test_class_indices_json_generation(self):
        """Verifies save_class_indices creates a valid JSON map."""
        test_path = os.path.join(os.path.dirname(CLASS_INDICES_PATH), 'test_indices.json')
        mapping = save_class_indices(test_path)
        self.assertTrue(os.path.exists(test_path))

        with open(test_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.assertIn('class_to_index', data)
        self.assertIn('index_to_class', data)
        self.assertIn('categories', data)
        self.assertEqual(data['categories'], CLASS_NAMES_DIR)

        if os.path.exists(test_path):
            os.remove(test_path)

    def test_dataset_validator_real_dataset(self):
        """Verifies dataset validator detects the real dataset images honestly."""
        report = validate_dataset(verbose=False)
        self.assertIsInstance(report, dict)
        self.assertIn('is_valid', report)
        self.assertIn('has_images', report)
        self.assertIn('total_images', report)
        self.assertIn('summary_table', report)

        # Dataset currently contains verified real images
        self.assertTrue(report['has_images'])
        self.assertTrue(report['is_valid'])
        self.assertEqual(report['counts']['totals_per_split']['test'], 9864)
        self.assertGreater(report['total_images'], 9864)

    def test_metrics_computation_numpy(self):
        """Verifies accuracy, precision, recall, F1, and confusion matrix math."""
        # Simulated scenario: 4 classes, 8 samples
        # Class 0: true [0, 0], pred [0, 0] -> TP=2, FP=0, FN=0
        # Class 1: true [1, 1], pred [1, 2] -> TP=1, FP=0, FN=1
        # Class 2: true [2, 2], pred [2, 1] -> TP=1, FP=1, FN=1
        # Class 3: true [3, 3], pred [3, 3] -> TP=2, FP=0, FN=0
        y_true = [0, 0, 1, 1, 2, 2, 3, 3]
        y_pred = [0, 0, 1, 2, 2, 1, 3, 3]

        metrics = compute_metrics_numpy(y_true, y_pred, num_classes=4)
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics['total_samples'], 8)
        # 6 correct out of 8 -> accuracy 0.75
        self.assertEqual(metrics['accuracy'], 0.75)

        # Check Wet Waste (class 0): 2/2 = 1.0
        self.assertEqual(metrics['per_class']['Wet Waste']['precision'], 1.0)
        self.assertEqual(metrics['per_class']['Wet Waste']['recall'], 1.0)
        self.assertEqual(metrics['per_class']['Wet Waste']['f1_score'], 1.0)

        # Confusion Matrix check
        cm = metrics['confusion_matrix']
        self.assertEqual(len(cm), 4)
        self.assertEqual(len(cm[0]), 4)
        self.assertEqual(cm[0][0], 2)
        self.assertEqual(cm[3][3], 2)

    def test_metrics_empty_samples(self):
        """Verifies compute_metrics handles empty inputs without crashing."""
        metrics = compute_metrics_numpy([], [], num_classes=4)
        self.assertIsNone(metrics)

    def test_classification_report_formatting(self):
        """Verifies report text and confusion matrix ASCII formatters."""
        y_true = [0, 1, 2, 3]
        y_pred = [0, 1, 2, 3]
        metrics = compute_metrics_numpy(y_true, y_pred, num_classes=4)

        report_text = format_classification_report(metrics)
        self.assertIn("Precision", report_text)
        self.assertIn("Recall", report_text)
        self.assertIn("F1-Score", report_text)
        self.assertIn("Wet Waste", report_text)

        cm_ascii = format_confusion_matrix_ascii(metrics['confusion_matrix'])
        self.assertIn("True \\ Pred", cm_ascii)

    @patch('ml_training.train_model.validate_dataset', return_value={'is_valid': False})
    def test_train_model_empty_dataset_guard(self, mock_val):
        """Ensures train_model suspends cleanly and returns False when dataset is invalid."""
        result = train_model()
        self.assertFalse(result)

    def test_evaluate_model_missing_model_guard(self):
        """Ensures evaluate_model suspends cleanly and returns None when model is absent."""
        result = evaluate_model(model_path='nonexistent_model_path.keras')
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
