"""
EcoSort AI - Model Loading & Prediction Service
Encapsulates Keras model management, inference execution, category mapping,
confidence thresholding, and graceful untrained-model handling.
"""

import os
from pathlib import Path
import numpy as np

from config import Config
from .categories import (
    WASTE_CATEGORIES,
    CATEGORY_NAMES,
    get_category_by_index
)
from .preprocessor import (
    preprocess_image,
    is_valid_image
)

# Global model cache (lazy-loaded on first inference request)
_LOADED_MODEL = None
_LOADED_MODEL_PATH = None

# Default classification confidence threshold (50%)
DEFAULT_CONFIDENCE_THRESHOLD = 0.50


def is_model_available() -> bool:
    """Checks whether the trained MobileNetV2 model artifact exists on disk."""
    return os.path.isfile(Config.MODEL_PATH)


def get_model_info() -> dict:
    """
    Returns transparent diagnostic information about the AI model status.
    Academic integrity: truthfully reports whether a trained model exists.
    """
    available = is_model_available()
    model_size = os.path.getsize(Config.MODEL_PATH) if available else 0

    return {
        'is_trained': available,
        'model_architecture': 'MobileNetV2 (Transfer Learning)',
        'target_input_shape': '(224, 224, 3)',
        'target_classes': CATEGORY_NAMES,
        'model_path': Config.MODEL_PATH,
        'file_exists': available,
        'file_size_bytes': model_size,
        'confidence_threshold': DEFAULT_CONFIDENCE_THRESHOLD,
        'status': (
            'Trained model weights ready for inference'
            if available
            else 'Model not trained yet (Phase 6 training pipeline ready)'
        )
    }


def load_model(force_reload=False):
    """
    Loads the Keras model from disk using lazy loading.
    Only imports TensorFlow when model weights are actually present.
    
    :param force_reload: If True, forces reloading from disk even if cached.
    :return: Loaded tf.keras.Model or None if model file does not exist.
    """
    global _LOADED_MODEL, _LOADED_MODEL_PATH

    if not is_model_available():
        _LOADED_MODEL = None
        _LOADED_MODEL_PATH = None
        return None

    if _LOADED_MODEL is not None and not force_reload and _LOADED_MODEL_PATH == Config.MODEL_PATH:
        return _LOADED_MODEL

    try:
        from .keras_engine import load_model_from_file
        _LOADED_MODEL = load_model_from_file(Config.MODEL_PATH)
        _LOADED_MODEL_PATH = Config.MODEL_PATH
        return _LOADED_MODEL
    except Exception as e:
        print(f"[AI ENGINE ERROR] Failed to load model from {Config.MODEL_PATH}: {e}")
        _LOADED_MODEL = None
        return None


def predict_waste(image_source, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD) -> dict:
    """
    Executes end-to-end waste classification on an image input.
    
    Behavior:
    1. If model is not trained yet: Gracefully returns 'model_not_trained' status.
       Does NOT fabricate fake predictions, fake classes, or fake accuracy.
    2. If image is invalid or corrupted: Returns 'validation_error'.
    3. If preprocessing fails: Returns 'preprocessing_error'.
    4. If inference succeeds:
       - Computes Softmax probabilities for all 4 categories.
       - Identifies top predicted category and confidence percentage.
       - Applies confidence thresholding to flag ambiguous/low-certainty items.
       - Maps category to disposal instructions and SDG 12 guidance.
       
    :param image_source: File path, file-like stream, bytes, or PIL Image.
    :param confidence_threshold: Minimum confidence required (0.0 - 1.0, default 0.50).
    :return: Structured dictionary with prediction results and disposal advice.
    """
    # 1. Truthful Untrained Model Check
    if not is_model_available():
        return {
            'success': False,
            'status': 'model_not_trained',
            'is_trained': False,
            'message': (
                'The waste classification model is currently being prepared and has not been trained yet. '
                'Please train the model before using AI classification.'
            ),
            'prediction': None,
            'category': None,
            'confidence': None,
            'confidence_percentage': None,
            'bin_info': None,
            'disposal_advice': None,
            'probabilities': None
        }

    # 2. Image Validation
    is_valid, val_err = is_valid_image(image_source)
    if not is_valid:
        return {
            'success': False,
            'status': 'validation_error',
            'is_trained': True,
            'message': val_err,
            'prediction': None
        }

    # 3. Image Preprocessing
    tensor_input, prep_err = preprocess_image(image_source)
    if prep_err is not None:
        return {
            'success': False,
            'status': 'preprocessing_error',
            'is_trained': True,
            'message': prep_err,
            'prediction': None
        }

    # 4. Load Model
    model = load_model()
    if model is None:
        return {
            'success': False,
            'status': 'model_load_error',
            'is_trained': True,
            'message': 'Failed to instantiate the trained neural network model.',
            'prediction': None
        }

    # 5. Run Forward Inference
    try:
        raw_predictions = model.predict(tensor_input, verbose=0)
        probabilities = raw_predictions[0]  # Array of 4 probabilities

        # Determine top predicted class
        top_idx = int(np.argmax(probabilities))
        top_confidence = float(probabilities[top_idx])
        confidence_percent = round(top_confidence * 100.0, 2)

        # Retrieve full category metadata
        category_meta = get_category_by_index(top_idx)

        # Build full distribution breakdown for transparency
        class_distribution = {
            CATEGORY_NAMES[i]: round(float(probabilities[i]) * 100.0, 2)
            for i in range(len(CATEGORY_NAMES))
        }

        # 6. Confidence Threshold Handling
        is_low_confidence = top_confidence < confidence_threshold

        status = 'low_confidence' if is_low_confidence else 'success'
        warning_msg = (
            "Confidence is below threshold (50%). Item may be ambiguous, "
            "partially occluded, or outside the training domain."
            if is_low_confidence
            else ""
        )

        return {
            'success': True,
            'status': status,
            'is_trained': True,
            'is_low_confidence': is_low_confidence,
            'message': warning_msg or 'Waste classified successfully.',
            'category_id': category_meta['id'],
            'category_index': top_idx,
            'category_name': category_meta['name'],
            'confidence': top_confidence,
            'confidence_percentage': confidence_percent,
            'bin_name': category_meta['bin_name'],
            'bin_color_code': category_meta['bin_color_code'],
            'badge_class': category_meta['badge_class'],
            'disposal_instructions': category_meta['disposal_instructions'],
            'description': category_meta['description'],
            'examples': category_meta['examples'],
            'sdg_impact': category_meta['sdg_impact'],
            'environmental_impact': category_meta['sdg_impact'],
            'class_distribution': class_distribution
        }

    except Exception as e:
        return {
            'success': False,
            'status': 'inference_error',
            'is_trained': True,
            'message': f"Inference execution failed: {str(e)}",
            'prediction': None
        }
