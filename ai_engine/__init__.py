"""
EcoSort AI - AI Engine Package
Provides image preprocessing, category mappings, and model inference services.
"""

from .categories import (
    WASTE_CATEGORIES,
    CATEGORY_NAMES,
    get_category_by_index,
    get_category_by_name,
    get_category_id_by_name,
    get_all_categories
)
from .preprocessor import (
    is_valid_image,
    preprocess_image,
    TARGET_IMAGE_SIZE
)
from .model_service import (
    is_model_available,
    get_model_info,
    predict_waste,
    DEFAULT_CONFIDENCE_THRESHOLD
)

__all__ = [
    'WASTE_CATEGORIES',
    'CATEGORY_NAMES',
    'get_category_by_index',
    'get_category_by_name',
    'get_category_id_by_name',
    'get_all_categories',
    'is_valid_image',
    'preprocess_image',
    'TARGET_IMAGE_SIZE',
    'is_model_available',
    'get_model_info',
    'predict_waste',
    'DEFAULT_CONFIDENCE_THRESHOLD'
]
