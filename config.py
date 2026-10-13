"""
EcoSort AI - Configuration Module
Smart Waste Segregation System
"""

import os
from pathlib import Path

# Load .env file if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Base Project Directory
BASE_DIR = Path(__file__).resolve().parent


class Config:
    """Base application configuration class."""

    # Project Directories
    BASE_DIR = BASE_DIR

    # Flask Settings
    SECRET_KEY = os.environ.get('SECRET_KEY', 'ecosort_ai_academic_secret_key_2026')
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() in ('true', '1', 't')

    # Database Configuration (MySQL / XAMPP)
    DB_HOST = os.environ.get('DB_HOST', 'localhost')
    DB_PORT = int(os.environ.get('DB_PORT', 3306))
    DB_USER = os.environ.get('DB_USER', 'root')
    DB_PASSWORD = os.environ.get('DB_PASSWORD', '')
    DB_NAME = os.environ.get('DB_NAME', 'ecosort_ai')

    # File Upload Configuration
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH', 5 * 1024 * 1024))  # 5 MB max
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

    # AI Model Artifact Paths
    MODEL_DIR = os.path.join(BASE_DIR, 'ai_engine', 'models')
    MODEL_PATH = os.path.join(MODEL_DIR, 'ecosort_mobilenetv2.keras')
    CLASS_INDICES_PATH = os.path.join(MODEL_DIR, 'class_indices.json')

    # Target Waste Categories
    CATEGORIES = ['Wet Waste', 'Dry Waste', 'Recyclable Waste', 'E-Waste']
