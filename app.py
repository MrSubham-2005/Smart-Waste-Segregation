"""
EcoSort AI - Smart Waste Segregation System
Main Flask Application
"""

import os
import re
import uuid
from flask import (
    Flask, render_template, request, redirect,
    url_for, flash, session, jsonify, abort
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from werkzeug.exceptions import HTTPException

from config import Config
from database.db_connection import get_db_connection, test_connection
from auth_decorators import login_required, admin_required
from ai_engine import (
    get_model_info,
    predict_waste,
    get_all_categories,
    is_model_available
)

# Initialize Flask application
app = Flask(__name__)
app.config.from_object(Config)

# Ensure required runtime folders exist
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

# Email validation regex pattern
EMAIL_REGEX = r'^[\w\.-]+@[\w\.-]+\.\w+$'


# =====================================================================
# Error Handlers
# =====================================================================
@app.errorhandler(403)
def forbidden_error(error):
    """Handle 403 Forbidden errors when non-admin accesses admin areas."""
    return render_template('403.html'), 403


@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 Not Found errors."""
    return render_template('base.html'), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 Internal Server errors without exposing stack traces."""
    flash("An unexpected server error occurred. Please try again later.", "danger")
    return redirect(url_for('home'))


# =====================================================================
# Public Routes
# =====================================================================
@app.route('/')
def home():
    """Landing page introducing EcoSort AI and system features."""
    return render_template('index.html', db_name=Config.DB_NAME)


@app.route('/analyze')
def analyze():
    """Waste analyzer page for image upload and classification."""
    model_info = get_model_info()
    categories = get_all_categories()
    return render_template('analyze.html', model_info=model_info, categories=categories)


@app.route('/awareness')
def awareness():
    """Public environmental awareness page explaining waste segregation."""
    categories = get_all_categories()
    return render_template('awareness.html', categories=categories)


# =====================================================================
# Authentication Routes
# =====================================================================
@app.route('/register', methods=['GET', 'POST'])
def register():
    """
    User Registration:
    - Validates name, email, password, and confirmation.
    - Ensures unique email address in MySQL users table.
    - Securely hashes password before persistence.
    - Always assigns 'user' role (never allows choosing admin).
    """
    if 'user_id' in session:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        # 1. Required fields validation
        if not name or not email or not password or not confirm_password:
            flash("All fields are required.", "danger")
            return render_template('register.html')

        # 2. Email format validation
        if not re.match(EMAIL_REGEX, email):
            flash("Please enter a valid email address.", "danger")
            return render_template('register.html')

        # 3. Password length check
        if len(password) < 6:
            flash("Password must be at least 6 characters long.", "danger")
            return render_template('register.html')

        # 4. Password confirmation check
        if password != confirm_password:
            flash("Passwords do not match. Please re-enter.", "danger")
            return render_template('register.html')

        try:
            conn = get_db_connection()
            with conn.cursor() as cursor:
                # 5. Duplicate email check
                cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
                if cursor.fetchone():
                    flash("An account with this email already exists. Please log in.", "warning")
                    conn.close()
                    return render_template('register.html')

                # 6. Secure password hashing and record insertion
                password_hash = generate_password_hash(password)
                cursor.execute(
                    "INSERT INTO users (name, email, password_hash, role) VALUES (%s, %s, %s, 'user')",
                    (name, email, password_hash)
                )

            conn.close()
            flash("Registration successful! You can now sign in.", "success")
            return redirect(url_for('login'))

        except Exception as e:
            flash("Database service temporarily unavailable. Please try again later.", "danger")
            return render_template('register.html')

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """
    User Login:
    - Validates email and password inputs.
    - Verifies password against stored werkzeug hash.
    - Initializes session with user_id, name, email, and role.
    - Supports safe 'next' redirection.
    """
    if 'user_id' in session:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        next_url = request.form.get('next', '').strip()

        if not email or not password:
            flash("Both email and password are required.", "danger")
            return render_template('login.html')

        try:
            conn = get_db_connection()
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT id, name, email, password_hash, role FROM users WHERE email = %s",
                    (email,)
                )
                user = cursor.fetchone()

            conn.close()

            # Verify credentials
            if not user or not check_password_hash(user['password_hash'], password):
                flash("Invalid email or password. Please try again.", "danger")
                return render_template('login.html')

            # Populate secure session
            session.clear()
            session['user_id'] = user['id']
            session['name'] = user['name']
            session['email'] = user['email']
            session['role'] = user['role']

            flash(f"Welcome back, {user['name']}!", "success")

            # Validate next_url to prevent open redirect vulnerabilities
            if next_url and next_url.startswith('/') and not next_url.startswith('//'):
                return redirect(next_url)

            if user['role'] == 'admin':
                return redirect(url_for('admin'))
            return redirect(url_for('dashboard'))

        except Exception as e:
            flash("Database service temporarily unavailable. Please try again later.", "danger")
            return render_template('login.html')

    return render_template('login.html')


@app.route('/logout')
def logout():
    """Clears user session and logs out."""
    session.clear()
    flash("You have been logged out successfully.", "info")
    return redirect(url_for('login'))


# =====================================================================
# Protected User Routes
# =====================================================================
@app.route('/dashboard')
@login_required
def dashboard():
    """Protected user dashboard displaying profile, real scan counts per category, and status."""
    db_status = test_connection()
    model_info = get_model_info()

    user_scans_count = 0
    recent_scans = []
    category_counts = {
        'Wet Waste': 0,
        'Dry Waste': 0,
        'Recyclable Waste': 0,
        'E-Waste': 0
    }

    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            # 1. Total count for this user
            cursor.execute(
                "SELECT COUNT(*) AS total FROM predictions WHERE user_id = %s",
                (session['user_id'],)
            )
            count_row = cursor.fetchone()
            if count_row:
                user_scans_count = count_row['total']

            # 2. Category distribution for this user
            cursor.execute(
                """
                SELECT w.category_name, COUNT(p.id) AS count
                FROM waste_categories w
                LEFT JOIN predictions p ON w.id = p.predicted_category_id AND p.user_id = %s
                GROUP BY w.id, w.category_name
                ORDER BY w.id
                """,
                (session['user_id'],)
            )
            for row in cursor.fetchall():
                if row['category_name'] in category_counts:
                    category_counts[row['category_name']] = row['count']

            # 3. Recent 5 predictions for this user
            cursor.execute(
                """
                SELECT p.id, p.image_filename, p.confidence_score, p.created_at,
                       w.category_name, w.bin_name, w.bin_color_code
                FROM predictions p
                JOIN waste_categories w ON p.predicted_category_id = w.id
                WHERE p.user_id = %s
                ORDER BY p.created_at DESC
                LIMIT 5
                """,
                (session['user_id'],)
            )
            recent_scans = cursor.fetchall()
        conn.close()
    except Exception:
        pass

    return render_template(
        'dashboard.html',
        db_status=db_status,
        model_info=model_info,
        user_scans_count=user_scans_count,
        category_counts=category_counts,
        recent_scans=recent_scans
    )


@app.route('/history')
@login_required
def history():
    """Protected prediction history view with pagination for authenticated user."""
    page = request.args.get('page', 1, type=int)
    if page < 1:
        page = 1
    per_page = 10

    total_scans = 0
    scans = []
    total_pages = 1

    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(*) AS total FROM predictions WHERE user_id = %s",
                (session['user_id'],)
            )
            count_row = cursor.fetchone()
            if count_row:
                total_scans = count_row['total']

            total_pages = max(1, (total_scans + per_page - 1) // per_page)
            if page > total_pages:
                page = total_pages
            offset = (page - 1) * per_page

            cursor.execute(
                """
                SELECT p.id, p.image_filename, p.confidence_score, p.created_at, p.user_feedback,
                       w.category_name, w.bin_name, w.bin_color_code, w.disposal_instructions
                FROM predictions p
                JOIN waste_categories w ON p.predicted_category_id = w.id
                WHERE p.user_id = %s
                ORDER BY p.created_at DESC
                LIMIT %s OFFSET %s
                """,
                (session['user_id'], per_page, offset)
            )
            scans = cursor.fetchall()
        conn.close()
    except Exception:
        pass

    return render_template(
        'history.html',
        scans=scans,
        page=page,
        total_pages=total_pages,
        total_scans=total_scans
    )


@app.route('/api/history', methods=['GET'])
@login_required
def api_history():
    """API endpoint returning paginated prediction history for current user."""
    page = request.args.get('page', 1, type=int)
    if page < 1:
        page = 1
    per_page = 10

    total_scans = 0
    scans = []
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) AS total FROM predictions WHERE user_id = %s", (session['user_id'],))
            total_scans = cursor.fetchone()['total']
            total_pages = max(1, (total_scans + per_page - 1) // per_page)
            offset = (page - 1) * per_page

            cursor.execute(
                """
                SELECT p.id, p.image_filename, p.confidence_score, p.created_at, p.user_feedback,
                       w.category_name, w.bin_name, w.bin_color_code
                FROM predictions p
                JOIN waste_categories w ON p.predicted_category_id = w.id
                WHERE p.user_id = %s
                ORDER BY p.created_at DESC
                LIMIT %s OFFSET %s
                """,
                (session['user_id'], per_page, offset)
            )
            scans = cursor.fetchall()
        conn.close()
    except Exception:
        total_pages = 1

    return jsonify({
        'success': True,
        'page': page,
        'total_pages': total_pages,
        'total_scans': total_scans,
        'predictions': scans
    })


@app.route('/prediction/<int:prediction_id>', methods=['GET'])
@login_required
def prediction_detail(prediction_id):
    """
    Displays full details of a single prediction:
    - Verifies ownership (user can only view their own predictions, unless admin).
    - Fetches category rules, bin instructions, and any existing user feedback.
    """
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT p.id, p.user_id, p.image_filename, p.confidence_score,
                       p.model_version, p.user_feedback, p.created_at,
                       w.id AS category_id, w.category_name, w.bin_name, w.bin_color_code,
                       w.description, w.disposal_instructions, w.examples
                FROM predictions p
                JOIN waste_categories w ON p.predicted_category_id = w.id
                WHERE p.id = %s
                """,
                (prediction_id,)
            )
            prediction = cursor.fetchone()

            if not prediction:
                conn.close()
                abort(404)

            # Security: User can only view their own prediction unless they are an admin
            if prediction['user_id'] != session['user_id'] and session.get('role') != 'admin':
                conn.close()
                abort(403)

            # Retrieve existing feedback if any
            cursor.execute(
                """
                SELECT id, feedback_type, user_notes, created_at
                FROM prediction_feedback
                WHERE prediction_id = %s AND user_id = %s
                ORDER BY created_at DESC LIMIT 1
                """,
                (prediction_id, session['user_id'])
            )
            existing_feedback = cursor.fetchone()

        conn.close()
        return render_template(
            'prediction_detail.html',
            prediction=prediction,
            existing_feedback=existing_feedback
        )
    except HTTPException:
        raise
    except Exception as e:
        flash("Could not retrieve prediction details.", "danger")
        return redirect(url_for('history'))


@app.route('/api/prediction/<int:prediction_id>', methods=['GET'])
@login_required
def api_prediction_detail(prediction_id):
    """JSON API endpoint returning prediction details with ownership verification."""
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT p.id, p.user_id, p.image_filename, p.confidence_score,
                       p.model_version, p.user_feedback, p.created_at,
                       w.category_name, w.bin_name, w.bin_color_code, w.disposal_instructions
                FROM predictions p
                JOIN waste_categories w ON p.predicted_category_id = w.id
                WHERE p.id = %s
                """,
                (prediction_id,)
            )
            prediction = cursor.fetchone()

            if not prediction:
                conn.close()
                return jsonify({'success': False, 'message': 'Prediction not found.'}), 404

            if prediction['user_id'] != session['user_id'] and session.get('role') != 'admin':
                conn.close()
                return jsonify({'success': False, 'message': 'Unauthorized access to prediction.'}), 403

            cursor.execute(
                "SELECT feedback_type, user_notes, created_at FROM prediction_feedback WHERE prediction_id = %s AND user_id = %s",
                (prediction_id, session['user_id'])
            )
            feedback = cursor.fetchone()
        conn.close()

        return jsonify({
            'success': True,
            'prediction': prediction,
            'feedback': feedback
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/prediction/<int:prediction_id>/feedback', methods=['POST'])
@app.route('/api/feedback', methods=['POST'])
@login_required
def submit_feedback(prediction_id=None):
    """
    Submits or updates classification feedback in prediction_feedback and predictions tables.
    Validates ownership of the prediction.
    """
    is_json = request.is_json
    if prediction_id is None and is_json:
        prediction_id = request.json.get('prediction_id')

    if not prediction_id:
        if is_json:
            return jsonify({'success': False, 'message': 'Prediction ID is required.'}), 400
        abort(400)

    helpful = (
        request.json.get('helpful') if is_json else request.form.get('helpful')
    )
    comment = (
        (request.json.get('comment') or request.json.get('notes') or '').strip()
        if is_json
        else request.form.get('comment', '').strip()
    )

    if not helpful or helpful.lower() not in ('yes', 'no', 'confirm', 'correction'):
        if is_json:
            return jsonify({'success': False, 'message': 'Feedback choice (yes/no) is required.'}), 400
        flash("Please select whether the classification was helpful.", "warning")
        return redirect(url_for('prediction_detail', prediction_id=prediction_id))

    is_helpful = helpful.lower() in ('yes', 'confirm')
    feedback_type = 'confirm' if is_helpful else 'correction'
    user_feedback_status = 'correct' if is_helpful else 'incorrect'

    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            # Verify prediction existence and ownership
            cursor.execute("SELECT id, user_id FROM predictions WHERE id = %s", (prediction_id,))
            pred = cursor.fetchone()
            if not pred:
                conn.close()
                if is_json:
                    return jsonify({'success': False, 'message': 'Prediction not found.'}), 404
                abort(404)

            if pred['user_id'] != session['user_id'] and session.get('role') != 'admin':
                conn.close()
                if is_json:
                    return jsonify({'success': False, 'message': 'Unauthorized to give feedback on this prediction.'}), 403
                abort(403)

            # Check if feedback record already exists
            cursor.execute(
                "SELECT id FROM prediction_feedback WHERE prediction_id = %s AND user_id = %s",
                (prediction_id, session['user_id'])
            )
            existing = cursor.fetchone()

            if existing:
                cursor.execute(
                    """
                    UPDATE prediction_feedback
                    SET feedback_type = %s, user_notes = %s, created_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                    """,
                    (feedback_type, comment, existing['id'])
                )
            else:
                cursor.execute(
                    """
                    INSERT INTO prediction_feedback (prediction_id, user_id, feedback_type, user_notes)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (prediction_id, session['user_id'], feedback_type, comment)
                )

            # Update prediction's user_feedback status flag
            cursor.execute(
                "UPDATE predictions SET user_feedback = %s WHERE id = %s",
                (user_feedback_status, prediction_id)
            )

        conn.close()

        if is_json:
            return jsonify({
                'success': True,
                'message': 'Feedback recorded successfully.',
                'feedback_type': feedback_type
            })

        flash("Thank you for your feedback! It helps improve model reliability.", "success")
        return redirect(url_for('prediction_detail', prediction_id=prediction_id))

    except HTTPException:
        raise
    except Exception as e:
        if is_json:
            return jsonify({'success': False, 'message': f"Database error: {str(e)}"}), 500
        flash("Could not save feedback. Please try again.", "danger")
        return redirect(url_for('prediction_detail', prediction_id=prediction_id))


# =====================================================================
# Protected Admin Route
# =====================================================================
@app.route('/admin')
@admin_required
def admin():
    """
    Protected administrator dashboard:
    - Strictly requires session['role'] == 'admin'.
    - Displays genuine platform metrics, category distribution, recent predictions, and feedback.
    - Zero fake statistics or mock figures.
    """
    users = []
    total_predictions = 0
    total_feedback = 0
    category_stats = []
    recent_predictions = []
    recent_feedback = []

    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            # 1. Registered Users
            cursor.execute(
                "SELECT id, name, email, role, created_at FROM users ORDER BY created_at DESC"
            )
            users = cursor.fetchall()

            # 2. Total platform predictions
            cursor.execute("SELECT COUNT(*) AS total FROM predictions")
            total_predictions = cursor.fetchone()['total']

            # 3. Total feedback entries
            cursor.execute("SELECT COUNT(*) AS total FROM prediction_feedback")
            total_feedback = cursor.fetchone()['total']

            # 4. Platform-wide category distribution
            cursor.execute(
                """
                SELECT w.id, w.category_name, w.bin_name, w.bin_color_code, COUNT(p.id) AS count
                FROM waste_categories w
                LEFT JOIN predictions p ON w.id = p.predicted_category_id
                GROUP BY w.id, w.category_name, w.bin_name, w.bin_color_code
                ORDER BY w.id
                """
            )
            category_stats = cursor.fetchall()

            # 5. Recent 10 platform-wide predictions
            cursor.execute(
                """
                SELECT p.id, p.image_filename, p.confidence_score, p.created_at,
                       u.name AS user_name, w.category_name, w.bin_name, w.bin_color_code
                FROM predictions p
                LEFT JOIN users u ON p.user_id = u.id
                JOIN waste_categories w ON p.predicted_category_id = w.id
                ORDER BY p.created_at DESC
                LIMIT 10
                """
            )
            recent_predictions = cursor.fetchall()

            # 6. Recent 10 feedback entries
            cursor.execute(
                """
                SELECT pf.id, pf.prediction_id, pf.feedback_type, pf.user_notes, pf.created_at,
                       u.name AS user_name, w.category_name
                FROM prediction_feedback pf
                LEFT JOIN users u ON pf.user_id = u.id
                JOIN predictions p ON pf.prediction_id = p.id
                JOIN waste_categories w ON p.predicted_category_id = w.id
                ORDER BY pf.created_at DESC
                LIMIT 10
                """
            )
            recent_feedback = cursor.fetchall()

        conn.close()
    except Exception:
        pass

    return render_template(
        'admin_dashboard.html',
        users=users,
        total_predictions=total_predictions,
        total_feedback=total_feedback,
        category_stats=category_stats,
        recent_predictions=recent_predictions,
        recent_feedback=recent_feedback,
        db_name=Config.DB_NAME,
        model_info=get_model_info()
    )


# =====================================================================
# AI Engine & API Endpoints (Phase 3)
# =====================================================================
@app.route('/api/categories', methods=['GET'])
def get_categories():
    """Returns all 4 waste categories with bin rules and disposal guidance."""
    return jsonify({
        'success': True,
        'categories': get_all_categories()
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    AI Waste Classification API Endpoint:
    - Requires authentication (logged-in user session).
    - Accepts multipart image upload ('image' or 'file').
    - Calls ai_engine.predict_waste.
    - If model is untrained: returns transparent 'model_not_trained' status without fake data.
    - If model is trained: persists prediction to MySQL and saves uploaded image.
    """
    # Enforce login requirement for waste analysis
    if 'user_id' not in session:
        return jsonify({
            'success': False,
            'status': 'unauthorized',
            'message': 'Authentication required. Please log in to analyze waste.'
        }), 401

    # Check if image file was uploaded
    image_file = request.files.get('image') or request.files.get('file')
    if not image_file or image_file.filename == '':
        return jsonify({
            'success': False,
            'status': 'no_file',
            'message': 'No image file was provided in the request.'
        }), 400

    # Execute classification pipeline
    result = predict_waste(image_file)

    # If model is not trained yet, return the honest status response
    if not result.get('is_trained', False):
        return jsonify(result), 200

    # If image failed validation or preprocessing
    if not result.get('success', False):
        return jsonify(result), 400

    # If model is trained and inference succeeded:
    # Save the uploaded file with a UUID
    original_filename = secure_filename(image_file.filename) or 'upload.jpg'
    file_ext = os.path.splitext(original_filename)[1].lower() or '.jpg'
    saved_filename = f"{uuid.uuid4().hex}{file_ext}"
    save_path = os.path.join(Config.UPLOAD_FOLDER, saved_filename)

    try:
        image_file.seek(0)
        image_file.save(save_path)
        result['image_filename'] = saved_filename
        result['image_url'] = url_for('static', filename=f'uploads/{saved_filename}')
    except Exception as e:
        result['image_url'] = None

    # Persist prediction in MySQL database
    user_id = session.get('user_id')  # NULL if guest user
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO predictions (
                    user_id, image_filename, predicted_category_id,
                    confidence_score, model_version, user_feedback
                ) VALUES (%s, %s, %s, %s, %s, 'unreviewed')
                """,
                (
                    user_id,
                    saved_filename,
                    result['category_id'],
                    result['confidence_percentage'],
                    'MobileNetV2-v1'
                )
            )
            result['prediction_id'] = cursor.lastrowid
        conn.close()
    except Exception as e:
        result['db_warning'] = f"Prediction could not be logged to database: {str(e)}"
        result['prediction_id'] = None

    return jsonify(result), 200


@app.route('/api/health', methods=['GET'])
def health_check():
    """API health check endpoint returning system, DB, and AI model readiness."""
    db_status = test_connection()
    model_info = get_model_info()

    return jsonify({
        'status': 'healthy',
        'project': 'EcoSort AI',
        'phase': 7,
        'database': {
            'target_db': Config.DB_NAME,
            'connected': db_status['connected'],
            'database_exists': db_status['database_exists'],
            'tables': db_status['tables'],
            'categories_seeded': db_status['categories_count'],
            'message': db_status['message']
        },
        'ai_engine': model_info,
        'upload_folder': {
            'path': Config.UPLOAD_FOLDER,
            'exists': os.path.exists(Config.UPLOAD_FOLDER)
        }
    })


@app.errorhandler(413)
def request_entity_too_large(error):
    """Graceful handler for files exceeding the 5MB upload limit."""
    if request.path.startswith('/api/'):
        return jsonify({
            'success': False,
            'status': 'file_too_large',
            'message': 'Uploaded file exceeds the maximum 5MB size limit.'
        }), 413
    flash("The uploaded image exceeds the 5MB size limit. Please choose a smaller image.", "danger")
    return redirect(request.url), 413


if __name__ == '__main__':
    print("Starting EcoSort AI Flask server (Phase 7 - Production Ready)...")
    print(f"Target Database: {Config.DB_NAME}")
    print(f"Model Path:      {Config.MODEL_PATH}")
    print(f"Model Available: {is_model_available()}")
    app.run(host='0.0.0.0', port=5000, debug=Config.DEBUG)

