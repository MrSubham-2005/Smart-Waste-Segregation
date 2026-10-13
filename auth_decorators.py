"""
EcoSort AI - Authentication Decorators
Provides @login_required and @admin_required route protection.
"""

from functools import wraps
from flask import session, redirect, url_for, flash, request, abort


def login_required(f):
    """
    Decorator to ensure user is logged in before accessing a route.
    Redirects unauthenticated users to /login with a friendly message.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login', next=request.path))
        return f(*args, **kwargs)
    return decorated_function


def admin_required(f):
    """
    Decorator to ensure logged-in user has the 'admin' role.
    - If not logged in: Redirects to /login.
    - If logged in as normal user: Returns 403 Forbidden.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in as an administrator to access this page.', 'warning')
            return redirect(url_for('login', next=request.path))
        if session.get('role') != 'admin':
            abort(403)
        return f(*args, **kwargs)
    return decorated_function
