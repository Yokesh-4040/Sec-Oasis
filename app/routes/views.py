from flask import Blueprint, render_template, redirect, url_for, request, flash
from flask_login import login_required, current_user

views_bp = Blueprint('views', __name__)

@views_bp.route('/')
def index():
    if current_user.is_authenticated:
        return redirect('/dashboard')
    return redirect('/login')

@views_bp.route('/login')
def login():
    if current_user.is_authenticated:
        return redirect('/dashboard')
    return render_template('auth/login.html')

@views_bp.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@views_bp.route('/employees')
@login_required
def employees():
    if not current_user.is_admin:
        return redirect('/dashboard')
    return render_template('employees/list.html')

@views_bp.route('/attendance')
@login_required
def attendance():
    return render_template('attendance/logs.html')

@views_bp.route('/reports')
@login_required
def reports():
    return render_template('reports/dashboard.html')

@views_bp.route('/profile')
@login_required
def profile():
    return render_template('profile.html')

@views_bp.route('/logout')
@login_required
def logout():
    return redirect('/api/auth/logout')

@views_bp.route('/onboard-employee')
@login_required
def onboard_employee():
    """Gamified employee onboarding page"""
    if not current_user.is_admin:
        flash('Access denied. Administrator privileges required.', 'error')
        return redirect(url_for('views.dashboard'))
    
    return render_template('onboard_employee.html') 