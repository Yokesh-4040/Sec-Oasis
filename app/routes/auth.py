from flask import Blueprint, request, jsonify
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash
from ..models import db
from ..models.employee import Employee

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    
    if not data or not data.get('email') or not data.get('password'):
        return jsonify({'error': 'Missing email or password'}), 400
        
    employee = Employee.query.filter_by(email=data['email']).first()
    
    if not employee or not employee.check_password(data['password']):
        return jsonify({'error': 'Invalid email or password'}), 401
        
    if not employee.is_active:
        return jsonify({'error': 'Account is disabled'}), 401
        
    login_user(employee)
    return jsonify({
        'message': 'Logged in successfully',
        'user': employee.to_dict()
    })

@auth_bp.route('/api/auth/logout')
@login_required
def logout():
    logout_user()
    return jsonify({'message': 'Logged out successfully'})

@auth_bp.route('/api/auth/register', methods=['POST'])
@login_required
def register():
    if not current_user.is_admin:
        return jsonify({'error': 'Unauthorized'}), 403
        
    data = request.get_json()
    
    required_fields = ['email', 'name', 'password', 'position', 'department']
    if not all(field in data for field in required_fields):
        return jsonify({'error': 'Missing required fields'}), 400
        
    if Employee.query.filter_by(email=data['email']).first():
        return jsonify({'error': 'Email already registered'}), 400
        
    new_employee = Employee(
        email=data['email'],
        name=data['name'],
        position=data['position'],
        department=data['department'],
        is_admin=data.get('is_admin', False)
    )
    new_employee.set_password(data['password'])
    
    db.session.add(new_employee)
    db.session.commit()
    
    return jsonify({
        'message': 'Employee registered successfully',
        'user': new_employee.to_dict()
    }), 201

@auth_bp.route('/api/auth/me')
@login_required
def get_current_user():
    return jsonify(current_user.to_dict())

@auth_bp.route('/api/auth/face-login', methods=['POST'])
def face_login():
    """Handle face recognition login"""
    data = request.get_json()
    employee_name = data.get('employee_name')
    face_data = data.get('face_data')
    
    if not employee_name:
        return jsonify({'success': False, 'message': 'Employee name required'}), 400
    
    # Find employee by name (in real implementation, this would use face encoding comparison)
    employee = Employee.query.filter_by(name=employee_name).first()
    
    if not employee:
        return jsonify({'success': False, 'message': 'Employee not found'}), 404
    
    if not employee.is_active:
        return jsonify({'success': False, 'message': 'Account is deactivated'}), 403
    
    # In real implementation, you would:
    # 1. Compare the received face_data with stored facial_id using face_recognition library
    # 2. Use your existing face recognition system (detect_ppe_and_names.py)
    # For now, we'll simulate successful recognition
    
    login_user(employee, remember=True)
    
    return jsonify({
        'success': True,
        'message': f'Welcome back, {employee.name}!',
        'employee': {
            'id': employee.id,
            'name': employee.name,
            'email': employee.email,
            'position': employee.position,
            'is_admin': employee.is_admin
        }
    }) 