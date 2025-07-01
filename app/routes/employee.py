from flask import Blueprint, request, jsonify
from flask_login import login_required, current_user
from ..models import db
from ..models.employee import Employee
import os
import base64
import uuid

employee_bp = Blueprint('employee', __name__)

@employee_bp.route('/api/employees', methods=['GET'])
def get_employees():
    employees = Employee.query.all()
    return jsonify({'employees': [emp.to_dict() for emp in employees]})

@employee_bp.route('/api/employees', methods=['POST'])
@login_required
def create_employee():
    """Create a new employee with optional face recognition data"""
    if not current_user.is_admin:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    
    data = request.get_json()
    
    # Validate required fields
    required_fields = ['name', 'email', 'position', 'department', 'password']
    for field in required_fields:
        if not data.get(field):
            return jsonify({'success': False, 'message': f'Missing required field: {field}'}), 400
    
    # Check if email already exists
    existing_employee = Employee.query.filter_by(email=data['email']).first()
    if existing_employee:
        return jsonify({'success': False, 'message': 'Email already exists'}), 400
    
    try:
        # Create new employee
        employee = Employee(
            name=data['name'],
            email=data['email'],
            position=data['position'],
            department=data['department'],
            is_admin=data.get('is_admin', False)
        )
        employee.set_password(data['password'])
        
        # Handle face data if provided
        face_data = data.get('faceData')
        if face_data and face_data.startswith('data:image'):
            try:
                # Extract base64 image data
                header, encoded = face_data.split(',', 1)
                image_data = base64.b64decode(encoded)
                
                # Create dataset directory if it doesn't exist
                dataset_dir = 'dataset'
                os.makedirs(dataset_dir, exist_ok=True)
                
                # Save image to dataset folder
                filename = f"{employee.name}.jpg"
                filepath = os.path.join(dataset_dir, filename)
                
                with open(filepath, 'wb') as f:
                    f.write(image_data)
                
                # Generate facial ID
                employee.facial_id = str(uuid.uuid4())
                
            except Exception as e:
                print(f"Error saving face image: {e}")
                # Continue without face data if there's an error
                
        db.session.add(employee)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Employee created successfully',
            'employee': employee.to_dict()
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)}), 500

@employee_bp.route('/api/employees/<int:id>', methods=['GET'])
@login_required
def get_employee(id):
    if not current_user.is_admin and current_user.id != id:
        return jsonify({'error': 'Unauthorized'}), 403
        
    employee = Employee.query.get_or_404(id)
    return jsonify(employee.to_dict())

@employee_bp.route('/api/employees/<int:id>', methods=['PUT'])
@login_required
def update_employee(id):
    if not current_user.is_admin:
        return jsonify({'error': 'Unauthorized'}), 403
        
    employee = Employee.query.get_or_404(id)
    data = request.get_json()
    
    if 'name' in data:
        employee.name = data['name']
    if 'position' in data:
        employee.position = data['position']
    if 'department' in data:
        employee.department = data['department']
    if 'is_active' in data:
        employee.is_active = data['is_active']
    if 'is_admin' in data:
        employee.is_admin = data['is_admin']
    if 'password' in data:
        employee.set_password(data['password'])
        
    db.session.commit()
    return jsonify(employee.to_dict())

@employee_bp.route('/api/employees/<int:id>', methods=['DELETE'])
@login_required
def delete_employee(id):
    if not current_user.is_admin:
        return jsonify({'error': 'Unauthorized'}), 403
        
    employee = Employee.query.get_or_404(id)
    
    if employee.is_admin:
        return jsonify({'error': 'Cannot delete admin user'}), 400
        
    db.session.delete(employee)
    db.session.commit()
    return jsonify({'message': 'Employee deleted successfully'})

@employee_bp.route('/api/employees/<int:id>/facial-id', methods=['POST'])
@login_required
def update_facial_id(id):
    if not current_user.is_admin and current_user.id != id:
        return jsonify({'error': 'Unauthorized'}), 403
        
    employee = Employee.query.get_or_404(id)
    data = request.get_json()
    
    if not data or 'facial_id' not in data:
        return jsonify({'error': 'Missing facial_id'}), 400
        
    employee.facial_id = data['facial_id']
    db.session.commit()
    
    return jsonify({
        'message': 'Facial ID updated successfully',
        'user': employee.to_dict()
    })

@employee_bp.route('/api/employees/<int:employee_id>/setup-face', methods=['POST'])
@login_required
def setup_face_recognition(employee_id):
    """Setup face recognition for an employee"""
    if not current_user.is_admin and current_user.id != employee_id:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    
    employee = Employee.query.get_or_404(employee_id)
    data = request.get_json()
    face_data = data.get('face_data')
    
    if not face_data:
        return jsonify({'success': False, 'message': 'Face data required'}), 400
    
    try:
        # In real implementation, you would:
        # 1. Process the face_data using your face recognition system
        # 2. Save the face encoding to the dataset folder as {employee.name}.jpg
        # 3. Generate a unique facial_id
        
        # For now, simulate the process
        import os
        import uuid
        
        # Generate unique facial ID
        facial_id = str(uuid.uuid4())
        
        # In real implementation, save face image to dataset folder
        dataset_path = os.path.join('dataset', f'{employee.name}.jpg')
        # Here you would save the actual face image
        
        # Update employee record
        employee.facial_id = facial_id
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Face recognition setup completed successfully',
            'facial_id': facial_id
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)}), 500

@employee_bp.route('/api/employees/capture-from-camera', methods=['POST'])
@login_required
def capture_from_rtsp_camera():
    """Capture face image from RTSP camera for employee onboarding"""
    if not current_user.is_admin:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    
    try:
        import cv2
        import face_recognition
        import sys
        import os
        
        # Add parent directory to path to import camera_config
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
        from camera_config import get_rtsp_url, CAMERA_CONFIG
        
        # Get RTSP URL
        rtsp_url = get_rtsp_url()
        
        # Connect to camera
        cap = cv2.VideoCapture(rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            return jsonify({'success': False, 'message': 'Could not connect to camera'}), 500
        
        # Capture frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return jsonify({'success': False, 'message': 'Could not capture frame from camera'}), 500
        
        # Convert frame to RGB for face detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        
        if not face_locations:
            return jsonify({'success': False, 'message': 'No face detected in camera frame'}), 400
        
        # Get the largest face (assuming it's the person being onboarded)
        largest_face = max(face_locations, key=lambda x: (x[2] - x[0]) * (x[1] - x[3]))
        top, right, bottom, left = largest_face
        
        # Add some padding around the face
        padding = 50
        height, width = rgb_frame.shape[:2]
        top = max(0, top - padding)
        left = max(0, left - padding)
        bottom = min(height, bottom + padding)
        right = min(width, right + padding)
        
        # Crop face from frame
        face_image = rgb_frame[top:bottom, left:right]
        
        # Convert back to BGR for OpenCV
        face_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
        
        # Encode image as JPEG
        _, buffer = cv2.imencode('.jpg', face_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        
        # Convert to base64
        import base64
        face_base64 = base64.b64encode(buffer).decode('utf-8')
        face_data_url = f"data:image/jpeg;base64,{face_base64}"
        
        return jsonify({
            'success': True,
            'message': 'Face captured successfully from camera',
            'face_data': face_data_url,
            'face_location': {
                'top': int(top), 'right': int(right), 
                'bottom': int(bottom), 'left': int(left)
            }
        })
        
    except ImportError as e:
        return jsonify({'success': False, 'message': f'Missing required packages: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error capturing from camera: {str(e)}'}), 500 