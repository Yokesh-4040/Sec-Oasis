from flask import Blueprint, request, jsonify
from flask_login import login_required, current_user
from datetime import datetime, date
from ..models import db
from ..models.employee import Employee
from ..models.attendance import AttendanceLog

attendance_bp = Blueprint('attendance', __name__)

@attendance_bp.route('/api/attendance/check-in', methods=['POST'])
def check_in():
    data = request.get_json()
    
    if not data or 'facial_id' not in data:
        return jsonify({'error': 'Missing facial_id'}), 400
        
    employee = Employee.query.filter_by(facial_id=data['facial_id']).first()
    
    if not employee:
        return jsonify({'error': 'Employee not found'}), 404
        
    if not employee.is_active:
        return jsonify({'error': 'Employee account is inactive'}), 400
        
    # Check if already checked in today
    today = date.today()
    existing_log = AttendanceLog.query.filter_by(
        employee_id=employee.id,
        date=today
    ).first()
    
    if existing_log and not existing_log.check_out:
        return jsonify({'error': 'Already checked in'}), 400
        
    # Create new attendance log
    now = datetime.now()
    attendance = AttendanceLog(
        employee_id=employee.id,
        check_in=now,
        date=today
    )
    
    db.session.add(attendance)
    db.session.commit()
    
    return jsonify({
        'message': 'Check-in successful',
        'attendance': attendance.to_dict()
    })

@attendance_bp.route('/api/attendance/check-out', methods=['POST'])
def check_out():
    data = request.get_json()
    
    if not data or 'facial_id' not in data:
        return jsonify({'error': 'Missing facial_id'}), 400
        
    employee = Employee.query.filter_by(facial_id=data['facial_id']).first()
    
    if not employee:
        return jsonify({'error': 'Employee not found'}), 404
        
    # Find today's active check-in
    today = date.today()
    attendance = AttendanceLog.query.filter_by(
        employee_id=employee.id,
        date=today,
        check_out=None
    ).first()
    
    if not attendance:
        return jsonify({'error': 'No active check-in found'}), 400
        
    attendance.check_out = datetime.now()
    db.session.commit()
    
    return jsonify({
        'message': 'Check-out successful',
        'attendance': attendance.to_dict()
    })

@attendance_bp.route('/api/attendance/logs', methods=['GET'])
def get_attendance_logs():
    # Parse query parameters
    employee_id = request.args.get('employee_id', type=int)
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    limit = request.args.get('limit', type=int)
    
    # Base query with employee join for names
    query = db.session.query(AttendanceLog, Employee).join(Employee, AttendanceLog.employee_id == Employee.id)
    
    # Apply filters
    if employee_id:
        query = query.filter(AttendanceLog.employee_id == employee_id)
        
    if start_date:
        query = query.filter(AttendanceLog.date >= start_date)
    if end_date:
        query = query.filter(AttendanceLog.date <= end_date)
        
    # Execute query and format results (order_by must come before limit)
    query = query.order_by(AttendanceLog.created_at.desc())
    
    # Apply limit after ordering
    if limit:
        query = query.limit(limit)
        
    results = query.all()
    
    logs = []
    for attendance_log, employee in results:
        # Create separate entries for check-in and check-out
        if attendance_log.check_in:
            logs.append({
                'id': f"{attendance_log.id}_in",
                'employee_id': employee.id,
                'employee_name': employee.name,
                'action': 'check_in',
                'created_at': attendance_log.check_in.isoformat(),
                'timestamp': attendance_log.check_in.isoformat(),
                'date': attendance_log.date.isoformat(),
                'status': attendance_log.status,
                'notes': attendance_log.notes
            })
        
        if attendance_log.check_out:
            logs.append({
                'id': f"{attendance_log.id}_out",
                'employee_id': employee.id,
                'employee_name': employee.name,
                'action': 'check_out', 
                'created_at': attendance_log.check_out.isoformat(),
                'timestamp': attendance_log.check_out.isoformat(),
                'date': attendance_log.date.isoformat(),
                'status': attendance_log.status,
                'notes': attendance_log.notes
            })
    
    # Sort by timestamp descending
    logs.sort(key=lambda x: x['timestamp'], reverse=True)
    
    if limit:
        logs = logs[:limit]
    
    return jsonify({'logs': logs})

@attendance_bp.route('/api/attendance/logs/<int:id>', methods=['PUT'])
@login_required
def update_attendance_log(id):
    if not current_user.is_admin:
        return jsonify({'error': 'Unauthorized'}), 403
        
    attendance = AttendanceLog.query.get_or_404(id)
    data = request.get_json()
    
    if 'check_in' in data:
        attendance.check_in = datetime.fromisoformat(data['check_in'])
    if 'check_out' in data:
        attendance.check_out = datetime.fromisoformat(data['check_out']) if data['check_out'] else None
    if 'status' in data:
        attendance.status = data['status']
    if 'notes' in data:
        attendance.notes = data['notes']
        
    db.session.commit()
    return jsonify(attendance.to_dict())

@attendance_bp.route('/api/attendance/face-checkin', methods=['POST'])
@login_required
def face_checkin():
    """Handle face recognition check-in using existing face recognition system"""
    data = request.get_json()
    face_data = data.get('face_data')  # This would be actual face encoding from camera
    
    if not face_data:
        return jsonify({'success': False, 'message': 'Face data required'}), 400
    
    try:
        # In real implementation, you would:
        # 1. Use your existing face recognition system (detect_ppe_and_names.py)
        # 2. Load known faces from dataset folder
        # 3. Compare face_data with known encodings
        # 4. Find matching employee
        
        # Simulate face recognition process
        import face_recognition
        import os
        from datetime import datetime
        
        # Load known faces (simulate using your existing system)
        dataset_dir = "dataset"
        known_encodings = []
        known_names = []
        
        if os.path.exists(dataset_dir):
            for filename in os.listdir(dataset_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # In real implementation, load and encode the face
                    # For now, just get the name from filename
                    name = os.path.splitext(filename)[0]
                    known_names.append(name)
        
        # Simulate recognition (in real implementation, use face_recognition.compare_faces)
        if known_names:
            recognized_name = known_names[0]  # Simulate first match
            
            # Find employee by name
            employee = Employee.query.filter_by(name=recognized_name).first()
            if not employee:
                return jsonify({'success': False, 'message': 'Employee not found in database'}), 404
                
            # Check if already checked in today
            today = datetime.now().date()
            existing_checkin = AttendanceLog.query.filter_by(
                employee_id=employee.id,
                action='check_in'
            ).filter(AttendanceLog.timestamp >= today).first()
            
            if existing_checkin:
                return jsonify({'success': False, 'message': 'Already checked in today'}), 400
            
            # Create attendance log
            log = AttendanceLog(
                employee_id=employee.id,
                action='check_in',
                facial_recognition=True
            )
            db.session.add(log)
            db.session.commit()
            
            return jsonify({
                'success': True,
                'message': f'Welcome, {employee.name}! Check-in successful.',
                'employee': {
                    'id': employee.id,
                    'name': employee.name,
                    'position': employee.position
                },
                'timestamp': log.timestamp.isoformat()
            })
        else:
            return jsonify({'success': False, 'message': 'Face not recognized'}), 404
            
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)}), 500

@attendance_bp.route('/api/attendance/face-checkout', methods=['POST'])
@login_required
def face_checkout():
    """Handle face recognition check-out"""
    data = request.get_json()
    face_data = data.get('face_data')
    
    if not face_data:
        return jsonify({'success': False, 'message': 'Face data required'}), 400
    
    try:
        # Similar process as face_checkin but for check-out
        # Simulate face recognition
        dataset_dir = "dataset"
        known_names = []
        
        if os.path.exists(dataset_dir):
            for filename in os.listdir(dataset_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    name = os.path.splitext(filename)[0]
                    known_names.append(name)
        
        if known_names:
            recognized_name = known_names[0]  # Simulate recognition
            
            employee = Employee.query.filter_by(name=recognized_name).first()
            if not employee:
                return jsonify({'success': False, 'message': 'Employee not found'}), 404
            
            # Check if checked in today
            today = datetime.now().date()
            checkin_log = AttendanceLog.query.filter_by(
                employee_id=employee.id,
                action='check_in'
            ).filter(AttendanceLog.timestamp >= today).first()
            
            if not checkin_log:
                return jsonify({'success': False, 'message': 'No check-in found for today'}), 400
            
            # Check if already checked out
            checkout_log = AttendanceLog.query.filter_by(
                employee_id=employee.id,
                action='check_out'
            ).filter(AttendanceLog.timestamp >= today).first()
            
            if checkout_log:
                return jsonify({'success': False, 'message': 'Already checked out today'}), 400
            
            # Create checkout log
            log = AttendanceLog(
                employee_id=employee.id,
                action='check_out',
                facial_recognition=True
            )
            db.session.add(log)
            db.session.commit()
            
            return jsonify({
                'success': True,
                'message': f'Goodbye, {employee.name}! Check-out successful.',
                'employee': {
                    'id': employee.id,
                    'name': employee.name,
                    'position': employee.position
                },
                'timestamp': log.timestamp.isoformat()
            })
        else:
            return jsonify({'success': False, 'message': 'Face not recognized'}), 404
            
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)}), 500

@attendance_bp.route('/api/attendance/camera-feed', methods=['GET'])
@login_required
def get_camera_feed():
    """Get current frame from RTSP camera for live feed"""
    try:
        import cv2
        import sys
        import os
        import base64
        
        # Add parent directory to path to import camera_config
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
        from camera_config import get_rtsp_url
        
        # Get RTSP URL
        rtsp_url = get_rtsp_url()
        
        # Connect to camera
        cap = cv2.VideoCapture(rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            return jsonify({'success': False, 'message': 'Camera not available'}), 500
        
        # Capture frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return jsonify({'success': False, 'message': 'Failed to capture frame'}), 500
        
        # Resize frame for web display (optional)
        height, width = frame.shape[:2]
        if width > 640:
            new_width = 640
            new_height = int(height * (new_width / width))
            frame = cv2.resize(frame, (new_width, new_height))
        
        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        
        # Convert to base64
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        frame_data_url = f"data:image/jpeg;base64,{frame_base64}"
        
        return jsonify({
            'success': True,
            'frame': frame_data_url,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@attendance_bp.route('/api/attendance/rtsp-checkin', methods=['POST'])
@login_required
def rtsp_face_checkin():
    """Enhanced face recognition check-in using RTSP camera"""
    try:
        import cv2
        import face_recognition
        import sys
        import os
        
        # Add parent directory to path to import camera_config
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
        from camera_config import get_rtsp_url
        
        # Get RTSP URL and capture frame
        rtsp_url = get_rtsp_url()
        cap = cv2.VideoCapture(rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            return jsonify({'success': False, 'message': 'Camera not available'}), 500
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return jsonify({'success': False, 'message': 'Failed to capture from camera'}), 500
        
        # Convert to RGB for face recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find faces in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        if not face_encodings:
            return jsonify({'success': False, 'message': 'No face detected in camera frame'}), 400
        
        # Load known employee faces from dataset
        dataset_dir = "dataset"
        known_encodings = []
        known_names = []
        
        for filename in os.listdir(dataset_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                face_path = os.path.join(dataset_dir, filename)
                image = face_recognition.load_image_file(face_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    known_encodings.append(encodings[0])
                    known_names.append(os.path.splitext(filename)[0])
        
        # Compare detected faces with known faces
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
            if True in matches:
                match_index = matches.index(True)
                recognized_name = known_names[match_index]
                
                # Find employee in database
                employee = Employee.query.filter_by(name=recognized_name).first()
                if not employee:
                    continue
                
                # Check if already checked in today
                today = datetime.now().date()
                existing_checkin = AttendanceLog.query.filter_by(
                    employee_id=employee.id,
                    action='check_in'
                ).filter(AttendanceLog.timestamp >= today).first()
                
                if existing_checkin:
                    return jsonify({
                        'success': False, 
                        'message': f'{employee.name} is already checked in today',
                        'employee': employee.to_dict()
                    }), 400
                
                # Create check-in log
                log = AttendanceLog(
                    employee_id=employee.id,
                    action='check_in',
                    facial_recognition=True
                )
                db.session.add(log)
                db.session.commit()
                
                return jsonify({
                    'success': True,
                    'message': f'Welcome, {employee.name}! Check-in successful.',
                    'employee': employee.to_dict(),
                    'timestamp': log.timestamp.isoformat()
                })
        
        # No matches found
        return jsonify({'success': False, 'message': 'Face not recognized. Please contact admin.'}), 404
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)}), 500

@attendance_bp.route('/api/attendance/rtsp-checkout', methods=['POST'])
@login_required
def rtsp_face_checkout():
    """Enhanced face recognition check-out using RTSP camera"""
    try:
        import cv2
        import face_recognition
        import sys
        import os
        
        # Add parent directory to path to import camera_config
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
        from camera_config import get_rtsp_url
        
        # Get RTSP URL and capture frame
        rtsp_url = get_rtsp_url()
        cap = cv2.VideoCapture(rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            return jsonify({'success': False, 'message': 'Camera not available'}), 500
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return jsonify({'success': False, 'message': 'Failed to capture from camera'}), 500
        
        # Convert to RGB for face recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find faces in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        if not face_encodings:
            return jsonify({'success': False, 'message': 'No face detected in camera frame'}), 400
        
        # Load known employee faces from dataset
        dataset_dir = "dataset"
        known_encodings = []
        known_names = []
        
        for filename in os.listdir(dataset_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                face_path = os.path.join(dataset_dir, filename)
                image = face_recognition.load_image_file(face_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    known_encodings.append(encodings[0])
                    known_names.append(os.path.splitext(filename)[0])
        
        # Compare detected faces with known faces
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
            if True in matches:
                match_index = matches.index(True)
                recognized_name = known_names[match_index]
                
                # Find employee in database
                employee = Employee.query.filter_by(name=recognized_name).first()
                if not employee:
                    continue
                
                # Check if checked in today
                today = datetime.now().date()
                checkin_log = AttendanceLog.query.filter_by(
                    employee_id=employee.id,
                    action='check_in'
                ).filter(AttendanceLog.timestamp >= today).first()
                
                if not checkin_log:
                    return jsonify({
                        'success': False, 
                        'message': f'{employee.name} has not checked in today',
                        'employee': employee.to_dict()
                    }), 400
                
                # Check if already checked out
                checkout_log = AttendanceLog.query.filter_by(
                    employee_id=employee.id,
                    action='check_out'
                ).filter(AttendanceLog.timestamp >= today).first()
                
                if checkout_log:
                    return jsonify({
                        'success': False, 
                        'message': f'{employee.name} is already checked out today',
                        'employee': employee.to_dict()
                    }), 400
                
                # Create check-out log
                log = AttendanceLog(
                    employee_id=employee.id,
                    action='check_out',
                    facial_recognition=True
                )
                db.session.add(log)
                db.session.commit()
                
                return jsonify({
                    'success': True,
                    'message': f'Goodbye, {employee.name}! Check-out successful.',
                    'employee': employee.to_dict(),
                    'timestamp': log.timestamp.isoformat()
                })
        
        # No matches found
        return jsonify({'success': False, 'message': 'Face not recognized. Please contact admin.'}), 404
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)}), 500 