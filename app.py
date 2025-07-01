from flask import Flask, render_template, jsonify, Response, request, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import shutil
import json
import threading
import time
import queue
from datetime import datetime
import os
import csv
from ultralytics import YOLO
import face_recognition
from camera_config import *
from onvif import ONVIFCamera

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ppe_detection_secret'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

class PPEDetectionServer:
    def __init__(self):
        print("[INFO] Initializing PPE Detection Server...")
        
        # Load YOLO model
        self.model = YOLO(PATHS["model_path"])
        print(f"[INFO] YOLO model loaded from {PATHS['model_path']}")
        
        # Load employee faces
        self.known_encodings = []
        self.known_names = []
        self._load_employee_faces()
        
        # Detection settings
        self.required_ppe = set(DETECTION_CONFIG["required_ppe"])
        self.output_dir = PATHS["output_directory"]
        os.makedirs(self.output_dir, exist_ok=True)
        
        # System state
        self.running = False
        self.current_frame = None
        self.detection_results = {
            'people': [],
            'ppe_items': [],
            'compliance_status': 'UNKNOWN',
            'stats': {'total_people': 0, 'compliant_people': 0}
        }
        self.unknown_people = []  # Store unknown person detections
        self.system_stats = {
            'frames_processed': 0,
            'start_time': None,
            'fps': 0,
            'connection_status': 'Disconnected'
        }
        
        # Threading
        self.frame_queue = queue.Queue(maxsize=10)
        self.capture_thread = None
        self.processing_thread = None
        
        print("[INFO] PPE Detection Server initialized")
        
        # Initialize camera control
        self.onvif_camera = None
        self._init_camera_control()
    
    def _init_camera_control(self):
        """Initialize ONVIF camera control"""
        try:
            self.onvif_camera = ONVIFCamera(
                CAMERA_CONFIG['ip'], 
                CAMERA_CONFIG.get('onvif_port', 2020),
                CAMERA_CONFIG['username'], 
                CAMERA_CONFIG['password']
            )
            # Create media and PTZ services
            self.media_service = self.onvif_camera.create_media_service()
            self.ptz_service = self.onvif_camera.create_ptz_service()
            
            # Get profiles
            profiles = self.media_service.GetProfiles()
            self.profile_token = profiles[0].token if profiles else None
            
            print("[INFO] ONVIF camera control initialized")
        except Exception as e:
            print(f"[WARNING] Could not initialize camera control: {e}")
            self.onvif_camera = None
    
    def _load_employee_faces(self):
        """Load known employee faces for recognition"""
        dataset_dir = PATHS["employee_faces"]
        if not os.path.exists(dataset_dir):
            print(f"[WARNING] Employee faces directory not found: {dataset_dir}")
            return
        
        for filename in os.listdir(dataset_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(dataset_dir, filename)
                try:
                    image = face_recognition.load_image_file(filepath)
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        self.known_encodings.append(encodings[0])
                        name = os.path.splitext(filename)[0]
                        self.known_names.append(name)
                        print(f"[INFO] Loaded face encoding for: {name}")
                except Exception as e:
                    print(f"[WARNING] Could not load {filename}: {e}")
    
    def _capture_frames(self):
        """Capture frames from RTSP stream"""
        rtsp_url = f"rtsp://{CAMERA_CONFIG['username']}:{CAMERA_CONFIG['password']}@{CAMERA_CONFIG['ip']}:{CAMERA_CONFIG['rtsp_port']}/{CAMERA_CONFIG['stream_path']}"
        
        print(f"[INFO] Connecting to RTSP stream: {rtsp_url.replace(CAMERA_CONFIG['password'], '***')}")
        
        # Try different backends
        backends_to_try = [cv2.CAP_FFMPEG, cv2.CAP_ANY]
        cap = None
        
        for backend in backends_to_try:
            cap = cv2.VideoCapture(rtsp_url, backend)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_CONFIG['buffer_size'])
            if cap.isOpened():
                print(f"[INFO] Connected with backend: {backend}")
                break
            cap.release()
        
        if not cap or not cap.isOpened():
            print("[ERROR] Failed to connect to RTSP stream")
            self.system_stats['connection_status'] = 'Failed'
            return
        
        self.system_stats['connection_status'] = 'Connected'
        frame_count = 0
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Failed to read frame")
                continue
            
            frame_count += 1
            if frame_count % CAMERA_CONFIG['frame_skip'] == 0:
                if not self.frame_queue.full():
                    self.frame_queue.put(frame.copy())
                    self.current_frame = frame.copy()
        
        cap.release()
        print("[INFO] Frame capture stopped")
    
    def _process_detections(self, frame):
        """Process PPE and face detection"""
        # Face recognition
        detected_people = []
        if self.known_encodings:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb, model=DETECTION_CONFIG["face_model"])
            face_encodings = face_recognition.face_encodings(rgb, face_locations)
            
            for encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(
                    self.known_encodings, encoding, tolerance=DETECTION_CONFIG["face_tolerance"]
                )
                name = "Unknown Person"
                if True in matches:
                    best_idx = matches.index(True)
                    name = self.known_names[best_idx]
                
                detected_people.append({
                    'name': name,
                    'bbox': [left, top, right, bottom],
                    'confidence': 1.0,
                    'face_encoding': encoding.tolist()  # Store encoding for unknown people
                })
                
                # Handle unknown people (only if enabled)
                if name == "Unknown Person" and DETECTION_CONFIG.get("detect_unknown_people", True):
                    self._handle_unknown_person(frame, [left, top, right, bottom], encoding)
        
        # PPE detection (only if enabled)
        detected_ppe = []
        if DETECTION_CONFIG.get("detect_ppe", True):
            results = self.model.predict(frame, verbose=False, conf=DETECTION_CONFIG["yolo_confidence"])
            boxes = results[0].boxes
            names = self.model.names
            
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    label = names[class_id]
                    
                    detected_ppe.append({
                        'label': label,
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2],
                        'required': label in self.required_ppe
                    })
        
        return detected_people, detected_ppe
    
    def _handle_unknown_person(self, frame, bbox, face_encoding):
        """Handle detection of unknown person"""
        # Check if this unknown person was recently detected (avoid duplicates)
        current_time = datetime.now()
        
        # Compare with recent unknown detections to avoid duplicates
        is_duplicate = False
        cooldown_seconds = DETECTION_CONFIG.get("unknown_detection_cooldown", 30)
        for unknown in self.unknown_people:
            if (current_time - datetime.fromisoformat(unknown['timestamp'])).seconds < cooldown_seconds:
                # Compare face encodings to see if it's the same person
                similarity = face_recognition.compare_faces([unknown['face_encoding']], face_encoding, tolerance=0.6)
                if similarity[0]:
                    is_duplicate = True
                    unknown['last_seen'] = current_time.isoformat()
                    break
        
        if not is_duplicate:
            # Extract face image
            left, top, right, bottom = bbox
            face_img = frame[top:bottom, left:right]
            
            # Create unique filename
            timestamp_str = current_time.strftime("%Y%m%d_%H%M%S")
            unknown_id = f"unknown_{timestamp_str}_{len(self.unknown_people)}"
            filename = f"{unknown_id}.jpg"
            filepath = os.path.join(self.output_dir, "unknown_people", filename)
            
            # Create unknown_people directory if it doesn't exist
            os.makedirs(os.path.join(self.output_dir, "unknown_people"), exist_ok=True)
            
            # Save face image
            cv2.imwrite(filepath, face_img)
            
            # Add to unknown people list
            unknown_person = {
                'id': unknown_id,
                'filename': filename,
                'filepath': filepath,
                'timestamp': current_time.isoformat(),
                'last_seen': current_time.isoformat(),
                'face_encoding': face_encoding.tolist(),
                'bbox': bbox,
                'added_to_database': False
            }
            
            self.unknown_people.append(unknown_person)
            
            # Keep only last 50 unknown people to prevent memory issues
            if len(self.unknown_people) > 50:
                # Remove oldest unknown person file
                oldest = self.unknown_people.pop(0)
                if os.path.exists(oldest['filepath']):
                    os.remove(oldest['filepath'])
            
            print(f"[INFO] Unknown person detected and saved: {unknown_id}")
    
    def _check_compliance(self, detected_ppe):
        """Check PPE compliance"""
        # If PPE detection is disabled, always return compliant
        if not DETECTION_CONFIG.get("detect_ppe", True):
            return {
                'status': "FACE-ONLY MODE",
                'present_ppe': [],
                'missing_ppe': [],
                'compliance_percentage': 100
            }
        
        detected_labels = set(ppe['label'] for ppe in detected_ppe)
        present_ppe = detected_labels.intersection(self.required_ppe)
        missing_ppe = self.required_ppe - present_ppe
        
        status = "COMPLIANT" if not missing_ppe else "NON-COMPLIANT"
        
        return {
            'status': status,
            'present_ppe': list(present_ppe),
            'missing_ppe': list(missing_ppe),
            'compliance_percentage': len(present_ppe) / len(self.required_ppe) * 100 if self.required_ppe else 100
        }
    
    def _process_frames(self):
        """Process frames and emit results"""
        print("[INFO] Starting frame processing thread")
        fps_counter = 0
        fps_start_time = time.time()
        frame_count = 0
        
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1)
                frame_count += 1
                
                # Performance optimization: skip frames based on detection_interval
                detection_interval = CAMERA_CONFIG.get("detection_interval", 1)
                if frame_count % detection_interval != 0:
                    # Skip detection processing, just send the frame
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    frame_b64 = base64.b64encode(buffer).decode('utf-8')
                    
                    socketio.emit('detection_update', {
                        'frame': frame_b64,
                        'detections': self.detection_results,  # Use previous results
                        'stats': self.system_stats,
                        'unknown_people_count': len(self.unknown_people)
                    })
                    continue
                
                # Process detections
                detected_people, detected_ppe = self._process_detections(frame)
                compliance = self._check_compliance(detected_ppe)
                
                # Update detection results
                self.detection_results = {
                    'people': detected_people,
                    'ppe_items': detected_ppe,
                    'compliance_status': compliance['status'],
                    'compliance_details': compliance,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Update system stats
                self.system_stats['frames_processed'] += 1
                fps_counter += 1
                
                if time.time() - fps_start_time >= 1.0:
                    self.system_stats['fps'] = fps_counter / (time.time() - fps_start_time)
                    fps_counter = 0
                    fps_start_time = time.time()
                
                # Draw detections on frame
                annotated_frame = self._draw_detections(frame.copy(), detected_people, detected_ppe, compliance)
                
                # Convert frame to base64 for web transmission
                _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                
                # Emit to connected clients
                socketio.emit('detection_update', {
                    'frame': frame_b64,
                    'detections': self.detection_results,
                    'stats': self.system_stats,
                    'unknown_people_count': len(self.unknown_people)
                })
                
                # Log to CSV periodically
                if self.system_stats['frames_processed'] % DETECTION_CONFIG["log_interval"] == 0:
                    self._log_detection(detected_people, compliance)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ERROR] Processing error: {e}")
        
        print("[INFO] Frame processing stopped")
    
    def _draw_detections(self, frame, people, ppe_items, compliance):
        """Draw detection results on frame"""
        # Draw people
        for person in people:
            x1, y1, x2, y2 = person['bbox']
            color = (255, 0, 255)  # Magenta for faces
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, person['name'], (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw PPE
        for ppe in ppe_items:
            x1, y1, x2, y2 = ppe['bbox']
            color = (0, 255, 0) if ppe['required'] else (0, 0, 255)  # Green for required, red for others
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"{ppe['label']} ({ppe['confidence']:.2f})"
            cv2.putText(frame, text, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add status overlay
        status_color = (0, 255, 0) if compliance['status'] == 'COMPLIANT' else (0, 0, 255)
        cv2.putText(frame, f"Status: {compliance['status']}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(frame, f"People: {len(people)}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {self.system_stats['fps']:.1f}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def _log_detection(self, people, compliance):
        """Log detection to CSV"""
        log_file = os.path.join(self.output_dir, PATHS["log_filename"])
        
        if not os.path.exists(log_file):
            with open(log_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["timestamp", "people_detected", "ppe_status", "present_ppe", "missing_ppe"])
                writer.writeheader()
        
        with open(log_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "people_detected", "ppe_status", "present_ppe", "missing_ppe"])
            people_names = [p['name'] for p in people]
            writer.writerow({
                "timestamp": datetime.now().isoformat(),
                "people_detected": ", ".join(people_names) if people_names else "None",
                "ppe_status": compliance['status'],
                "present_ppe": ", ".join(compliance['present_ppe']),
                "missing_ppe": ", ".join(compliance['missing_ppe'])
            })
    
    def start(self):
        """Start the detection system"""
        if self.running:
            return False
        
        self.running = True
        self.system_stats['start_time'] = datetime.now().isoformat()
        
        # Start threads
        self.capture_thread = threading.Thread(target=self._capture_frames)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        self.processing_thread = threading.Thread(target=self._process_frames)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        print("[INFO] PPE Detection System started")
        return True
    
    def stop(self):
        """Stop the detection system"""
        self.running = False
        self.system_stats['connection_status'] = 'Disconnected'
        print("[INFO] PPE Detection System stopped")
    
    def control_camera(self, action, params):
        """Control camera PTZ functions"""
        if not self.onvif_camera or not self.profile_token:
            return {'success': False, 'message': 'Camera control not available'}
        
        try:
            if action == 'pan_tilt':
                # Pan and tilt relative movement
                x = params.get('x', 0)  # -1 to 1
                y = params.get('y', 0)  # -1 to 1
                
                # Create PTZ request
                request = self.ptz_service.create_type('RelativeMove')
                request.ProfileToken = self.profile_token
                request.Translation = {
                    'PanTilt': {'x': x, 'y': y},
                    'Zoom': {'x': 0}
                }
                request.Speed = {
                    'PanTilt': {'x': 0.5, 'y': 0.5},
                    'Zoom': {'x': 0}
                }
                
                self.ptz_service.RelativeMove(request)
                return {'success': True, 'message': f'Camera moved (x={x}, y={y})'}
                
            elif action == 'zoom':
                # Zoom in/out
                direction = params.get('direction', 'in')
                zoom_value = 0.1 if direction == 'in' else -0.1
                
                request = self.ptz_service.create_type('RelativeMove')
                request.ProfileToken = self.profile_token
                request.Translation = {
                    'PanTilt': {'x': 0, 'y': 0},
                    'Zoom': {'x': zoom_value}
                }
                request.Speed = {
                    'PanTilt': {'x': 0, 'y': 0},
                    'Zoom': {'x': 0.5}
                }
                
                self.ptz_service.RelativeMove(request)
                return {'success': True, 'message': f'Camera zoomed {direction}'}
                
            elif action == 'preset':
                # Preset operations
                preset_action = params.get('action', 'goto')  # 'set' or 'goto'
                preset_id = params.get('presetId', 1)
                
                if preset_action == 'set':
                    request = self.ptz_service.create_type('SetPreset')
                    request.ProfileToken = self.profile_token
                    request.PresetToken = str(preset_id)
                    request.PresetName = f"Position_{preset_id}"
                    self.ptz_service.SetPreset(request)
                    return {'success': True, 'message': f'Preset {preset_id} saved'}
                    
                elif preset_action == 'goto':
                    request = self.ptz_service.create_type('GotoPreset')
                    request.ProfileToken = self.profile_token
                    request.PresetToken = str(preset_id)
                    request.Speed = {
                        'PanTilt': {'x': 0.5, 'y': 0.5},
                        'Zoom': {'x': 0.5}
                    }
                    self.ptz_service.GotoPreset(request)
                    return {'success': True, 'message': f'Moved to preset {preset_id}'}
                    
            else:
                return {'success': False, 'message': f'Unknown action: {action}'}
                
        except Exception as e:
            print(f"[ERROR] Camera control error: {e}")
            return {'success': False, 'message': f'Camera control failed: {str(e)}'}

# Initialize detection system
detection_system = PPEDetectionServer()

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    return jsonify({
        'running': detection_system.running,
        'stats': detection_system.system_stats,
        'detections': detection_system.detection_results
    })

@app.route('/api/start', methods=['POST'])
def start_detection():
    success = detection_system.start()
    return jsonify({'success': success, 'message': 'System started' if success else 'System already running'})

@app.route('/api/stop', methods=['POST'])
def stop_detection():
    detection_system.stop()
    return jsonify({'success': True, 'message': 'System stopped'})

@app.route('/api/camera/control', methods=['POST'])
def control_camera():
    """Camera PTZ control endpoint"""
    try:
        data = request.json
        action = data.get('action')
        params = data.get('params', {})
        
        if not action:
            return jsonify({'success': False, 'message': 'Action required'}), 400
            
        result = detection_system.control_camera(action, params)
        return jsonify(result)
        
    except Exception as e:
        print(f"[ERROR] Camera control API error: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/config')
def get_config():
    """Get current system configuration"""
    try:
        return jsonify({
            "camera": {
                "ip": CAMERA_CONFIG.get('ip', ''),
                "port": CAMERA_CONFIG.get('rtsp_port', 554),
                "stream_path": CAMERA_CONFIG.get('stream_path', ''),
                "detection_interval": CAMERA_CONFIG.get('detection_interval', 1),
                "use_low_resolution": CAMERA_CONFIG.get('use_low_resolution', False),
                "frame_skip": CAMERA_CONFIG.get('frame_skip', 3),
                "buffer_size": CAMERA_CONFIG.get('buffer_size', 1),
                "fps_limit": CAMERA_CONFIG.get('fps_limit', 15)
            },
            "detection": {
                "detect_unknown_people": DETECTION_CONFIG.get('detect_unknown_people', True),
                "detect_ppe": DETECTION_CONFIG.get('detect_ppe', True),
                "unknown_detection_cooldown": DETECTION_CONFIG.get('unknown_detection_cooldown', 30),
                "yolo_confidence": DETECTION_CONFIG.get('yolo_confidence', 0.5),
                "face_tolerance": DETECTION_CONFIG.get('face_tolerance', 0.6),
                "required_ppe": DETECTION_CONFIG.get('required_ppe', ["Hardhat", "Mask", "Safety Vest"]),
                "alert_cooldown": DETECTION_CONFIG.get('alert_cooldown', 30),
                "log_interval": DETECTION_CONFIG.get('log_interval', 30),
                "auto_cleanup_logs": DETECTION_CONFIG.get('auto_cleanup_logs', False),
                "max_log_days": DETECTION_CONFIG.get('max_log_days', 30)
            },
            "system": {
                "model_path": PATHS.get('model_path', ''),
                "employee_faces_dir": PATHS.get('employee_faces', ''),
                "output_dir": PATHS.get('output_directory', ''),
                "version": "2.0.0",
                "last_updated": datetime.now().isoformat()
            }
        })
    except Exception as e:
        print(f"[ERROR] Failed to get config: {e}")
        return jsonify({"error": "Failed to retrieve configuration"}), 500

@app.route('/api/config/detection', methods=['PUT'])
def update_detection_config():
    """Update detection configuration settings"""
    try:
        data = request.get_json()
        
        # Validate input data
        if not data:
            return jsonify({"success": False, "message": "No data provided"})
        
        # Define allowed settings with validation
        allowed_settings = {
            'detect_unknown_people': bool,
            'detect_ppe': bool,
            'unknown_detection_cooldown': (int, 5, 300),  # min, max values
            'yolo_confidence': (float, 0.1, 1.0),
            'face_tolerance': (float, 0.1, 1.0),
            'detection_interval': (int, 1, 10),
            'use_low_resolution': bool,
            'required_ppe': list,
            'alert_cooldown': (int, 5, 300),
            'log_interval': (int, 10, 300),
            'auto_cleanup_logs': bool,
            'max_log_days': (int, 7, 365),
            'frame_skip': (int, 0, 10),
            'buffer_size': (int, 1, 10),
            'fps_limit': (int, 5, 30)
        }
        
        # Validate and update settings
        updated_settings = {}
        errors = []
        
        for key, value in data.items():
            if key not in allowed_settings:
                errors.append(f"Unknown setting: {key}")
                continue
                
            setting_type = allowed_settings[key]
            
            # Type validation
            if isinstance(setting_type, type):
                if not isinstance(value, setting_type):
                    errors.append(f"{key} must be of type {setting_type.__name__}")
                    continue
                updated_settings[key] = value
            elif isinstance(setting_type, tuple):
                # Range validation for numeric types
                expected_type, min_val, max_val = setting_type
                if not isinstance(value, expected_type):
                    errors.append(f"{key} must be of type {expected_type.__name__}")
                    continue
                if not (min_val <= value <= max_val):
                    errors.append(f"{key} must be between {min_val} and {max_val}")
                    continue
                updated_settings[key] = value
        
        # Special validation for required_ppe
        if 'required_ppe' in updated_settings:
            valid_ppe_items = ["Hardhat", "Mask", "Safety Vest", "Safety Glasses", "Gloves", "Safety Boots", "High-Vis Jacket"]
            ppe_list = updated_settings['required_ppe']
            
            if not isinstance(ppe_list, list):
                errors.append("required_ppe must be a list")
            elif data.get('detect_ppe', True) and len(ppe_list) == 0:
                errors.append("At least one PPE item must be selected when PPE detection is enabled")
            elif not all(item in valid_ppe_items for item in ppe_list):
                errors.append(f"Invalid PPE items. Must be from: {', '.join(valid_ppe_items)}")
        
        if errors:
            return jsonify({"success": False, "message": "; ".join(errors)})
        
        # Update the global configuration
        for key, value in updated_settings.items():
            if key in ['detect_unknown_people', 'detect_ppe', 'unknown_detection_cooldown', 
                      'yolo_confidence', 'face_tolerance', 'required_ppe', 'alert_cooldown', 
                      'log_interval', 'auto_cleanup_logs', 'max_log_days']:
                DETECTION_CONFIG[key] = value
            elif key in ['detection_interval', 'use_low_resolution', 'frame_skip', 'buffer_size', 'fps_limit']:
                CAMERA_CONFIG[key] = value
        
        # Update the detection system's settings if it's running
        if detection_system:
            # Update required PPE
            if 'required_ppe' in updated_settings:
                detection_system.required_ppe = set(updated_settings['required_ppe'])
            
            # Update other detection settings
            for key in ['detect_unknown_people', 'detect_ppe', 'unknown_detection_cooldown']:
                if key in updated_settings:
                    setattr(detection_system, key, updated_settings[key])
        
        # Save configuration to file
        _save_config_to_file()
        
        return jsonify({
            "success": True, 
            "message": "Settings updated successfully",
            "updated_settings": updated_settings
        })
        
    except Exception as e:
        print(f"[ERROR] Failed to update config: {e}")
        return jsonify({"success": False, "message": f"Failed to update settings: {str(e)}"})

def _save_config_to_file():
    """Save current configuration to camera_config.py file"""
    try:
        config_content = f'''"""
Camera Configuration for Live PPE Detection System
Auto-generated by app.py - Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

# TP-Link Tapo C520 Camera Configuration
CAMERA_CONFIG = {{
    # Network Configuration
    "ip": "{CAMERA_CONFIG['ip']}",
    "rtsp_port": {CAMERA_CONFIG['rtsp_port']},
    "onvif_port": {CAMERA_CONFIG.get('onvif_port', 2020)},
    
    # Authentication
    "username": "{CAMERA_CONFIG['username']}",
    "password": "{CAMERA_CONFIG['password']}",
    
    # Stream Configuration
    "stream_path": "{CAMERA_CONFIG['stream_path']}",
    "use_tcp": {CAMERA_CONFIG.get('use_tcp', True)},
    
    # Performance Settings
    "fps_limit": {CAMERA_CONFIG.get('fps_limit', 15)},
    "buffer_size": {CAMERA_CONFIG.get('buffer_size', 1)},
    "frame_skip": {CAMERA_CONFIG.get('frame_skip', 3)},
    "use_low_resolution": {CAMERA_CONFIG.get('use_low_resolution', False)},
    "detection_interval": {CAMERA_CONFIG.get('detection_interval', 1)},
}}

# Detection Configuration
DETECTION_CONFIG = {{
    "required_ppe": {DETECTION_CONFIG['required_ppe']},
    "yolo_confidence": {DETECTION_CONFIG.get('yolo_confidence', 0.5)},
    "face_tolerance": {DETECTION_CONFIG.get('face_tolerance', 0.6)},
    "face_model": "{DETECTION_CONFIG.get('face_model', 'hog')}",
    "log_interval": {DETECTION_CONFIG.get('log_interval', 30)},
    "alert_cooldown": {DETECTION_CONFIG.get('alert_cooldown', 30)},
    "detect_unknown_people": {DETECTION_CONFIG.get('detect_unknown_people', True)},
    "unknown_detection_cooldown": {DETECTION_CONFIG.get('unknown_detection_cooldown', 30)},
    "detect_ppe": {DETECTION_CONFIG.get('detect_ppe', True)},
    "auto_cleanup_logs": {DETECTION_CONFIG.get('auto_cleanup_logs', False)},
    "max_log_days": {DETECTION_CONFIG.get('max_log_days', 30)},
}}

# Directory Configuration
PATHS = {{
    "employee_faces": "{PATHS['employee_faces']}",
    "output_directory": "{PATHS['output_directory']}",
    "model_path": "{PATHS['model_path']}",
    "log_filename": "{PATHS['log_filename']}"
}}

# Display Configuration
DISPLAY_CONFIG = {{
    "window_name": "{DISPLAY_CONFIG['window_name']}",
    "show_fps": {DISPLAY_CONFIG.get('show_fps', True)},
    "show_confidence": {DISPLAY_CONFIG.get('show_confidence', True)},
    "face_color": {DISPLAY_CONFIG['face_color']},
    "ppe_compliant_color": {DISPLAY_CONFIG['ppe_compliant_color']},
    "ppe_other_color": {DISPLAY_CONFIG['ppe_other_color']},
    "status_compliant_color": {DISPLAY_CONFIG['status_compliant_color']},
    "status_violation_color": {DISPLAY_CONFIG['status_violation_color']},
}}

def get_rtsp_url():
    """Generate RTSP URL from configuration"""
    config = CAMERA_CONFIG
    return f"rtsp://{{config['username']}}:{{config['password']}}@{{config['ip']}}:{{config['rtsp_port']}}/{{config['stream_path']}}"

def print_config():
    """Print current configuration (hiding password)"""
    print("=" * 50)
    print("LIVE PPE DETECTION SYSTEM CONFIGURATION")
    print("=" * 50)
    
    config = CAMERA_CONFIG.copy()
    config['password'] = '*' * len(config['password'])
    
    print(f"Camera IP: {{config['ip']}}")
    print(f"RTSP Port: {{config['rtsp_port']}}")
    print(f"Username: {{config['username']}}")
    print(f"Password: {{config['password']}}")
    print(f"Stream Path: {{config['stream_path']}}")
    print(f"RTSP URL: rtsp://{{config['username']}}:***@{{config['ip']}}:{{config['rtsp_port']}}/{{config['stream_path']}}")
    print(f"Required PPE: {{', '.join(DETECTION_CONFIG['required_ppe'])}}")
    print(f"Model Path: {{PATHS['model_path']}}")
    print(f"Employee Faces Directory: {{PATHS['employee_faces']}}")
    print("=" * 50)

if __name__ == "__main__":
    print_config()
'''
        
        # Write to file
        with open('camera_config.py', 'w') as f:
            f.write(config_content)
            
        print("[INFO] Configuration saved to camera_config.py")
        
    except Exception as e:
        print(f"[ERROR] Failed to save config to file: {e}")

@app.route('/api/logs')
def get_logs():
    log_file = os.path.join(detection_system.output_dir, PATHS["log_filename"])
    logs = []
    
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            reader = csv.DictReader(f)
            logs = list(reader)[-50:]  # Last 50 entries
    
    return jsonify(logs)

@app.route('/api/unknown-people')
def get_unknown_people():
    """Get list of detected unknown people"""
    unknown_list = []
    for person in detection_system.unknown_people:
        # Create web-accessible image data
        try:
            with open(person['filepath'], 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            unknown_list.append({
                'id': person['id'],
                'timestamp': person['timestamp'],
                'last_seen': person['last_seen'],
                'image_data': image_data,
                'added_to_database': person['added_to_database']
            })
        except Exception as e:
            print(f"[WARNING] Could not load image for {person['id']}: {e}")
    
    return jsonify(unknown_list)

@app.route('/api/add-person', methods=['POST'])
def add_person_to_database():
    """Add unknown person to the known employees database"""
    data = request.json
    unknown_id = data.get('unknown_id')
    person_name = data.get('name', '').strip()
    
    if not unknown_id or not person_name:
        return jsonify({'success': False, 'message': 'Missing unknown_id or name'}), 400
    
    # Find the unknown person
    unknown_person = None
    for person in detection_system.unknown_people:
        if person['id'] == unknown_id:
            unknown_person = person
            break
    
    if not unknown_person:
        return jsonify({'success': False, 'message': 'Unknown person not found'}), 404
    
    try:
        # Create new employee image file
        dataset_dir = PATHS["employee_faces"]
        new_filename = f"{person_name}.jpg"
        new_filepath = os.path.join(dataset_dir, new_filename)
        
        # Copy image to dataset directory
        import shutil
        shutil.copy2(unknown_person['filepath'], new_filepath)
        
        # Add face encoding to known faces
        face_encoding = np.array(unknown_person['face_encoding'])
        detection_system.known_encodings.append(face_encoding)
        detection_system.known_names.append(person_name)
        
        # Mark as added to database
        unknown_person['added_to_database'] = True
        unknown_person['assigned_name'] = person_name
        
        print(f"[INFO] Added {person_name} to employee database")
        
        return jsonify({
            'success': True, 
            'message': f'Successfully added {person_name} to employee database'
        })
        
    except Exception as e:
        print(f"[ERROR] Failed to add person {person_name}: {e}")
        return jsonify({'success': False, 'message': f'Failed to add person: {str(e)}'}), 500

@app.route('/api/dismiss-unknown/<unknown_id>', methods=['DELETE'])
def dismiss_unknown_person(unknown_id):
    """Dismiss/remove an unknown person detection"""
    try:
        # Find and remove the unknown person
        for i, person in enumerate(detection_system.unknown_people):
            if person['id'] == unknown_id:
                # Remove image file
                if os.path.exists(person['filepath']):
                    os.remove(person['filepath'])
                
                # Remove from list
                detection_system.unknown_people.pop(i)
                
                return jsonify({'success': True, 'message': 'Unknown person dismissed'})
        
        return jsonify({'success': False, 'message': 'Unknown person not found'}), 404
        
    except Exception as e:
        print(f"[ERROR] Failed to dismiss unknown person {unknown_id}: {e}")
        return jsonify({'success': False, 'message': f'Failed to dismiss: {str(e)}'}), 500

@app.route('/api/unknown-people/image/<unknown_id>')
def get_unknown_person_image(unknown_id):
    """Serve unknown person image"""
    for person in detection_system.unknown_people:
        if person['id'] == unknown_id:
            try:
                return send_file(person['filepath'], mimetype='image/jpeg')
            except Exception as e:
                return jsonify({'error': str(e)}), 404
    
    return jsonify({'error': 'Unknown person not found'}), 404

# WebSocket events
@socketio.on('connect')
def handle_connect():
    print(f"[INFO] Client connected: {request.sid}")
    emit('status_update', {
        'running': detection_system.running,
        'stats': detection_system.system_stats
    })

@socketio.on('disconnect')
def handle_disconnect():
    print(f"[INFO] Client disconnected: {request.sid}")

if __name__ == '__main__':
    print("PPE Detection Web Server")
    print("=" * 40)
    print("Access the dashboard at: http://localhost:8080")
    print("=" * 40)
    
    socketio.run(app, host='0.0.0.0', port=8080, debug=True) 