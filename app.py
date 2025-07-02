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
try:
    from opencv_face_recognition import OpenCVFaceRecognition
    FACE_RECOGNITION_AVAILABLE = True
    opencv_face_rec = OpenCVFaceRecognition()
    if not opencv_face_rec.is_available():
        print("[WARNING] OpenCV face recognition models not found")
        FACE_RECOGNITION_AVAILABLE = False
    else:
        print("[INFO] OpenCV face recognition module loaded successfully")
except ImportError as e:
    print(f"[WARNING] Face recognition not available: {e}")
    FACE_RECOGNITION_AVAILABLE = False
    opencv_face_rec = None
from camera_config import *
from onvif import ONVIFCamera

app = Flask(__name__)
app.config['SECRET_KEY'] = 's_oasis_secret'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

class SOasisServer:
    def __init__(self):
        print("[INFO] Initializing S-Oasis Server...")
        
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
        self.live_detection_events = []  # Store live detection events
        self.system_stats = {
            'frames_processed': 0,
            'start_time': None,
            'fps': 0,
            'connection_status': 'Disconnected'
        }
        
        # Camera management
        self.current_camera_type = "rtsp"  # Default to RTSP camera
        self.available_cameras = self._detect_available_cameras()
        
        # Threading
        self.frame_queue = queue.Queue(maxsize=10)
        self.capture_thread = None
        self.processing_thread = None
        
        print("[INFO] S-Oasis Server initialized")
        
        # Initialize camera control
        self.onvif_camera = None
        self._init_camera_control()
    
    def _detect_available_cameras(self):
        """Detect available cameras (built-in and RTSP)"""
        cameras = []
        
        # Check for built-in Mac camera
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    cameras.append({
                        'id': 'builtin',
                        'name': 'Built-in Camera',
                        'type': 'usb',
                        'index': 0
                    })
                    print("[INFO] Built-in camera detected")
            cap.release()
        except Exception as e:
            print(f"[WARNING] Could not detect built-in camera: {e}")
        
        # Add RTSP camera (always available in config)
        cameras.append({
            'id': 'rtsp',
            'name': f'RTSP Camera ({CAMERA_CONFIG["ip"]})',
            'type': 'rtsp',
            'url': f"rtsp://{CAMERA_CONFIG['username']}:{CAMERA_CONFIG['password']}@{CAMERA_CONFIG['ip']}:{CAMERA_CONFIG['rtsp_port']}/{CAMERA_CONFIG['stream_path']}"
        })
        
        print(f"[INFO] Available cameras: {[cam['name'] for cam in cameras]}")
        return cameras

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
        if not FACE_RECOGNITION_AVAILABLE or opencv_face_rec is None:
            print("[WARNING] Face recognition disabled - OpenCV models not available")
            return
            
        dataset_dir = PATHS["employee_faces"]
        if not os.path.exists(dataset_dir):
            print(f"[WARNING] Employee faces directory not found: {dataset_dir}")
            return
        
        # Try to load existing model first
        if opencv_face_rec.load_model():
            print("[INFO] Loaded existing face recognition model")
            return
        
        # Load and train new model
        num_faces = opencv_face_rec.load_known_faces(dataset_dir)
        if num_faces > 0:
            print(f"[INFO] Loaded and trained face recognition with {num_faces} faces")
        else:
            print("[WARNING] No faces found in dataset directory")
    
    def _capture_frames(self):
        """Capture frames from selected camera source"""
        cap = None
        
        if self.current_camera_type == "rtsp":
            # RTSP Camera
            rtsp_url = f"rtsp://{CAMERA_CONFIG['username']}:{CAMERA_CONFIG['password']}@{CAMERA_CONFIG['ip']}:{CAMERA_CONFIG['rtsp_port']}/{CAMERA_CONFIG['stream_path']}"
            print(f"[INFO] Connecting to RTSP stream: {rtsp_url.replace(CAMERA_CONFIG['password'], '***')}")
            
            # Try different backends for RTSP
            backends_to_try = [cv2.CAP_FFMPEG, cv2.CAP_ANY]
            for backend in backends_to_try:
                cap = cv2.VideoCapture(rtsp_url, backend)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_CONFIG['buffer_size'])
                if cap.isOpened():
                    print(f"[INFO] Connected to RTSP with backend: {backend}")
                    break
                cap.release()
        
        elif self.current_camera_type == "builtin":
            # Built-in Camera
            print("[INFO] Connecting to built-in camera...")
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                print("[INFO] Connected to built-in camera")
                # Set some basic properties for built-in camera
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap or not cap.isOpened():
            print(f"[ERROR] Failed to connect to {self.current_camera_type} camera")
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
            # For built-in camera, process every frame; for RTSP, use frame skip
            frame_skip = 1 if self.current_camera_type == "builtin" else CAMERA_CONFIG['frame_skip']
            
            if frame_count % frame_skip == 0:
                if not self.frame_queue.full():
                    self.frame_queue.put(frame.copy())
                    self.current_frame = frame.copy()
        
        cap.release()
        print("[INFO] Frame capture stopped")
    
    def _process_detections(self, frame):
        """Process PPE and face detection"""
        # Face recognition
        detected_people = []
        if FACE_RECOGNITION_AVAILABLE and opencv_face_rec is not None:
            try:
                # Detect faces using OpenCV DNN
                detected_faces = opencv_face_rec.detect_faces(frame)
                
                for face_data in detected_faces:
                    bbox = face_data['bbox']
                    confidence = face_data['confidence']
                    
                    # Recognize the face
                    name, recognition_confidence = opencv_face_rec.recognize_face(frame, bbox)
                    
                    # Get face encoding for unknown person handling
                    face_encoding = opencv_face_rec.get_face_encoding(frame, bbox)
                    
                    detected_people.append({
                        'name': name,
                        'bbox': [int(x) for x in bbox],  # Convert to regular int
                        'confidence': float(confidence),
                        'recognition_confidence': float(recognition_confidence),
                        'face_encoding': face_encoding
                    })
                    
                    # Add live detection event
                    self._add_live_detection_event(name, confidence, recognition_confidence)
                    
                    # Handle unknown people (only if enabled and with additional checks)
                    if (name == "Unknown Person" and 
                        DETECTION_CONFIG.get("detect_unknown_people", True) and
                        recognition_confidence < 50):  # Only log truly unknown people
                        if face_encoding:
                            self._handle_unknown_person(frame, bbox, face_encoding)
                            
            except Exception as e:
                print(f"[WARNING] Face recognition error: {e}")
                # Continue without face recognition if there's an error
        
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
        if not FACE_RECOGNITION_AVAILABLE or opencv_face_rec is None:
            return
            
        # Check if this unknown person was recently detected (avoid duplicates)
        current_time = datetime.now()
        
        # Compare with recent unknown detections to avoid duplicates
        is_duplicate = False
        cooldown_seconds = DETECTION_CONFIG.get("unknown_detection_cooldown", 60)  # Increased cooldown
        for unknown in self.unknown_people:
            try:
                time_diff = (current_time - datetime.fromisoformat(unknown['timestamp'])).total_seconds()
                if time_diff < cooldown_seconds:
                    # Simple bbox-based comparison for faster duplicate detection
                    old_bbox = unknown.get('bbox', [0, 0, 0, 0])
                    # Check if faces are in similar location (within 50 pixels)
                    if (abs(bbox[0] - old_bbox[0]) < 50 and 
                        abs(bbox[1] - old_bbox[1]) < 50):
                        is_duplicate = True
                        unknown['last_seen'] = current_time.isoformat()
                        break
            except Exception as e:
                print(f"[WARNING] Error in duplicate detection: {e}")
                continue
        
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
                'face_encoding': face_encoding if isinstance(face_encoding, list) else face_encoding,
                'bbox': [int(x) for x in bbox] if bbox else [0, 0, 0, 0],  # Ensure int conversion
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
    
    def _add_live_detection_event(self, name, detection_confidence, recognition_confidence):
        """Add a live detection event"""
        current_time = datetime.now()
        
        # Check if this person was detected recently (within 5 seconds) to avoid spam
        for event in reversed(self.live_detection_events[-10:]):
            try:
                event_time = datetime.fromisoformat(event['timestamp'])
                if (current_time - event_time).total_seconds() < 5 and event['name'] == name:
                    return  # Skip if same person detected recently
            except:
                continue
        
        event = {
            'id': f"event_{int(current_time.timestamp())}_{len(self.live_detection_events)}",
            'timestamp': current_time.isoformat(),
            'time_display': current_time.strftime("%H:%M:%S"),
            'name': name,
            'detection_confidence': float(detection_confidence),
            'recognition_confidence': float(recognition_confidence),
            'event_type': 'face_detection'
        }
        
        self.live_detection_events.append(event)
        
        # Keep only last 100 events to prevent memory buildup
        if len(self.live_detection_events) > 100:
            self.live_detection_events = self.live_detection_events[-50:]
    
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
        
        print("[INFO] S-Oasis System started")
        return True
    
    def stop(self):
        """Stop the detection system"""
        self.running = False
        self.system_stats['connection_status'] = 'Disconnected'
        print("[INFO] S-Oasis System stopped")
    
    def switch_camera(self, camera_id):
        """Switch to a different camera"""
        # Check if camera is available
        camera_found = None
        for cam in self.available_cameras:
            if cam['id'] == camera_id:
                camera_found = cam
                break
        
        if not camera_found:
            return {"success": False, "error": f"Camera {camera_id} not found"}
        
        # Stop current capture if running
        was_running = self.running
        if was_running:
            self.stop()
            time.sleep(1)  # Give time for threads to stop
        
        # Switch camera
        self.current_camera_type = camera_id
        print(f"[INFO] Switched to camera: {camera_found['name']}")
        
        # Restart if it was running
        if was_running:
            success = self.start()
            return {"success": success, "message": f"Switched to {camera_found['name']} and restarted"}
        
        return {"success": True, "message": f"Switched to {camera_found['name']}"}

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
server = SOasisServer()

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    return jsonify({
        'running': server.running,
        'stats': server.system_stats,
        'detections': server.detection_results
    })

@app.route('/api/start', methods=['POST'])
def start_detection():
    success = server.start()
    return jsonify({'success': success, 'message': 'System started' if success else 'System already running'})

@app.route('/api/stop', methods=['POST'])
def stop_detection():
    server.stop()
    return jsonify({'success': True, 'message': 'System stopped'})

@app.route('/api/cameras')
def get_cameras():
    """Get available cameras"""
    return jsonify({
        'success': True,
        'cameras': server.available_cameras,
        'current_camera': server.current_camera_type
    })

@app.route('/api/camera/switch', methods=['POST'])
def switch_camera():
    """Switch camera endpoint"""
    try:
        data = request.json
        camera_id = data.get('camera_id')
        
        if not camera_id:
            return jsonify({'success': False, 'message': 'Camera ID required'}), 400
            
        result = server.switch_camera(camera_id)
        return jsonify(result)
        
    except Exception as e:
        print(f"[ERROR] Camera switch API error: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/camera/control', methods=['POST'])
def control_camera():
    """Camera PTZ control endpoint"""
    try:
        data = request.json
        action = data.get('action')
        params = data.get('params', {})
        
        if not action:
            return jsonify({'success': False, 'message': 'Action required'}), 400
            
        result = server.control_camera(action, params)
        return jsonify(result)
        
    except Exception as e:
        print(f"[ERROR] Camera control API error: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/config')
def get_config():
    return jsonify({
        'camera': CAMERA_CONFIG,
        'detection': DETECTION_CONFIG,
        'display': DISPLAY_CONFIG
    })

@app.route('/api/config/detection', methods=['PUT'])
def update_detection_config():
    """Update detection configuration settings"""
    try:
        data = request.json
        
        # Update allowed settings
        if 'detect_unknown_people' in data:
            DETECTION_CONFIG['detect_unknown_people'] = bool(data['detect_unknown_people'])
        
        if 'detect_ppe' in data:
            DETECTION_CONFIG['detect_ppe'] = bool(data['detect_ppe'])
        
        if 'unknown_detection_cooldown' in data:
            cooldown = int(data['unknown_detection_cooldown'])
            if 5 <= cooldown <= 300:  # Between 5 seconds and 5 minutes
                DETECTION_CONFIG['unknown_detection_cooldown'] = cooldown
        
        if 'yolo_confidence' in data:
            confidence = float(data['yolo_confidence'])
            if 0.1 <= confidence <= 1.0:
                DETECTION_CONFIG['yolo_confidence'] = confidence
        
        if 'face_tolerance' in data:
            tolerance = float(data['face_tolerance'])
            if 0.1 <= tolerance <= 1.0:
                DETECTION_CONFIG['face_tolerance'] = tolerance
        
        if 'detection_interval' in data:
            interval = int(data['detection_interval'])
            if 1 <= interval <= 10:  # Process every 1-10 frames
                CAMERA_CONFIG['detection_interval'] = interval
        
        if 'use_low_resolution' in data:
            CAMERA_CONFIG['use_low_resolution'] = bool(data['use_low_resolution'])
            # Update stream path
            if CAMERA_CONFIG['use_low_resolution']:
                CAMERA_CONFIG['stream_path'] = 'stream2'  # Lower resolution
            else:
                CAMERA_CONFIG['stream_path'] = 'stream1'  # Higher resolution
        
        # Save updated config to file
        _save_config_to_file()
        
        return jsonify({
            'success': True, 
            'message': 'Detection configuration updated',
            'config': DETECTION_CONFIG
        })
        
    except Exception as e:
        print(f"[ERROR] Failed to update detection config: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

def _save_config_to_file():
    """Save current configuration back to camera_config.py"""
    try:
        config_content = f'''"""
Camera Configuration for Live PPE Detection System
Auto-generated by setup_system.py
"""

# TP-Link Tapo C520 Camera Configuration
CAMERA_CONFIG = {{
    # Network Configuration
    "ip": "{CAMERA_CONFIG['ip']}",
    "rtsp_port": {CAMERA_CONFIG['rtsp_port']},
    
    # Authentication
    "username": "{CAMERA_CONFIG['username']}",
    "password": "{CAMERA_CONFIG['password']}",
    
    # Stream Configuration
    "stream_path": "{CAMERA_CONFIG['stream_path']}",
         "use_tcp": {CAMERA_CONFIG['use_tcp']},
     
     # Performance Settings
     "fps_limit": {CAMERA_CONFIG['fps_limit']},
     "buffer_size": {CAMERA_CONFIG['buffer_size']},
     "frame_skip": {CAMERA_CONFIG['frame_skip']},
     "use_low_resolution": {CAMERA_CONFIG.get('use_low_resolution', False)},
     "detection_interval": {CAMERA_CONFIG.get('detection_interval', 1)},
}}

# Detection Configuration
DETECTION_CONFIG = {{
    "required_ppe": {DETECTION_CONFIG['required_ppe']},
    "yolo_confidence": {DETECTION_CONFIG['yolo_confidence']},
    "face_tolerance": {DETECTION_CONFIG['face_tolerance']},
    "face_model": "{DETECTION_CONFIG['face_model']}",
    "log_interval": {DETECTION_CONFIG['log_interval']},
         "alert_cooldown": {DETECTION_CONFIG['alert_cooldown']},
     "detect_unknown_people": {DETECTION_CONFIG['detect_unknown_people']},
     "unknown_detection_cooldown": {DETECTION_CONFIG['unknown_detection_cooldown']},
     "detect_ppe": {DETECTION_CONFIG.get('detect_ppe', True)},
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
    "show_fps": {DISPLAY_CONFIG['show_fps']},
    "show_confidence": {DISPLAY_CONFIG['show_confidence']},
    "face_color": {DISPLAY_CONFIG['face_color']},
    "ppe_compliant_color": {DISPLAY_CONFIG['ppe_compliant_color']},
    "ppe_other_color": {DISPLAY_CONFIG['ppe_other_color']},
    "status_compliant_color": {DISPLAY_CONFIG['status_compliant_color']},
    "status_violation_color": {DISPLAY_CONFIG['status_violation_color']},
}}

def get_rtsp_url():
    """Generate RTSP URL from configuration"""
    config = CAMERA_CONFIG
    return f"rtsp://{config['username']}:{config['password']}@{config['ip']}:{config['rtsp_port']}/{config['stream_path']}"

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
        
        with open('camera_config.py', 'w') as f:
            f.write(config_content)
            
        print("[INFO] Configuration saved to camera_config.py")
        
    except Exception as e:
        print(f"[ERROR] Failed to save config to file: {e}")

@app.route('/api/logs')
def get_logs():
    log_file = os.path.join(server.output_dir, PATHS["log_filename"])
    logs = []
    
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            reader = csv.DictReader(f)
            logs = list(reader)[-50:]  # Last 50 entries
    
    return jsonify(logs)

@app.route('/api/logs/clear', methods=['DELETE'])
def clear_logs():
    """Clear all detection logs"""
    try:
        log_file = os.path.join(server.output_dir, PATHS["log_filename"])
        if os.path.exists(log_file):
            os.remove(log_file)
        
        # Also clear live detection events
        server.live_detection_events = []
        
        return jsonify({'success': True, 'message': 'Logs cleared successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/logs/live')
def get_live_logs():
    """Get recent live detection events"""
    return jsonify(server.live_detection_events[-30:])  # Return last 30 events

@app.route('/api/unknown-people')
def get_unknown_people():
    """Get list of detected unknown people"""
    unknown_list = []
    for person in server.unknown_people:
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
    """Add unknown person to the known employees database or link to existing person"""
    data = request.json
    unknown_id = data.get('unknown_id')
    person_name = data.get('name', '').strip()
    action_type = data.get('action', 'new')  # 'new' or 'existing'
    
    if not unknown_id or not person_name:
        return jsonify({'success': False, 'message': 'Missing unknown_id or name'}), 400
    
    # Find the unknown person
    unknown_person = None
    for person in server.unknown_people:
        if person['id'] == unknown_id:
            unknown_person = person
            break
    
    if not unknown_person:
        return jsonify({'success': False, 'message': 'Unknown person not found'}), 404
    
    try:
        dataset_dir = PATHS["employee_faces"]
        
        if action_type == 'existing':
            # Link to existing person - add as additional training image
            existing_count = 0
            for filename in os.listdir(dataset_dir):
                if filename.startswith(f"{person_name}_") or filename == f"{person_name}.jpg":
                    existing_count += 1
            
            new_filename = f"{person_name}_{existing_count + 1}.jpg"
            new_filepath = os.path.join(dataset_dir, new_filename)
            
            print(f"[INFO] Adding additional training image for existing person: {person_name}")
        else:
            # Create new person
            new_filename = f"{person_name}.jpg"
            new_filepath = os.path.join(dataset_dir, new_filename)
            
            print(f"[INFO] Creating new person: {person_name}")
        
        # Copy image to dataset directory
        import shutil
        shutil.copy2(unknown_person['filepath'], new_filepath)
        
        # Reload face recognition to include the new image
        if hasattr(server, 'face_recognition') and server.face_recognition:
            server.face_recognition.load_known_faces(dataset_dir)
            print(f"[INFO] Face recognition model retrained with new image")
        
        # Mark as added to database
        unknown_person['added_to_database'] = True
        unknown_person['assigned_name'] = person_name
        unknown_person['action_type'] = action_type
        
        message = f'Successfully added additional training image for {person_name}' if action_type == 'existing' else f'Successfully added {person_name} to employee database'
        
        return jsonify({
            'success': True, 
            'message': message
        })
        
    except Exception as e:
        print(f"[ERROR] Failed to add person {person_name}: {e}")
        return jsonify({'success': False, 'message': f'Failed to add person: {str(e)}'}), 500

@app.route('/api/known-people')
def get_known_people():
    """Get list of known people for linking unknown detections"""
    try:
        known_people = []
        
        # Check multiple sources for known people
        if hasattr(server, 'face_recognition') and server.face_recognition:
            if hasattr(server.face_recognition, 'known_names') and server.face_recognition.known_names:
                known_people = list(set(server.face_recognition.known_names))  # Remove duplicates
            elif hasattr(server.face_recognition, 'person_to_label') and server.face_recognition.person_to_label:
                known_people = list(server.face_recognition.person_to_label.keys())
        
        # Fallback: scan dataset directory for known people
        if not known_people:
            import os
            dataset_dir = PATHS["employee_faces"]
            if os.path.exists(dataset_dir):
                for filename in os.listdir(dataset_dir):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        base_name = os.path.splitext(filename)[0]
                        if '_' in base_name and base_name.split('_')[-1].isdigit():
                            person_name = '_'.join(base_name.split('_')[:-1])
                        else:
                            person_name = base_name
                        
                        if person_name not in known_people:
                            known_people.append(person_name)
        
        print(f"[INFO] Found {len(known_people)} known people: {known_people}")
        
        return jsonify({
            'success': True,
            'people': sorted(known_people)  # Sort alphabetically
        })
    except Exception as e:
        print(f"[ERROR] Failed to get known people: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/add-person-direct', methods=['POST'])
def add_person_direct():
    """Add a new person directly with uploaded image"""
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'message': 'No image provided'}), 400
        
        file = request.files['image']
        person_name = request.form.get('name', '').strip()
        
        if not person_name:
            return jsonify({'success': False, 'message': 'Person name is required'}), 400
        
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No image selected'}), 400
        
        # Check if person already exists
        dataset_dir = PATHS["employee_faces"]
        existing_files = []
        if os.path.exists(dataset_dir):
            for filename in os.listdir(dataset_dir):
                if filename.startswith(person_name) and filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    existing_files.append(filename)
        
        if existing_files:
            return jsonify({
                'success': False, 
                'message': f'Person "{person_name}" already exists. Use "Add Training Image" to add more photos.'
            }), 400
        
        # Validate file type
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            return jsonify({
                'success': False, 
                'message': 'Invalid file type. Please use JPG, PNG, or BMP images.'
            }), 400
        
        # Create dataset directory if it doesn't exist
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Save the image
        filename = f"{person_name}.jpg"
        filepath = os.path.join(dataset_dir, filename)
        
        # Convert uploaded file to OpenCV format for face detection
        import cv2
        import numpy as np
        
        # Read image data
        file_data = file.read()
        np_array = np.frombuffer(file_data, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'success': False, 'message': 'Invalid image format'}), 400
        
        # Check if there's a face in the image
        if hasattr(server, 'face_recognition') and server.face_recognition:
            faces = server.face_recognition.detect_faces(image)
            if not faces:
                return jsonify({
                    'success': False, 
                    'message': 'No face detected in the image. Please use a clear photo with a visible face.'
                }), 400
        
        # Save the processed image
        cv2.imwrite(filepath, image)
        
        # Reload face recognition model to include the new person
        if hasattr(server, 'face_recognition') and server.face_recognition:
            server.face_recognition.load_known_faces(dataset_dir)
            print(f"[INFO] Face recognition model retrained with new person: {person_name}")
        
        return jsonify({
            'success': True,
            'message': f'Successfully added {person_name} to the database'
        })
        
    except Exception as e:
        print(f"[ERROR] Failed to add person directly: {e}")
        return jsonify({'success': False, 'message': f'Failed to add person: {str(e)}'}), 500

@app.route('/api/add-training-image', methods=['POST'])
def add_training_image():
    """Add additional training image for an existing person"""
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'message': 'No image provided'}), 400
        
        file = request.files['image']
        person_name = request.form.get('name', '').strip()
        
        if not person_name:
            return jsonify({'success': False, 'message': 'Person name is required'}), 400
        
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No image selected'}), 400
        
        # Check if person exists
        dataset_dir = PATHS["employee_faces"]
        person_exists = False
        if os.path.exists(dataset_dir):
            for filename in os.listdir(dataset_dir):
                if filename.startswith(person_name) and filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    person_exists = True
                    break
        
        if not person_exists:
            return jsonify({
                'success': False,
                'message': f'Person "{person_name}" not found. Add them as a new person first.'
            }), 400
        
        # Count existing images for this person
        existing_count = 0
        for filename in os.listdir(dataset_dir):
            if filename.startswith(f"{person_name}_") or filename == f"{person_name}.jpg":
                existing_count += 1
        
        # Validate file type
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            return jsonify({
                'success': False,
                'message': 'Invalid file type. Please use JPG, PNG, or BMP images.'
            }), 400
        
        # Process and save the image
        import cv2
        import numpy as np
        
        file_data = file.read()
        np_array = np.frombuffer(file_data, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'success': False, 'message': 'Invalid image format'}), 400
        
        # Check for face
        if hasattr(server, 'face_recognition') and server.face_recognition:
            faces = server.face_recognition.detect_faces(image)
            if not faces:
                return jsonify({
                    'success': False,
                    'message': 'No face detected in the image. Please use a clear photo with a visible face.'
                }), 400
        
        # Save with incremented filename
        filename = f"{person_name}_{existing_count + 1}.jpg"
        filepath = os.path.join(dataset_dir, filename)
        cv2.imwrite(filepath, image)
        
        # Retrain model
        if hasattr(server, 'face_recognition') and server.face_recognition:
            server.face_recognition.load_known_faces(dataset_dir)
            print(f"[INFO] Added training image {existing_count + 1} for {person_name}")
        
        return jsonify({
            'success': True,
            'message': f'Successfully added training image #{existing_count + 1} for {person_name}'
        })
        
    except Exception as e:
        print(f"[ERROR] Failed to add training image: {e}")
        return jsonify({'success': False, 'message': f'Failed to add training image: {str(e)}'}), 500

@app.route('/api/add-multiple-training-images', methods=['POST'])
def add_multiple_training_images():
    """Add multiple training images from camera capture"""
    try:
        person_name = request.form.get('name', '').strip()
        
        if not person_name:
            return jsonify({'success': False, 'message': 'Person name is required'}), 400
        
        # Get all uploaded files
        uploaded_files = []
        for key in request.files:
            if key.startswith('image_'):
                uploaded_files.append(request.files[key])
        
        if not uploaded_files:
            return jsonify({'success': False, 'message': 'No images provided'}), 400
        
        dataset_dir = PATHS["employee_faces"]
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Check if person exists (for training images) or doesn't exist (for new person)
        person_exists = False
        existing_count = 0
        if os.path.exists(dataset_dir):
            for filename in os.listdir(dataset_dir):
                if filename.startswith(person_name) and filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    person_exists = True
                    if filename.startswith(f"{person_name}_") or filename == f"{person_name}.jpg":
                        existing_count += 1
        
        import cv2
        import numpy as np
        
        saved_images = []
        
        for i, file in enumerate(uploaded_files):
            if file.filename == '':
                continue
                
            # Validate file type
            allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            file_ext = os.path.splitext(file.filename)[1].lower()
            if file_ext not in allowed_extensions:
                continue
            
            # Process image
            file_data = file.read()
            np_array = np.frombuffer(file_data, np.uint8)
            image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
            
            if image is None:
                continue
            
            # Check for face
            if hasattr(server, 'face_recognition') and server.face_recognition:
                faces = server.face_recognition.detect_faces(image)
                if not faces:
                    continue
            
            # Determine filename
            if i == 0 and not person_exists:
                # First image for new person
                filename = f"{person_name}.jpg"
            else:
                # Additional training images
                filename = f"{person_name}_{existing_count + len(saved_images) + 1}.jpg"
            
            filepath = os.path.join(dataset_dir, filename)
            cv2.imwrite(filepath, image)
            saved_images.append(filename)
        
        if not saved_images:
            return jsonify({
                'success': False,
                'message': 'No valid images with faces could be processed'
            }), 400
        
        # Retrain model
        if hasattr(server, 'face_recognition') and server.face_recognition:
            server.face_recognition.load_known_faces(dataset_dir)
            if person_exists:
                print(f"[INFO] Added {len(saved_images)} training images for {person_name}")
            else:
                print(f"[INFO] Added new person {person_name} with {len(saved_images)} training images")
        
        message = f'Successfully added {len(saved_images)} training images for {person_name}'
        if not person_exists:
            message = f'Successfully added {person_name} with {len(saved_images)} training images'
        
        return jsonify({
            'success': True,
            'message': message,
            'images_saved': len(saved_images)
        })
        
    except Exception as e:
        print(f"[ERROR] Failed to add multiple training images: {e}")
        return jsonify({'success': False, 'message': f'Failed to add images: {str(e)}'}), 500

@app.route('/api/dismiss-unknown/<unknown_id>', methods=['DELETE'])
def dismiss_unknown_person(unknown_id):
    """Dismiss/remove an unknown person detection"""
    try:
        # Find and remove the unknown person
        for i, person in enumerate(server.unknown_people):
            if person['id'] == unknown_id:
                # Remove image file
                if os.path.exists(person['filepath']):
                    os.remove(person['filepath'])
                
                # Remove from list
                server.unknown_people.pop(i)
                
                return jsonify({'success': True, 'message': 'Unknown person dismissed'})
        
        return jsonify({'success': False, 'message': 'Unknown person not found'}), 404
        
    except Exception as e:
        print(f"[ERROR] Failed to dismiss unknown person {unknown_id}: {e}")
        return jsonify({'success': False, 'message': f'Failed to dismiss: {str(e)}'}), 500

@app.route('/api/unknown-people/image/<unknown_id>')
def get_unknown_person_image(unknown_id):
    """Serve unknown person image"""
    for person in server.unknown_people:
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
        'running': server.running,
        'stats': server.system_stats
    })

@socketio.on('disconnect')
def handle_disconnect():
    print(f"[INFO] Client disconnected: {request.sid}")

if __name__ == '__main__':
    print("S-Oasis Web Server")
    print("=" * 40)
    print("Access the dashboard at: http://localhost:8080")
    print("=" * 40)
    
    socketio.run(app, host='0.0.0.0', port=8080, debug=True) 