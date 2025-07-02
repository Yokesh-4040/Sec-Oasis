import cv2
import numpy as np
import os
from datetime import datetime
import pickle
import json

class OpenCVFaceRecognition:
    def __init__(self):
        """Initialize OpenCV-based face recognition"""
        self.face_cascade = None
        self.face_recognizer = None
        self.known_faces = {}
        self.known_names = []
        self.confidence_threshold = 0.5
        self.recognition_threshold = 100  # Increased threshold for better recognition
        
        # Initialize face detection
        self._load_face_detector()
        
        # Initialize face recognition
        self._init_face_recognizer()
        
    def _load_face_detector(self):
        """Load OpenCV Haar cascade face detection model"""
        try:
            # Use OpenCV's built-in Haar cascade for face detection
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            if not self.face_cascade.empty():
                print("[INFO] OpenCV Haar cascade face detector loaded successfully")
                return True
            else:
                print("[WARNING] Failed to load Haar cascade face detector")
                return False
        except Exception as e:
            print(f"[ERROR] Failed to load face detector: {e}")
            return False
    
    def _init_face_recognizer(self):
        """Initialize face recognizer"""
        try:
            # Use LBPH (Local Binary Patterns Histograms) face recognizer with better parameters
            self.face_recognizer = cv2.face.LBPHFaceRecognizer_create(
                radius=1, neighbors=8, grid_x=8, grid_y=8, threshold=100.0
            )
            print("[INFO] Face recognizer initialized")
        except Exception as e:
            print(f"[ERROR] Failed to initialize face recognizer: {e}")
            # Fallback to basic template matching if LBPH is not available
            self.face_recognizer = None
    
    def detect_faces(self, frame):
        """Detect faces in frame using OpenCV Haar cascade with NMS to eliminate duplicates"""
        if self.face_cascade is None:
            return []
        
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces with more sensitive parameters
            faces_rect = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=3,  # Reduced to detect more faces, we'll filter with NMS
                minSize=(30, 30)
            )
            
            if len(faces_rect) == 0:
                return []
            
            # Apply Non-Maximum Suppression to eliminate overlapping detections
            boxes = []
            confidences = []
            
            for (x, y, w, h) in faces_rect:
                boxes.append([x, y, w, h])
                confidences.append(0.9)  # Assign default confidence
            
            # Convert to numpy arrays
            boxes = np.array(boxes)
            confidences = np.array(confidences)
            
            # Apply NMS
            indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), 0.5, 0.4)
            
            faces = []
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    # Convert to x, y, x1, y1 format
                    x1 = x + w
                    y1 = y + h
                    
                    faces.append({
                        'bbox': [x, y, x1, y1],
                        'confidence': 0.9  # Haar cascades don't provide confidence scores
                    })
            
            return faces
        except Exception as e:
            print(f"[ERROR] Face detection failed: {e}")
            return []
    
    def extract_face_features(self, frame, bbox):
        """Extract face features for recognition"""
        try:
            x, y, x1, y1 = bbox
            face_roi = frame[y:y1, x:x1]
            
            if face_roi.size == 0:
                return None
            
            # Convert to grayscale and resize
            face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            face_resized = cv2.resize(face_gray, (100, 100))
            
            return face_resized
        except Exception as e:
            print(f"[ERROR] Feature extraction failed: {e}")
            return None
    
    def load_known_faces(self, dataset_dir):
        """Load known faces from dataset directory with support for multiple images per person"""
        if not os.path.exists(dataset_dir):
            print(f"[WARNING] Dataset directory not found: {dataset_dir}")
            return
        
        faces = []
        labels = []
        self.known_names = []
        self.person_to_label = {}
        self.person_face_counts = {}
        
        label_id = 0
        
        for filename in os.listdir(dataset_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(dataset_dir, filename)
                
                # Extract person name (handle multiple images with _1, _2, etc.)
                base_name = os.path.splitext(filename)[0]
                if '_' in base_name and base_name.split('_')[-1].isdigit():
                    # Handle names like "John_1.jpg", "John_2.jpg"
                    person_name = '_'.join(base_name.split('_')[:-1])
                else:
                    person_name = base_name
                
                try:
                    # Load image
                    image = cv2.imread(filepath)
                    if image is None:
                        continue
                    
                    # Detect faces in the image
                    detected_faces = self.detect_faces(image)
                    
                    if detected_faces:
                        # Use the first detected face
                        face_bbox = detected_faces[0]['bbox']
                        face_features = self.extract_face_features(image, face_bbox)
                        
                        if face_features is not None:
                            # Check if we've seen this person before
                            if person_name not in self.person_to_label:
                                self.person_to_label[person_name] = label_id
                                self.known_names.append(person_name)
                                self.known_faces[person_name] = label_id
                                self.person_face_counts[person_name] = 0
                                label_id += 1
                            
                            # Add face with the person's label
                            faces.append(face_features)
                            labels.append(self.person_to_label[person_name])
                            self.person_face_counts[person_name] += 1
                            
                            print(f"[INFO] Loaded face #{self.person_face_counts[person_name]} for: {person_name}")
                
                except Exception as e:
                    print(f"[WARNING] Could not load {filename}: {e}")
        
        # Train the recognizer if we have faces
        if faces and self.face_recognizer is not None:
            try:
                self.face_recognizer.train(faces, np.array(labels))
                total_images = len(faces)
                total_people = len(self.person_face_counts)
                print(f"[INFO] Face recognizer trained with {total_images} images from {total_people} people")
                
                # Print training summary
                for person, count in self.person_face_counts.items():
                    print(f"[INFO] {person}: {count} training image(s)")
                
                # Save the trained model
                self.save_model()
            except Exception as e:
                print(f"[ERROR] Failed to train recognizer: {e}")
        
        return len(faces)
    
    def add_face_image_to_person(self, person_name, image_path):
        """Add additional training image for an existing person"""
        try:
            # Load the new image
            image = cv2.imread(image_path)
            if image is None:
                print(f"[ERROR] Could not load image: {image_path}")
                return False
            
            # Detect faces in the image
            detected_faces = self.detect_faces(image)
            
            if not detected_faces:
                print(f"[ERROR] No face detected in image: {image_path}")
                return False
            
            # Extract face features
            face_bbox = detected_faces[0]['bbox']
            face_features = self.extract_face_features(image, face_bbox)
            
            if face_features is None:
                print(f"[ERROR] Could not extract face features from: {image_path}")
                return False
            
            # Check if person exists
            if person_name not in self.person_to_label:
                print(f"[ERROR] Person {person_name} not found in known faces")
                return False
            
            # Add the new face to training data
            # We need to retrain with all existing faces plus this new one
            # For now, we'll save the image to dataset and reload
            
            # Find next available image number
            dataset_dir = os.path.dirname(image_path)
            if not dataset_dir:
                dataset_dir = "dataset"
            
            existing_count = self.person_face_counts.get(person_name, 0)
            new_filename = f"{person_name}_{existing_count + 1}.jpg"
            new_path = os.path.join(dataset_dir, new_filename)
            
            # Copy the image to the dataset directory
            import shutil
            shutil.copy2(image_path, new_path)
            
            print(f"[INFO] Added new training image for {person_name}: {new_filename}")
            
            # Reload and retrain the model
            self.load_known_faces(dataset_dir)
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to add face image: {e}")
            return False
    
    def recognize_face(self, frame, bbox):
        """Recognize a face in the given bounding box"""
        try:
            face_features = self.extract_face_features(frame, bbox)
            if face_features is None:
                return "Unknown Person", 0.0
            
            if self.face_recognizer is not None and self.known_names:
                # Use LBPH recognizer
                label, confidence = self.face_recognizer.predict(face_features)
                
                # Convert confidence to percentage (lower is better for LBPH)
                confidence_percentage = max(0, 100 - confidence)
                
                # More lenient recognition - if we have a match and reasonable confidence
                if label < len(self.known_names) and confidence < self.recognition_threshold:
                    return self.known_names[label], confidence_percentage
                else:
                    return "Unknown Person", confidence_percentage
            else:
                # Fallback to template matching
                return self._template_matching(face_features)
        
        except Exception as e:
            print(f"[ERROR] Face recognition failed: {e}")
            return "Unknown Person", 0.0
    
    def _template_matching(self, face_features):
        """Fallback template matching method"""
        if not self.known_faces:
            return "Unknown Person", 0.0
        
        # This is a simple template matching fallback
        # In a real implementation, you'd store template features
        return "Unknown Person", 0.0
    
    def save_model(self, filepath="face_recognition_model.yml"):
        """Save the trained face recognition model"""
        try:
            if self.face_recognizer is not None:
                self.face_recognizer.save(filepath)
                
                # Save the names mapping
                with open("face_names.json", "w") as f:
                    json.dump({
                        "names": self.known_names,
                        "faces": self.known_faces
                    }, f)
                
                print(f"[INFO] Face recognition model saved to {filepath}")
        except Exception as e:
            print(f"[ERROR] Failed to save model: {e}")
    
    def load_model(self, filepath="face_recognition_model.yml"):
        """Load a pre-trained face recognition model"""
        try:
            if os.path.exists(filepath) and self.face_recognizer is not None:
                self.face_recognizer.read(filepath)
                
                # Load the names mapping
                if os.path.exists("face_names.json"):
                    with open("face_names.json", "r") as f:
                        data = json.load(f)
                        self.known_names = data.get("names", [])
                        self.known_faces = data.get("faces", {})
                
                print(f"[INFO] Face recognition model loaded from {filepath}")
                return True
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
        
        return False
    
    def is_available(self):
        """Check if face recognition is available"""
        return self.face_cascade is not None
    
    def get_face_encoding(self, frame, bbox):
        """Get face encoding (for compatibility with existing code)"""
        face_features = self.extract_face_features(frame, bbox)
        if face_features is not None:
            # Flatten the face features to create an "encoding"
            return face_features.flatten().tolist()
        return None
    
    def compare_faces(self, known_encodings, face_encoding, tolerance=0.6):
        """Compare face encodings (for compatibility)"""
        if not known_encodings or face_encoding is None:
            return []
        
        # Simple distance-based comparison for compatibility
        face_encoding = np.array(face_encoding)
        matches = []
        
        for known_encoding in known_encodings:
            known_encoding = np.array(known_encoding)
            distance = np.linalg.norm(known_encoding - face_encoding)
            matches.append(distance < tolerance * 1000)  # Scale tolerance
        
        return matches 