"""
Face Recognition Integration with existing detect_ppe_and_names.py system
"""
import os
import cv2
import face_recognition
import numpy as np
from pathlib import Path

# Configuration from your existing system
EMPLOYEE_DIR = "dataset"
RESULTS_DIR = "results"
UNKNOWN_DIR = os.path.join(RESULTS_DIR, "unknown_people")

# Ensure directories exist
os.makedirs(EMPLOYEE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(UNKNOWN_DIR, exist_ok=True)

class FaceRecognitionSystem:
    """Integration with your existing face recognition system"""
    
    def __init__(self):
        self.known_encodings = []
        self.known_names = []
        self.load_known_faces()
    
    def load_known_faces(self):
        """Load known employee faces from dataset directory"""
        self.known_encodings = []
        self.known_names = []
        
        dataset_path = Path(EMPLOYEE_DIR)
        if not dataset_path.exists():
            print(f"[WARNING] Dataset directory not found: {EMPLOYEE_DIR}")
            return
        
        for file_path in dataset_path.glob("*"):
            if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                try:
                    # Load image and encode face
                    image = face_recognition.load_image_file(str(file_path))
                    encodings = face_recognition.face_encodings(image)
                    
                    if encodings:
                        self.known_encodings.append(encodings[0])
                        # Extract name from filename (without extension)
                        name = file_path.stem
                        self.known_names.append(name)
                        print(f"[INFO] Loaded face encoding for: {name}")
                    else:
                        print(f"[WARNING] No face found in: {file_path}")
                        
                except Exception as e:
                    print(f"[ERROR] Failed to load {file_path}: {e}")
    
    def recognize_face(self, face_encoding, tolerance=0.6):
        """
        Recognize a face encoding against known faces
        
        Args:
            face_encoding: Face encoding array
            tolerance: Recognition tolerance (lower = stricter)
            
        Returns:
            str: Employee name if recognized, "Unknown" otherwise
        """
        if not self.known_encodings:
            return "Unknown"
        
        # Compare with known faces
        matches = face_recognition.compare_faces(
            self.known_encodings, 
            face_encoding, 
            tolerance=tolerance
        )
        
        if True in matches:
            # Find the best match
            best_match_index = matches.index(True)
            return self.known_names[best_match_index]
        
        return "Unknown"
    
    def recognize_face_from_image(self, image_path_or_array, tolerance=0.6):
        """
        Recognize faces from an image file or numpy array
        
        Args:
            image_path_or_array: Path to image file or numpy array
            tolerance: Recognition tolerance
            
        Returns:
            list: List of recognized names
        """
        try:
            # Load image
            if isinstance(image_path_or_array, str):
                image = face_recognition.load_image_file(image_path_or_array)
            else:
                image = image_path_or_array
            
            # Find face locations and encodings
            face_locations = face_recognition.face_locations(image)
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            recognized_people = []
            for encoding in face_encodings:
                name = self.recognize_face(encoding, tolerance)
                recognized_people.append(name)
            
            return recognized_people
            
        except Exception as e:
            print(f"[ERROR] Face recognition failed: {e}")
            return []
    
    def add_new_employee_face(self, employee_name, image_data):
        """
        Add a new employee face to the dataset
        
        Args:
            employee_name: Name of the employee
            image_data: Image data (numpy array or file path)
            
        Returns:
            bool: Success status
        """
        try:
            # Prepare file path
            safe_name = "".join(c for c in employee_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            file_path = os.path.join(EMPLOYEE_DIR, f"{safe_name}.jpg")
            
            # Handle different image data types
            if isinstance(image_data, str):
                # Copy from existing file
                import shutil
                shutil.copy2(image_data, file_path)
            elif isinstance(image_data, np.ndarray):
                # Save numpy array as image
                cv2.imwrite(file_path, image_data)
            else:
                print(f"[ERROR] Unsupported image data type: {type(image_data)}")
                return False
            
            # Reload known faces to include the new employee
            self.load_known_faces()
            
            print(f"[INFO] Added new employee face: {employee_name}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to add employee face: {e}")
            return False
    
    def save_unknown_face(self, image_data, face_location):
        """
        Save an unknown face for later identification
        
        Args:
            image_data: Full image as numpy array
            face_location: Tuple of (top, right, bottom, left)
            
        Returns:
            str: Path to saved unknown face image
        """
        try:
            from datetime import datetime
            
            # Extract face from full image
            top, right, bottom, left = face_location
            face_image = image_data[top:bottom, left:right]
            
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"unknown_{timestamp}.jpg"
            file_path = os.path.join(UNKNOWN_DIR, filename)
            
            # Save face image
            cv2.imwrite(file_path, face_image)
            
            print(f"[INFO] Saved unknown face: {filename}")
            return file_path
            
        except Exception as e:
            print(f"[ERROR] Failed to save unknown face: {e}")
            return None
    
    def get_employee_stats(self):
        """Get statistics about loaded employees"""
        return {
            'total_employees': len(self.known_names),
            'employee_names': self.known_names.copy(),
            'dataset_directory': EMPLOYEE_DIR,
            'unknown_faces_directory': UNKNOWN_DIR
        }

# Global instance for use across the application
face_recognition_system = FaceRecognitionSystem()

def recognize_employee_from_camera():
    """
    Capture and recognize employee from camera feed
    This is a simplified version - you can extend it with your camera integration
    """
    try:
        # This would integrate with your camera system
        # For now, return simulation
        return face_recognition_system.get_employee_stats()['employee_names']
    except Exception as e:
        print(f"[ERROR] Camera recognition failed: {e}")
        return []

def integrate_with_existing_system():
    """
    Integration point with your existing detect_ppe_and_names.py system
    This function can be called to sync with your current setup
    """
    try:
        # Reload faces from dataset
        face_recognition_system.load_known_faces()
        
        # Get current stats
        stats = face_recognition_system.get_employee_stats()
        
        print(f"[INFO] Face recognition system initialized")
        print(f"[INFO] Loaded {stats['total_employees']} employee faces")
        print(f"[INFO] Employee names: {', '.join(stats['employee_names'])}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Integration failed: {e}")
        return False

# Initialize the system when module is imported
if __name__ == "__main__":
    integrate_with_existing_system() 