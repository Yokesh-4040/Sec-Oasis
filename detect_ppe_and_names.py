import os
import cv2
import csv
import face_recognition
from datetime import datetime
from ultralytics import YOLO

# Paths
image_dir = "test_images"
employee_dir = "dataset"
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)
log_file = os.path.join(output_dir, "ppe_face_log.csv")

# Load YOLO model (custom trained for PPE)
model = YOLO("best.pt")

# Define PPE items you expect
required_ppe = {"Hardhat", "Mask", "Safety Vest"}

# Load known employee faces
known_encodings = []
known_names = []

for file in os.listdir(employee_dir):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        image = face_recognition.load_image_file(os.path.join(employee_dir, file))
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(os.path.splitext(file)[0])
        else:
            print(f"[!] No face encoding found in: {file}")

# Prepare CSV log
with open(log_file, "w", newline="") as logfile:
    fieldnames = ["timestamp", "image", "person", "detected_objects", "counts", "missing_ppe"]
    writer = csv.DictWriter(logfile, fieldnames=fieldnames)
    writer.writeheader()

    for image_name in os.listdir(image_dir):
        if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(image_dir, image_name)
        image = cv2.imread(img_path)
        if image is None:
            continue

        # --- FACE RECOGNITION ---
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb)
        face_encodings = face_recognition.face_encodings(rgb, face_locations)

        recognized_people = []

        for encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.6)
            name = "Unknown"
            if True in matches:
                best_idx = matches.index(True)
                name = known_names[best_idx]
            recognized_people.append(name)
            cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 255), 2)
            cv2.putText(image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        # --- PPE DETECTION ---
        results = model.predict(image, verbose=False)
        boxes = results[0].boxes
        names = model.names

        detected_labels = [names[int(cls)] for cls in boxes.cls]
        counts = {label: detected_labels.count(label) for label in set(detected_labels)}

        present_ppe = {label for label in detected_labels if label in required_ppe}
        missing_ppe = list(required_ppe - present_ppe)

        # Draw PPE detection boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            label = names[class_id]
            color = (0, 255, 0) if label in required_ppe else (0, 0, 255)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Save annotated image
        cv2.imwrite(os.path.join(output_dir, image_name), image)

        # Write log entry
        writer.writerow({
            "timestamp": datetime.now().isoformat(),
            "image": image_name,
            "person": ", ".join(recognized_people) if recognized_people else "Unknown",
            "detected_objects": ", ".join(detected_labels),
            "counts": str(counts),
            "missing_ppe": ", ".join(missing_ppe) if missing_ppe else "None"
        })
