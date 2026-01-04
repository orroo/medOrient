import cv2
import mediapipe as mp
import os
import csv
import pandas as pd
import numpy as np

# ===============================
# Paths
# ===============================
frames_dir = r"C:\Users\Lenovo-Thinkpad\Desktop\4AI\projet s1\strokedetection\dataimages\datasets of images\face"
input_csv = r"C:\Users\Lenovo-Thinkpad\Desktop\4AI\projet s1\strokedetection\face\stroke_dataset.csv"
output_csv = r"C:\Users\Lenovo-Thinkpad\Desktop\4AI\projet s1\strokedetection\face\stroke_dataset_landmarks.csv"

# ===============================
# Mediapipe setup
# ===============================
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ===============================
# Load CSV with image names
# ===============================
df = pd.read_csv(input_csv)

# ===============================
# Prepare new CSV with selected landmarks + labels
# ===============================
header = ["frame",
          "nose_x", "nose_y", "nose_z",
          "left_mouth_x", "left_mouth_y", "left_mouth_z",
          "right_mouth_x", "right_mouth_y", "right_mouth_z",
          "left_eye_x", "left_eye_y", "left_eye_z",
          "right_eye_x", "right_eye_y", "right_eye_z",
          "facial_droop_flag", "severity_score"]

with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)

# ===============================
# Helper functions
# ===============================
def find_image(filename, search_dir):
    """Search recursively for the file in subdirectories"""
    for root, _, files in os.walk(search_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None

def normalize_point(point, nose, w, h):
    """Convert to pixel coordinates relative to nose"""
    x = (point.x - nose.x) * w
    y = (point.y - nose.y) * h
    z = point.z - nose.z
    return x, y, z

# ===============================
# Process each image
# ===============================
for idx, row in df.iterrows():
    frame_file = row["frame"]
    frame_path = find_image(frame_file, frames_dir)
    
    if frame_path is None:
        print(f"Image not found: {frame_file}")
        continue

    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"Cannot read image: {frame_file}")
        continue

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if not results.multi_face_landmarks:
        print(f"No face detected: {frame_file}")
        continue

    lm = results.multi_face_landmarks[0].landmark

    # Select key landmarks (nose tip, mouth corners, eyes outer)
    key_indices = [1, 61, 291, 33, 263]
    key_points = [lm[i] for i in key_indices]

    # Normalize relative to nose in pixels
    nose = key_points[0]
    normalized = []
    for pt in key_points:
        normalized.extend(normalize_point(pt, nose, w, h))

    # Get labels from original CSV
    facial_droop_flag = row.get("facial_droop_flag", "")
    severity_score = row.get("severity_score", "")

    # Write to CSV
    with open(output_csv, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([frame_file] + normalized + [facial_droop_flag, severity_score])

print(f"Selected landmark extraction complete → {output_csv}")
