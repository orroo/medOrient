import cv2
import mediapipe as mp
import os
import numpy as np
from tqdm import tqdm

# Paths
input_dirs = [
    r"C:\Users\Lenovo-Thinkpad\Desktop\4AI\projet s1\strokedetection\dataimages\data\face\sorted_images\droop_1",
    r"C:\Users\Lenovo-Thinkpad\Desktop\4AI\projet s1\strokedetection\dataimages\data\face\sorted_images\droop_0"
]
output_base = r"C:\Users\Lenovo-Thinkpad\Desktop\4AI\projet s1\strokedetection\dataimages\data\face\sorted_images\data_aligned"
output_size = (224, 224)  # CNN input size

# Mediapipe setup
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Helper functions
def to_px(lm, w, h):
    return np.array([lm.x * w, lm.y * h])

def align_face(frame, left_eye, right_eye):
    eyes_center = ((left_eye[0] + right_eye[0]) / 2,
                   (left_eye[1] + right_eye[1]) / 2)
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))
    M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
    aligned = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]),
                             flags=cv2.INTER_CUBIC)
    return aligned

# Process images with counters
for input_dir in input_dirs:
    label = os.path.basename(input_dir)  # droop_1 or droop_0
    output_dir = os.path.join(output_base, label)
    os.makedirs(output_dir, exist_ok=True)

    count_processed = 0  # counter for successful images

    for img_file in tqdm(os.listdir(input_dir), desc=f"Processing {label}"):
        img_path = os.path.join(input_dir, img_file)
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            continue

        lm = results.multi_face_landmarks[0].landmark
        left_eye = to_px(lm[33], w, h)   # left eye outer
        right_eye = to_px(lm[263], w, h) # right eye outer

        aligned = align_face(frame, left_eye, right_eye)
        resized = cv2.resize(aligned, output_size)

        save_path = os.path.join(output_dir, img_file)
        cv2.imwrite(save_path, resized)
        count_processed += 1  # increment counter

    print(f"Total images processed for {label}: {count_processed}")

print("All images aligned and resized!")
