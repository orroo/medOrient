import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import os
from .description import explain_features

# -------------------------------
# Load model
# -------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "stroke_model.h5")
model = load_model(MODEL_PATH, compile=False)
print("[INFO] Model loaded successfully!")

# -------------------------------
# Mediapipe Face Mesh
# -------------------------------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=True,  # True for images
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7
)

# -------------------------------
# Helper functions
# -------------------------------
def to_px(lm, w, h):
    return np.array([int(lm.x * w), int(lm.y * h)])

def compute_general_symmetry(landmarks, w, h):
    nose = to_px(landmarks[1], w, h)
    diffs = []
    pairs = [(33, 263), (61, 291), (362, 133), (234, 454), (93, 323)]
    for l_i, r_i in pairs:
        left = to_px(landmarks[l_i], w, h)
        right = to_px(landmarks[r_i], w, h)
        diffs.append(abs(abs(left[0]-nose[0]) - abs(right[0]-nose[0])))
    return np.mean(diffs)

def get_handcrafted_features(lm, w, h):
    nose = to_px(lm[1], w, h)
    left_m = to_px(lm[61], w, h)
    right_m = to_px(lm[291], w, h)
    left_e = to_px(lm[33], w, h)
    right_e = to_px(lm[263], w, h)

    iod = np.linalg.norm(left_e - right_e)
    smile_v = abs(left_m[1] - right_m[1]) / iod
    mouth_h = abs(abs(left_m[0] - nose[0]) - abs(right_m[0] - nose[0])) / iod
    eye_h = abs(abs(left_e[0] - nose[0]) - abs(right_e[0] - nose[0])) / iod
    general_sym = compute_general_symmetry(lm, w, h) / iod

    return np.array([[smile_v, mouth_h, eye_h, general_sym]], dtype=np.float32), {
        "smile_vertical_asymmetry": smile_v,
        "mouth_horizontal_asymmetry": mouth_h,
        "eye_horizontal_asymmetry": eye_h,
        "general_facial_symmetry": general_sym
    }

def get_face_bbox(lm, w, h, scale=1.2):
    xs = [int(p.x * w) for p in lm]
    ys = [int(p.y * h) for p in lm]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    box = max(x_max - x_min, y_max - y_min)
    cx, cy = (x_min + x_max)//2, (y_min + y_max)//2
    half = int(box * scale // 2)
    return max(0, cx - half), max(0, cy - half), min(w, cx + half), min(h, cy + half)

def align_face_pixels(crop, lm_px):
    left = np.array(lm_px[33])
    right = np.array(lm_px[263])
    dy, dx = right[1] - left[1], right[0] - left[0]
    angle = np.degrees(np.arctan2(dy, dx))
    center = ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(crop, M, (crop.shape[1], crop.shape[0]))

# -------------------------------
# Function to predict severity from an image
# -------------------------------
def predict_image(image_path):
    """Run face severity prediction on an image file.

    Returns: dict with keys `face_severity` (float) and `face_description` (str).
    """
    img = cv2.imread(image_path)
    if img is None:
        return {"face_severity": None, "face_description": "Could not read image."}

    h, w, _ = img.shape
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return {"face_severity": None, "face_description": "No face detected."}

    lm = results.multi_face_landmarks[0].landmark
    x_min, y_min, x_max, y_max = get_face_bbox(lm, w, h)
    face_crop = img[y_min:y_max, x_min:x_max]
    lm_px = [(int(p.x * w) - x_min, int(p.y * h) - y_min) for p in lm]
    aligned = align_face_pixels(face_crop, lm_px)

    resized = cv2.resize(aligned, (224,224)) / 255.0
    resized = np.expand_dims(resized, 0).astype(np.float32)

    handcrafted, feats = get_handcrafted_features(lm, w, h)
    severity = float(model.predict([resized, handcrafted])[0][0])

    return {
        "face_severity": severity,
        "face_description": explain_features(feats, severity)
    }

# -------------------------------
# Example usage
# -------------------------------
# Module provides `predict_image(image_path)` for server-side usage.
