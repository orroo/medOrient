import cv2
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils  # optional, for visualization

import numpy as np
from tensorflow.keras.models import load_model
from collections import deque
import time
import os
from .description import explain_features

# -------------------------------
# Load model
# -------------------------------

MODEL_PATH = os.path.join(os.path.dirname(__file__), "stroke_model.h5")
model = load_model(MODEL_PATH, compile=False)
print("[INFO] Face model loaded successfully!")

import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils  # optional, for visualization

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
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

def is_face_centered(xmin, ymin, xmax, ymax, W, H, tol=0.3):
    cx, cy = (xmin+xmax)/2, (ymin+ymax)/2
    return abs(cx - W/2) < W*tol and abs(cy - H/2) < H*tol

def align_face_pixels(crop, lm_px):
    left = np.array(lm_px[33])
    right = np.array(lm_px[263])
    dy, dx = right[1] - left[1], right[0] - left[0]
    angle = np.degrees(np.arctan2(dy, dx))
    center = ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(crop, M, (crop.shape[1], crop.shape[0]))

# -------------------------------
# MAIN FUNCTION (IMPORTANT)
# -------------------------------
def run_face_inference():
    prediction_interval = 0.3
    severity_history = deque(maxlen=3)
    stable_threshold = 2.0
    stable_duration = 1.5

    last_pred_time = 0
    stable_start_time = None

    cap = cv2.VideoCapture(0)
    print("[INFO] Starting camera...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        display = frame.copy()
        now = time.time()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            cv2.putText(display, "No face detected", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow("Stroke Severity Detection", display)
            if cv2.waitKey(1) == 27:
                break
            continue

        lm = results.multi_face_landmarks[0].landmark
        x_min, y_min, x_max, y_max = get_face_bbox(lm, w, h)
        centered = is_face_centered(x_min, y_min, x_max, y_max, w, h)

        cv2.rectangle(display, (x_min,y_min), (x_max,y_max),
                      (0,255,0) if centered else (0,0,255), 2)

        if centered and now - last_pred_time >= prediction_interval:
            last_pred_time = now

            face_crop = frame[y_min:y_max, x_min:x_max]
            lm_px = [(int(p.x * w) - x_min, int(p.y * h) - y_min) for p in lm]
            aligned = align_face_pixels(face_crop, lm_px)

            img = cv2.resize(aligned, (224,224)) / 255.0
            img = np.expand_dims(img, 0).astype(np.float32)

            handcrafted, feats = get_handcrafted_features(lm, w, h)
            sev = model.predict([img, handcrafted], verbose=0)[0][0]

            severity_history.append(sev)
            disp = np.mean(severity_history)

            print(f"[PREDICTION] {sev:.2f} | Smoothed: {disp:.2f}")

            cv2.putText(display, f"Severity: {disp:.1f}",
                        (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0,0,255), 2)

            if len(severity_history) == severity_history.maxlen:
                if max(severity_history) - min(severity_history) <= stable_threshold:
                    if stable_start_time is None:
                        stable_start_time = now
                    elif now - stable_start_time >= stable_duration:
                        final_severity = float(np.mean(severity_history))
                        cap.release()
                        cv2.destroyAllWindows()

                        return {
                            "face_severity": final_severity,
                            "face_description": explain_features(feats, final_severity)
                        }
                else:
                    stable_start_time = None

        cv2.imshow("Stroke Severity Detection", display)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    return {
        "face_severity": None,
        "face_description": "Face analysis not completed"
    }
