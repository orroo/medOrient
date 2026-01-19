# ===============================
# Working Hybrid Severity Predictor (fixed normalization)
# ===============================

import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# -------------------------------
# 1. Paths
# -------------------------------
csv_path = r"C:\Users\Lenovo-Thinkpad\Desktop\4AI\projet s1\strokedetection\face\stroke_datasetwith calcul.csv"
image_dir = r"C:\Users\Lenovo-Thinkpad\Desktop\4AI\projet s1\strokedetection\dataimages\data\face\sorted_images\data_aligned"

# -------------------------------
# 2. Load CSV
# -------------------------------
df = pd.read_csv(csv_path)
num_cols = [
    'smile_vertical_asymmetry_norm', 'mouth_horizontal_asymmetry_norm',
    'eye_horizontal_asymmetry_norm', 'general_symmetry_score_norm',
    'severity_score'
]

for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=num_cols)

# ----- FIX: Normalize severity -----
scaler_y = StandardScaler()
targets = df['severity_score'].values.reshape(-1, 1)
targets_norm = scaler_y.fit_transform(targets).astype(np.float32)

handcrafted_features = df[
    ['smile_vertical_asymmetry_norm',
     'mouth_horizontal_asymmetry_norm',
     'eye_horizontal_asymmetry_norm',
     'general_symmetry_score_norm']
].values.astype(np.float32)

image_files = df['frame'].values

# -------------------------------
# 3. Load images
# -------------------------------
images = []
valid_indices = []

for idx, img_file in enumerate(image_files):
    path = None
    for folder in ['droop_0', 'droop_1']:
        temp_path = os.path.join(image_dir, folder, img_file)
        if os.path.exists(temp_path):
            path = temp_path
            break

    if path is None:
        continue

    img = cv2.imread(path)
    if img is None:
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    images.append(img)
    valid_indices.append(idx)

images = np.array(images, dtype=np.float32)
handcrafted_features = handcrafted_features[valid_indices]
targets_norm = targets_norm[valid_indices]

print(f"[INFO] Loaded {len(images)} images.")

# -------------------------------
# 4. Train/Val/Test split
# -------------------------------
X_img_train, X_img_temp, X_hand_train, X_hand_temp, y_train, y_temp = train_test_split(
    images, handcrafted_features, targets_norm, test_size=0.3, random_state=42
)

X_img_val, X_img_test, X_hand_val, X_hand_test, y_val, y_test = train_test_split(
    X_img_temp, X_hand_temp, y_temp, test_size=0.5, random_state=42
)

# -------------------------------
# 5. Augmentation
# -------------------------------
img_gen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

# -------------------------------
# 6. Hybrid generator
# -------------------------------
def hybrid_generator(imgs, features, targets, batch_size=16, augment=True):
    dataset_size = len(imgs)
    while True:
        idxs = np.random.choice(dataset_size, batch_size)
        batch_imgs = imgs[idxs]
        batch_features = features[idxs]
        batch_targets = targets[idxs]
        if augment:
            for i in range(batch_size):
                batch_imgs[i] = img_gen.random_transform(batch_imgs[i])
        yield {"Image_Input": batch_imgs, "Handcrafted_Input": batch_features}, batch_targets

# -------------------------------
# 7. Build Model
# -------------------------------
img_input = Input(shape=(224, 224, 3), name="Image_Input")
x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(img_input)
x = layers.MaxPooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D()(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
cnn_features = layers.Dense(64, activation='relu')(x)

hand_input = Input(shape=(handcrafted_features.shape[1],), name="Handcrafted_Input")
y = layers.Dense(32, activation='relu')(hand_input)
y = layers.Dropout(0.2)(y)
hand_features = layers.Dense(16, activation='relu')(y)

merged = layers.concatenate([cnn_features, hand_features])
z = layers.Dense(64, activation='relu')(merged)
z = layers.Dropout(0.2)(z)
z = layers.Dense(32, activation='relu')(z)
output = layers.Dense(1, activation='linear')(z)

model = Model(inputs=[img_input, hand_input], outputs=output)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# -------------------------------
# 8. Callbacks
# -------------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# -------------------------------
# 9. Train
# -------------------------------
batch_size = 16
steps_per_epoch = len(X_img_train) // batch_size

history = model.fit(
    hybrid_generator(X_img_train, X_hand_train, y_train, batch_size=batch_size),
    validation_data=([X_img_val, X_hand_val], y_val),
    steps_per_epoch=steps_per_epoch,
    epochs=50,
    callbacks=[early_stop, reduce_lr]
)

# -------------------------------
# 10. Save Model + Scaler
# -------------------------------
model.save("stroke_model_2.h5")
import joblib
joblib.dump(scaler_y, "severity_scaler.pkl")
print("[INFO] Model + scaler saved")

# -------------------------------
# 11. Evaluate (denormalize output)
# -------------------------------
loss, mae = model.evaluate([X_img_test, X_hand_test], y_test)
print("[INFO] Test MAE (normalized):", mae)

# Denormalize MAE
preds = model.predict([X_img_test, X_hand_test])
preds_real = scaler_y.inverse_transform(preds)
y_test_real = scaler_y.inverse_transform(y_test)

real_mae = np.mean(np.abs(preds_real - y_test_real))
print(f"[INFO] Test MAE (real scale): {real_mae:.2f}")
