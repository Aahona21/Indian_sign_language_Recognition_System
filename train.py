import os
import cv2
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import collections
import tensorflow as tf

# Import necessary Keras/TensorFlow components
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Import scikit-learn components
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score # For more granular metrics

# Set GPU device and image size
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sz = 128

# Define the path to your dataset
TRAIN_DIR = 'data/train'
TEST_DIR = 'data/test'

# --- Verify Dataset Paths ---
if not os.path.exists(TRAIN_DIR):
    print(f"Error: Training directory not found at {TRAIN_DIR}")
    exit()
if not os.path.exists(TEST_DIR):
    print(f"Error: Test directory not found at {TEST_DIR}")
    exit()
print(f"Train directory: {TRAIN_DIR}")
print(f"Test directory: {TEST_DIR}")


# === Data Augmentation ===
# ENHANCED DATA AUGMENTATION FOR PROBLEM CLASSES (M-N, O-N, R-Q, U-V, A-S, P-K)
# Note: You can experiment with these ranges. Sometimes *too* aggressive augmentation
# can make it harder for the model to learn. If initial results are very poor,
# try slightly reducing these ranges first (e.g., shear_range=0.1, rotation_range=15, etc.)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.25,
    zoom_range=0.25,
    rotation_range=30,
    horizontal_flip=True,
    width_shift_range=0.3,
    height_shift_range=0.3,
    brightness_range=[0.6, 1.4],
    fill_mode='nearest',
    channel_shift_range=0.15,
)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=(sz, sz), batch_size=32,
    color_mode='rgb', class_mode='categorical'
)
test_set = test_datagen.flow_from_directory(
    TEST_DIR, target_size=(sz, sz), batch_size=32,
    color_mode='rgb', class_mode='categorical', shuffle=False
)

class_indices = training_set.class_indices
reverse_class_indices = {v: k for k, v in class_indices.items()}

print("\n--- Class Indices Mapping ---")
print(class_indices)
print(f"Total classes found: {len(class_indices)}")

# Sanity Check: You can optionally display some augmented images to ensure they look correct.
# first_batch_images, first_batch_labels = next(training_set)
# plt.figure(figsize=(10, 10))
# for i in range(9):
#     plt.subplot(3, 3, i + 1)
#     plt.imshow(first_batch_images[i])
#     plt.title(f"Class: {reverse_class_indices[np.argmax(first_batch_labels[i])]}")
#     plt.axis('off')
# plt.tight_layout()
# plt.show()


# === Primary Model (MobileNetV2) with Fine-tuning Strategy ===
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(sz, sz, 3))

# Phase 1: Train only the top layers (classifier head)
base_model.trainable = False # Freeze the base model for the first phase

classifier = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(class_indices), activation='softmax') # Use len(class_indices) for flexibility
])
# Explicit learning rate for Adam optimizer in Phase 1
classifier.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                   loss='categorical_crossentropy', metrics=['accuracy'])

# === Verification Model (MobileNetV2) ===
verification_base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(sz, sz, 3))
verification_base.trainable = False

verification_model = Sequential([
    verification_base,
    GlobalAveragePooling2D(),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(len(class_indices), activation='softmax') # Use len(class_indices)
])
verification_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), # Explicit LR
                          loss='categorical_crossentropy', metrics=['accuracy'])

print("--- Initial Primary Classifier Model Summary (Frozen Base) ---")
classifier.summary()

# === Callbacks ===
# Increased patience for EarlyStopping and ReduceLROnPlateau
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=7, factor=0.5, min_lr=1e-7, verbose=1) # Adjusted min_lr slightly

# === Train Models (Phase 1: Train Head Only) ===
print("\n--- Phase 1: Training Primary Classifier Head (Base Frozen) ---")
history_phase1 = classifier.fit(
    training_set,
    epochs=150, # Can increase epochs, EarlyStopping will manage
    steps_per_epoch=len(training_set),
    validation_data=test_set,
    validation_steps=len(test_set),
    callbacks=[early_stopping, reduce_lr]
)

# === Phase 2: Fine-tuning MobileNetV2 Base ===
base_model.trainable = True

# Unfreeze layers from a specific point. MobileNetV2 has around 155-156 layers.
# Layer 120 means unfreezing the last ~35 layers. Adjust based on performance.
fine_tune_from_layer = 120
for layer in base_model.layers[:fine_tune_from_layer]:
    layer.trainable = False

print(f"\n--- Phase 2: Fine-tuning Primary Classifier (Unfreezing layers from index {fine_tune_from_layer}) ---")
print("--- Primary Classifier Model Summary (Fine-tuning) ---")
classifier.summary() # See which layers are now trainable

classifier.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # Very low LR for fine-tuning
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

history_phase2 = classifier.fit(
    training_set,
    epochs=150, # Additional epochs for fine-tuning, EarlyStopping will manage
    steps_per_epoch=len(training_set),
    validation_data=test_set,
    validation_steps=len(test_set),
    callbacks=[early_stopping, reduce_lr]
)

# Train Verification Model
test_set.reset()
print("\n--- Training Verification Model ---")
# Use a separate EarlyStopping for verification model if desired, or same one.
verification_model.fit(
    training_set,
    epochs=150, # Can be fewer epochs if it converges quickly, EarlyStopping manages
    steps_per_epoch=len(training_set),
    validation_data=test_set,
    validation_steps=len(test_set),
    callbacks=[early_stopping] # Using the same early_stopping callback
)


# === Save Models & Weights ===
print("\n--- Saving Models and Weights ---")
# Ensure the 'model' directory exists
os.makedirs('model', exist_ok=True)

classifier.save_weights('model/model-primary_optimized.weights.h5')
verification_model.save_weights('model/model-verification_optimized.weights.h5')

model_json = classifier.to_json()
with open("model/model-primary_optimized.json", "w") as json_file:
    json_file.write(model_json)

verification_json = verification_model.to_json()
with open("model/model-verification_optimized.json", "w") as json_file:
    json_file.write(verification_json)

print("Models saved successfully in the 'model/' directory.")


# === Primary Model Predictions and Feature Extraction ===
print("\n--- Making Primary Model Predictions ---")
test_set.reset()
primary_probs = classifier.predict(test_set, steps=len(test_set), verbose=1)
primary_preds = np.argmax(primary_probs, axis=1)
true_labels = test_set.classes
class_labels = list(class_indices.keys()) # Use the confirmed class_indices keys

print("\n--- Collecting all test images and extracting features for re-verification and KNN ---")
all_test_images = []
test_set.reset() # Reset again for image collection
for i in range(len(test_set)):
    batch_images, _ = test_set[i]
    all_test_images.append(batch_images)
all_test_images = np.concatenate(all_test_images, axis=0)
assert all_test_images.shape[0] == len(test_set.filenames), \
    f"Mismatch in collected images ({all_test_images.shape[0]}) and total files ({len(test_set.filenames)})!"
print(f"Collected {all_test_images.shape[0]} test images.")

feature_extractor = Sequential(classifier.layers[0:2]) # base_model + GlobalAveragePooling2D
feature_extractor.compile(optimizer='adam', loss='mse') # Recompile feature extractor
print("\n--- Extracting features for KNN ---")
test_set.reset() # Reset for feature extraction
test_features = feature_extractor.predict(test_set, steps=len(test_set), verbose=1)

scaler = StandardScaler()
scaled_test_features = scaler.fit_transform(test_features)


# === Confidence-Based Verification + KNN Post-Prediction Filtering ===
confidence_threshold = 0.85 # Keep this consistent with the Streamlit app

problematic_pairs_names = [
    ('M', 'N'), ('N', 'M'),
    ('O', 'N'), ('N', 'O'),
    ('R', 'Q'), ('Q', 'R'),
    ('U', 'V'), ('V', 'U'),
    ('A', 'S'), ('S', 'A'),
    ('P', 'K'), ('K', 'P'),
    ('C', 'R'), ('R', 'C'),
    ('B', 'R'), ('R', 'B'),
    ('Q', 'B'), ('B', 'Q'),
    ('P', 'D'), ('D', 'P'),
    ('T', 'L'), ('L', 'T'),
    ('U', 'G'), ('G', 'U')

]
# Ensure problematic_indices are derived from the *actual* class_indices
problematic_indices = []
for a, b in problematic_pairs_names:
    if a in class_indices and b in class_indices:
        problematic_indices.append((class_indices[a], class_indices[b]))
    else:
        print(f"Warning: Problematic pair {a}-{b} includes a class not found in `class_indices`. Skipping this pair.")

n_neighbors_knn = 5 # Tunable hyperparameter for KNN

final_corrected_labels = list(primary_preds.copy())
print("\n--- Performing Confidence-Based Verification and KNN Post-Filtering ---")
rechecked_by_secondary_count = 0
corrected_by_knn_count = 0

for idx, prob in enumerate(primary_probs):
    confidence_score = np.max(prob)
    predicted_label_idx = np.argmax(prob)

    current_image = np.expand_dims(all_test_images[idx], axis=0)
    current_feature = scaled_test_features[idx].reshape(1, -1)

    if confidence_score < confidence_threshold:
        rechecked_by_secondary_count += 1
        secondary_prob = verification_model.predict(current_image, verbose=0)
        secondary_label_idx = np.argmax(secondary_prob)

        # Check if primary and secondary predictions are problematic pair or secondary is better
        needs_knn_or_secondary_override = False
        if secondary_label_idx != predicted_label_idx:
            needs_knn_or_secondary_override = True # Secondary model proposes a different class
        else: # Primary and secondary agree, but on a problematic class (e.g., both say 'M', but it's easily confused with 'N')
            for p1, p2 in problematic_indices:
                if (predicted_label_idx == p1 and secondary_label_idx == p1) and \
                   (p1 in [pair[0] for pair in problematic_indices] or p1 in [pair[1] for pair in problematic_indices]):
                    needs_knn_or_secondary_override = True # Agreed on a problematic class, still check KNN
                    break

        if needs_knn_or_secondary_override:
            knn_model = NearestNeighbors(n_neighbors=n_neighbors_knn, metric='euclidean')
            knn_model.fit(scaled_test_features, primary_preds) # Fit KNN on features and primary model's labels
            distances, indices = knn_model.kneighbors(current_feature)
            neighbor_labels = primary_preds[indices.flatten()] # Labels of nearest neighbors

            votes = collections.Counter(neighbor_labels)
            knn_decision_label_idx = votes.most_common(1)[0][0]

            # Decision logic:
            # 1. If KNN provides a different, better-supported label for a problematic pair, use KNN.
            # 2. Otherwise, if secondary model proposed a different label, use secondary.
            # 3. Else, stick with primary (or agreed-upon secondary).

            # Check if KNN decision aligns with problematic pair and differs from primary/secondary
            is_knn_correcting_problematic_pair = False
            for p1, p2 in problematic_indices:
                if (predicted_label_idx == p1 and knn_decision_label_idx == p2) or \
                   (predicted_label_idx == p2 and knn_decision_label_idx == p1) or \
                   (secondary_label_idx == p1 and knn_decision_label_idx == p2) or \
                   (secondary_label_idx == p2 and knn_decision_label_idx == p1):
                    if knn_decision_label_idx != predicted_label_idx: # Ensure KNN is actually changing something
                        final_corrected_labels[idx] = knn_decision_label_idx
                        corrected_by_knn_count += 1
                        is_knn_correcting_problematic_pair = True
                        break

            if not is_knn_correcting_problematic_pair and secondary_label_idx != predicted_label_idx:
                # If KNN didn't correct a problematic pair, but secondary model disagreed with primary
                final_corrected_labels[idx] = secondary_label_idx
            # If KNN didn't correct, and secondary model agreed, then primary_preds[idx] remains (no change)

        # If needs_knn_or_secondary_override was False, it means primary was low confidence but secondary agreed
        # and it wasn't a problematic pair agreement, so it keeps the original primary_preds[idx]
    else: # Primary model had high confidence, no re-check needed
        final_corrected_labels[idx] = predicted_label_idx

print(f"Total predictions re-checked by secondary model: {rechecked_by_secondary_count}")
print(f"Total predictions potentially corrected by KNN: {corrected_by_knn_count}")

final_corrected_labels = np.array(final_corrected_labels)

# === Plotting Training History ===
# This will show the history for Phase 1 and Phase 2.
plt.figure(figsize=(14, 5))

# Primary Model Accuracy
plt.subplot(1, 2, 1)
plt.plot(history_phase1.history['accuracy'], label='Train Acc (P1)')
plt.plot(history_phase1.history['val_accuracy'], label='Val Acc (P1)')
if history_phase2:
    # Concatenate histories for a continuous plot
    combined_accuracy = history_phase1.history['accuracy'] + history_phase2.history['accuracy']
    combined_val_accuracy = history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy']
    plt.plot(combined_accuracy, label='Combined Train Acc')
    plt.plot(combined_val_accuracy, label='Combined Val Acc')
plt.title('Primary Model Accuracy History')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Primary Model Loss
plt.subplot(1, 2, 2)
plt.plot(history_phase1.history['loss'], label='Train Loss (P1)')
plt.plot(history_phase1.history['val_loss'], label='Val Loss (P1)')
if history_phase2:
    combined_loss = history_phase1.history['loss'] + history_phase2.history['loss']
    combined_val_loss = history_phase1.history['val_loss'] + history_phase2.history['val_loss']
    plt.plot(combined_loss, label='Combined Train Loss')
    plt.plot(combined_val_loss, label='Combined Val Loss')
plt.title('Primary Model Loss History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Compute and Plot Confusion Matrix ===
print("\n--- Computing and Plotting Confusion Matrix ---")
cm_corrected = confusion_matrix(true_labels, final_corrected_labels)
plt.figure(figsize=(12, 10))
sns.heatmap(cm_corrected, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.title('Corrected Confusion Matrix (After Verification & KNN)')
plt.xlabel('Predicted Label'), plt.ylabel('Actual Label')
plt.show()

# === Final Evaluation Metrics ===
print("\n--- Final Evaluation Metrics ---")
# Use sklearn's metrics for a comprehensive view
accuracy = accuracy_score(true_labels, final_corrected_labels)
# Use 'weighted' average to account for class imbalance, if any
precision = precision_score(true_labels, final_corrected_labels, average='weighted', zero_division=0)
recall = recall_score(true_labels, final_corrected_labels, average='weighted', zero_division=0)
f1 = f1_score(true_labels, final_corrected_labels, average='weighted', zero_division=0)

metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision (Weighted)', 'Recall (Weighted)', 'F1 Score (Weighted)'],
    'Score': [accuracy, precision, recall, f1]
})
print(metrics_df)

print("\n--- Script Finished Successfully ---")
