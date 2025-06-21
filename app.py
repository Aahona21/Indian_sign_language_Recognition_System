import streamlit as st
import cv2
import operator
import numpy as np
from keras.models import model_from_json, Sequential
from string import ascii_uppercase
from PIL import Image
import time
import os
import collections  # Import collections for Counter

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


class SignRecognitionApp:
    def __init__(self):
        self.directory = "model/"
        self.model = None
        self.verification_model = None
        self.scaler = None
        self.knn_model = None
        self.feature_extractor = None

        # --- CRITICAL: Verify this class mapping against your training_set.class_indices ---
        self.reverse_class_indices = {i: char for i, char in enumerate(ascii_uppercase)}
        self.class_indices = {char: i for i, char in enumerate(ascii_uppercase)}

        self.problematic_pairs_names = [
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
        self.problematic_indices = []
        for a, b in self.problematic_pairs_names:
            if a in self.class_indices and b in self.class_indices:
                self.problematic_indices.append((self.class_indices[a], self.class_indices[b]))
            else:
                st.warning(f"Problematic pair {a}-{b} includes a class not found in `self.class_indices`. Skipping.")

        self.confidence_threshold = 0.85
        self.n_neighbors_knn = 5

        # --- Temporal Smoothing Parameters ---
        self.prediction_history_size = 10  # Number of past predictions to consider
        self.prediction_history = collections.deque(maxlen=self.prediction_history_size)
        self.display_min_confidence = 0.70  # Only display a prediction if its confidence is above this (after voting)

        # --- Load Primary Model ---
        try:
            json_path = os.path.join(self.directory, "model-primary_optimized.json")
            weights_path = os.path.join(self.directory,"model-primary_optimized.weights.h5")

            st.write(f"Loading primary model architecture from: {json_path}")
            if not os.path.exists(json_path):
                st.error(f"Primary model JSON file not found at: {json_path}")
                self.model = None
                return

            with open(json_path, "r") as json_file:
                model_json = json_file.read()
            self.model = model_from_json(model_json)
            st.success("Primary model architecture loaded successfully.")

            st.write(f"Loading primary weights from: {weights_path}")
            if not os.path.exists(weights_path):
                st.error(f"Primary model weights file not found at: {weights_path}")
                self.model = None
                return
            self.model.load_weights(weights_path)
            st.success("Primary model weights loaded successfully.")

            if len(self.model.layers) >= 2:
                self.feature_extractor = Sequential(self.model.layers[0:2])
                self.feature_extractor.compile(optimizer='adam', loss='mse')
                st.success("Feature extractor initialized.")
            else:
                st.warning("Primary model has too few layers to set up feature extractor for KNN.")
                self.feature_extractor = None

        except Exception as e:
            st.error(f"Error loading primary model or setting up feature extractor: {e}")
            print(f"Exception loading primary model: {e}")
            self.model = None
            return

        # --- Load Verification Model ---
        try:
            verification_json_path = os.path.join(self.directory, "model-verification_optimized.json")
            verification_weights_path = os.path.join(self.directory,"model-verification_optimized.weights.h5")

            st.write(f"Loading verification model architecture from: {verification_json_path}")
            if not os.path.exists(verification_json_path):
                st.error(f"Verification model JSON file not found at: {verification_json_path}")
                self.verification_model = None
                return
            with open(verification_json_path, "r") as json_file:
                verification_json = json_file.read()
            self.verification_model = model_from_json(verification_json)
            st.success("Verification model architecture loaded successfully.")

            st.write(f"Loading verification model weights from: {verification_weights_path}")
            if not os.path.exists(verification_weights_path):
                st.error(f"Verification model weights file not found at: {verification_weights_path}")
                self.verification_model = None
                return
            self.verification_model.load_weights(verification_weights_path)
            st.success("Verification model weights loaded successfully.")

        except Exception as e:
            st.error(f"Error loading verification model: {e}")
            print(f"Exception loading verification model: {e}")
            self.verification_model = None
            return

        self.current_symbol = "None"

        self.scaler = StandardScaler()
        self.knn_model = NearestNeighbors(n_neighbors=self.n_neighbors_knn, metric='euclidean')

        st.info("Remember to re-train and save your models with the '_optimized' suffix if you haven't already!")
        st.info(
            "For optimal KNN, you should pre-fit StandardScaler and NearestNeighbors on your training data's features offline and load them here.")

    def predict_with_verification(self, processed_image):
        """
        Performs prediction using the primary model and applies verification/KNN logic
        for low-confidence predictions. Returns the raw prediction index and its confidence.
        """
        try:
            primary_result = self.model.predict(processed_image, verbose=0)
            primary_probs = primary_result[0]
            primary_pred_idx = np.argmax(primary_probs)
            primary_confidence = np.max(primary_probs)

            # Verification logic: If confidence is low, consult secondary model
            if primary_confidence < self.confidence_threshold:
                secondary_result = self.verification_model.predict(processed_image, verbose=0)
                secondary_probs = secondary_result[0]
                secondary_pred_idx = np.argmax(secondary_probs)

                # If secondary model disagrees, its prediction is considered
                if secondary_pred_idx != primary_pred_idx:
                    return secondary_pred_idx, np.max(secondary_probs)  # Return secondary's prediction and confidence
                # Else, primary and secondary agree (or secondary also low confidence on primary's choice),
                # so stick with primary's prediction for further temporal smoothing
            return primary_pred_idx, primary_confidence

        except Exception as e:
            st.error(f"Prediction and verification error: {e}")
            print(f"Prediction and verification error: {e}")
            return -1, 0.0  # Return invalid index and 0 confidence on error

    def run(self):
        """
        Runs the Streamlit application for real-time sign language recognition.
        """
        # --- Custom CSS for Background Image and Text ---
        # REPLACE 'YOUR_IMAGE_URL_OR_BASE64_HERE' with your actual image URL or Base64 string
        # You can adjust the rgba values (e.g., rgba(0,0,0,0.4)) to change the fading intensity and color.
        # The first two rgba values create a black overlay, making the image "faded" and text more readable.
        st.markdown(
            """
            <style>
            .stApp {
                background-image: linear-gradient(rgba(0, 0, 0, 0.3), rgba(0, 0, 0, 0.3)), url("https://news.exeter.ac.uk/wp-content/uploads/2024/11/Untitled27.jpg");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
                color: #FFFFFF !important; /* Added !important */
            }
            /* Adjust heading colors for readability on the new background */
            .stMarkdown h1, .stMarkdown h3 {
                color: #F0F2F6 !important; /* Added !important */
            }
            /* Make the main content area slightly transparent or give it a subtle background */
            .st-emotion-cache-1pxyv22 { /* This class might need adjustment based on Streamlit version */
                background-color: rgba(255, 255, 255, 0.1) !important; /* Added !important */
                padding: 20px !important; /* Added !important */
                border-radius: 10px !important; /* Added !important */
            }
            /* For the sidebar background, if needed */
            .st-emotion-cache-vk3305 { /* This class might need adjustment based on Streamlit version */
                background-color: rgba(0, 0, 0, 0.2) !important; /* Added !important */
                border-radius: 10px !important; /* Added !important */
            }
            """,
            unsafe_allow_html=True
        )
        # --- End Custom CSS ---

        st.title("Sign Language Recognition")
        st.write("Use your webcam to recognize hand signs in real-time.")

        if 'run_webcam' not in st.session_state:
            st.session_state.run_webcam = False

        run_webcam_checkbox_value = st.checkbox("Start Webcam", value=st.session_state.run_webcam)
        st.session_state.run_webcam = run_webcam_checkbox_value

        if st.session_state.run_webcam:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error(
                    "Error: Could not open webcam. Make sure no other application is using it or check camera permissions.")
                st.session_state.run_webcam = False
                st.experimental_rerun()
                return

            frame_placeholder = st.empty()
            label_placeholder = st.empty()

            st.write("Webcam started. Processing frames...")

            while st.session_state.run_webcam:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to grab frame from webcam. Exiting recognition.")
                    st.session_state.run_webcam = False
                    break

                frame = cv2.flip(frame, 1)

                frame_height, frame_width, _ = frame.shape
                roi_size = min(frame_width, frame_height) // 2
                x1 = (frame_width - roi_size) // 2
                y1 = (frame_height - roi_size) // 2
                x2 = x1 + roi_size
                y2 = y1 + roi_size

                cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 149, 237), 2)

                roi = frame[y1:y2, x1:x2]

                try:
                    if len(roi.shape) == 2:
                        processed_roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
                    elif roi.shape[2] == 4:
                        processed_roi = cv2.cvtColor(roi, cv2.COLOR_BGRA2BGR)
                        processed_roi = cv2.cvtColor(processed_roi, cv2.COLOR_BGR2RGB)
                    else:
                        processed_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

                    processed_roi = cv2.resize(processed_roi, (128, 128))
                    processed_roi_for_model = processed_roi.reshape(1, 128, 128, 3) / 255.0

                    # Get prediction index and confidence from the model
                    prediction_idx, prediction_confidence = self.predict_with_verification(processed_roi_for_model)

                    # --- Apply Temporal Smoothing (Prediction History) ---
                    if prediction_idx != -1:  # Only add valid predictions to history
                        self.prediction_history.append(prediction_idx)

                    # Get the most common prediction from history
                    if self.prediction_history:
                        most_common_prediction_idx = collections.Counter(self.prediction_history).most_common(1)[0][0]
                        most_common_prediction_symbol = self.reverse_class_indices.get(most_common_prediction_idx,
                                                                                       "Unknown")

                        # Get the confidence for the most common prediction from the *current* frame's probabilities
                        # This is an approximation; a more robust way would be to average confidences too.
                        current_frame_probs = self.model.predict(processed_roi_for_model, verbose=0)[0]
                        confidence_of_most_common = current_frame_probs[most_common_prediction_idx]

                        if confidence_of_most_common >= self.display_min_confidence:
                            self.current_symbol = most_common_prediction_symbol
                        else:
                            self.current_symbol = "Thinking..."  # Low confidence
                    else:
                        self.current_symbol = "Waiting..."  # History is empty yet

                    display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(display_frame)
                    frame_placeholder.image(img, use_column_width=True)

                    label_placeholder.markdown(f"### Recognized Sign: **{self.current_symbol}**")

                    time.sleep(0.05)

                except Exception as e:
                    st.error(f"Frame processing or prediction error: {e}")
                    print(f"Frame processing or prediction error: {e}")
                    st.session_state.run_webcam = False
                    break

            cap.release()
            st.write("Webcam stopped.")
            if run_webcam_checkbox_value:
                st.session_state.run_webcam = False
                st.experimental_rerun()

        else:
            st.write("Check the box to start webcam and begin sign recognition.")


if __name__ == "__main__":
    app = SignRecognitionApp()
    if app.model is not None and app.verification_model is not None:
        app.run()
    else:
        st.error(
            "Application cannot start because one or more models failed to load. Please check the error messages above and ensure model files are correctly placed in the 'model/' directory.")
