import cv2
import joblib
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model

# Load trained SVM model
svm_model = joblib.load("svm_material_classifier.pkl")

# Load CNN for feature extraction
base_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
cnn_model = Model(inputs=base_model.input, outputs=base_model.output)

def extract_features(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = np.expand_dims(frame, axis=0)
    frame = preprocess_input(frame)
    features = cnn_model.predict(frame, verbose=0)
    return features.flatten()

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    feat = extract_features(frame)
    probs = svm_model.predict_proba([feat])[0]
    confidence = np.max(probs)

    
    label = svm_model.classes_[np.argmax(probs)]

    cv2.putText(
        frame,
        f"{label}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Material Stream Identification", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
