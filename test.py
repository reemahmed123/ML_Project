import os
import cv2
import numpy as np
import joblib
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model

def predict(dataFilePath, bestModelPath):
    """
    This function loads images from a folder, extracts features using ResNet50,
    and predicts their class using the provided SVM model.
    """
    
    # 1. Load the Feature Extractor (ResNet50)
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(224, 224, 3)
    )
    # This creates the feature extraction pipeline
    feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

    # 2. Load the Best Model (SVM)
    classifier = joblib.load(bestModelPath)

    # List to store final predictions
    predictions = []
    
    # Get list of images and sort them to ensure consistent order
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = sorted([f for f in os.listdir(dataFilePath) if f.lower().endswith(valid_exts)])

    # 3. Process each image
    for img_file in image_files:
        img_path = os.path.join(dataFilePath, img_file)
        
        try:
            # --- 3.1. Preprocessing (Same as training) ---
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
            img = cv2.resize(img, (224, 224))          # Resize to 224x224
            
            # Prepare format for ResNet (batch dimension + preprocessing)
            x = np.expand_dims(img, axis=0)
            x = preprocess_input(x)

            # --- 3.2. Feature Extraction ---
            features = feature_extractor.predict(x, verbose=0)
            
            # Flatten to 1D array (although pooling='avg' does this, good to be safe for sklearn)
            features = features.flatten().reshape(1, -1)

            # --- 3.3. Prediction ---
            # The SVM model pipeline includes StandardScaler, so we pass features directly
            pred_label = classifier.predict(features)[0]
            
            predictions.append(pred_label)
            
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            predictions.append("unknown") # Fallback in case of error

    return predictions