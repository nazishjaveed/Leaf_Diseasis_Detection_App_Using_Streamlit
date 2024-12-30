import streamlit as st
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image

# Function to extract color histograms from the image
def extract_color_histogram(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Extract color histogram from the image (using 256 bins for each channel)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    # Normalize the histogram
    hist = hist.flatten()
    return hist

# Load dataset (replace with your leaf disease dataset)
def load_leaf_disease_dataset():
    # Example: you should load your leaf disease dataset here.
    # For now, we use a dummy dataset (replace this with your actual dataset)
    # Assume X is the feature set and y is the label set (e.g., disease labels)
    
    # Dummy example data (this should be replaced with actual dataset)
    X = np.random.rand(100, 256)  # 100 samples, 256 features (color histogram)
    y = np.random.randint(0, 2, 100)  # 100 labels (0 or 1, for two classes)
    
    return X, y

# Train the model
X, y = load_leaf_disease_dataset()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Encode the labels if necessary (if labels are categorical)
le = LabelEncoder()
y_train = le.fit_transform(y_train)

# Streamlit app interface
st.title("Leaf Disease Detection")

# File uploader to upload leaf image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("Classifying...")

    # Convert the uploaded image to OpenCV format
    image_array = np.array(image)

    # Extract color histogram (or other features) from the image
    image_features = extract_color_histogram(image_array)

    # Predict the disease category
    prediction = model.predict([image_features])
    disease_category = le.inverse_transform(prediction)

    st.write(f"Disease Detected: {disease_category[0]}")
