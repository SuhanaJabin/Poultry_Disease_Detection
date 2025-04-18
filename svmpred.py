import cv2
import numpy as np
import pandas as pd
from sklearn.svm import SVC

# Function to extract elongation feature
def extract_elongation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    _, _, w, h = cv2.boundingRect(largest_contour)
    ratio = h / w
    return ratio

# Function to extract circularity feature
def extract_circularity(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    isoperimetric_ratio = (4 * np.pi * area) / (perimeter * perimeter)
    return isoperimetric_ratio

# Function to extract area linear rate feature
def extract_area_linear_rate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    area_circumference_ratio = area / perimeter
    return area_circumference_ratio

# Function to extract shape features
def extract_shape_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.04 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    num_corners = len(approx)
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    circularity = 4 * np.pi * (area / perimeter**2) if perimeter != 0 else 0
    hull = cv2.convexHull(largest_contour)
    solidity = area / cv2.contourArea(hull) if cv2.contourArea(hull) != 0 else 0
    return num_corners, circularity, solidity

# Function to predict using SVM model
def predict_with_svm(features):
    # Load the trained SVM model
    svm_model = SVC()
    svm_model.load_model("svm_model.joblib")  # Replace with your trained model path
    
    # Make prediction
    prediction = svm_model.predict(features)
    return prediction

# Load the input image
input_image_path = "/home/labadmin/R7A_group11/augmented dataset/data/healthy_augmented_actual/DSC_4891_blurred.jpg"  # Replace with your input image path
image = cv2.imread(input_image_path)

# Extract features
elongation = extract_elongation(image)
circularity = extract_circularity(image)
area_linear_rate = extract_area_linear_rate(image)
num_corners, circularity, solidity = extract_shape_features(image)

# Create a DataFrame for the extracted features
data = {'Elongation': [elongation],
        'Circularity': [circularity],
        'Area_Linear_Rate': [area_linear_rate],
        'Number_of_Corners': [num_corners],
        'Circularity_Solidity': [circularity],
        'Solidity': [solidity]}
df = pd.DataFrame(data)

# Save the features to an Excel sheet
output_excel_path = "/home/labadmin/R7A_group11/augmented dataset/data/"  # Replace with your desired output Excel path
df.to_excel(output_excel_path, index=False)

# Predict using SVM model
prediction = predict_with_svm(df)

# Print the prediction
print("Prediction:", prediction)
