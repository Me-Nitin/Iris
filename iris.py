import cv2
import numpy as np
from scipy.spatial.distance import euclidean


def preprocess_iris(image_path, target_size=(250, 250)):
   
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print("Error: Could not load the image.")
        return None
    
   
    image = cv2.resize(image, target_size)
    
    
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    
    _, thresholded = cv2.threshold(blurred, 70, 255, cv2.THRESH_BINARY_INV)
    
  
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(thresholded)
    
   
    kernel = np.ones((3, 3), np.uint8)
    processed = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
    
   
    edges = cv2.Canny(processed, 50, 150)
    
    return edges


def store_user_iris_features(user_id, image_path):
    features = preprocess_iris(image_path)
    if features is not None:
        
        np.save(f'user_{user_id}_iris_features.npy', features)
        print(f"User {user_id} iris features saved successfully.")
    else:
        print(f"Error processing the iris image for user {user_id}.")


def check_iris_user(image_path):
    
    features = preprocess_iris(image_path)
    if features is None:
        return
    
    
    registered_user_iris_features = [
        np.load('user_1_iris_features.npy'),
        np.load('user_2_iris_features.npy'),
    
    ]
    
   
    distances = [euclidean(features.flatten(), registered_features.flatten()) for registered_features in registered_user_iris_features]
    
    
    threshold = 5000  

    min_distance = min(distances)
    if min_distance < threshold:
        matched_user = distances.index(min_distance) + 1 
        print(f"Match found! Registered user {matched_user} detected.")
    else:
        print("Unregistered user detected.")

store_user_iris_features(1, 'user1.jpg')
store_user_iris_features(2, 'user2.jpg')

check_iris_user('unknown_user1.jpg')
