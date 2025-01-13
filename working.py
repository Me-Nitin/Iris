import cv2
import numpy as np

image = cv2.imread('user2.jpg', cv2.IMREAD_GRAYSCALE)

blurred = cv2.GaussianBlur(image, (5, 5), 0)

_, thresholded = cv2.threshold(blurred, 70, 255, cv2.THRESH_BINARY_INV)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(thresholded)

kernel = np.ones((3, 3), np.uint8)
processed = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)

edges = cv2.Canny(processed, 50, 150)

cv2.imshow('Original Image', image)
cv2.imshow('Blurred Image', blurred)
cv2.imshow('Thresholded Image', thresholded)
cv2.imshow('Contrast Enhanced', enhanced)
cv2.imshow('Processed Image', processed)
cv2.imshow('Edge Detection', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('iris_detection_result.jpg', processed)
