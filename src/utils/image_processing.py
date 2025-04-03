import cv2
import numpy as np

def preprocess_image(image):
    """
    Preprocess an image for edge detection
    
    Parameters:
    - image: Input image (BGR format)
    
    Returns:
    - Preprocessed grayscale image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    return blurred

def undistort_image(image, camera_matrix, dist_coeffs):
    """
    Undistort an image using camera calibration parameters
    
    Parameters:
    - image: Input image
    - camera_matrix: Camera matrix from calibration
    - dist_coeffs: Distortion coefficients from calibration
    
    Returns:
    - Undistorted image
    """
    return cv2.undistort(image, camera_matrix, dist_coeffs, None, camera_matrix)

def perspective_transform(image, src_points, dst_points):
    """
    Apply perspective transform to get bird's eye view
    
    Parameters:
    - image: Input image
    - src_points: Source points in the original image
    - dst_points: Destination points in the transformed image
    
    Returns:
    - Transformed image
    """
    # Calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Apply the transform
    warped = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))
    
    return warped

def enhance_contrast(image):
    """
    Enhance contrast in an image using CLAHE
    
    Parameters:
    - image: Input grayscale image
    
    Returns:
    - Contrast-enhanced image
    """
    # Create a CLAHE object
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    # Apply CLAHE
    enhanced = clahe.apply(image)
    
    return enhanced

def color_threshold(image, lower_bound, upper_bound, color_space='HSV'):
    """
    Apply color thresholding to an image
    
    Parameters:
    - image: Input image (BGR format)
    - lower_bound: Lower bound for thresholding
    - upper_bound: Upper bound for thresholding
    - color_space: Color space to use ('HSV', 'HLS', etc.)
    
    Returns:
    - Binary mask where pixels within the threshold are white
    """
    # Convert to the specified color space
    if color_space == 'HSV':
        converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif color_space == 'HLS':
        converted = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    else:
        converted = image  # Use as is
    
    # Apply threshold
    mask = cv2.inRange(converted, lower_bound, upper_bound)
    
    return mask