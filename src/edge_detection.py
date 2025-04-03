import cv2
import numpy as np

def sobel_detector(image, kernel_size=3, threshold=50):
    """
    Apply Sobel edge detection to an image
    
    Parameters:
    - image: Input grayscale image
    - kernel_size: Size of the Sobel kernel
    - threshold: Threshold for edge detection
    
    Returns:
    - Binary edge map
    """
    # Apply Sobel in x and y directions
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel_size)
    
    # Calculate the gradient magnitude
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Normalize to 0-255
    gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    # Apply threshold
    _, binary_output = cv2.threshold(gradient_magnitude, threshold, 255, cv2.THRESH_BINARY)
    
    return binary_output

def canny_detector(image, low_threshold=50, high_threshold=150):
    """
    Apply Canny edge detection to an image
    
    Parameters:
    - image: Input grayscale image
    - low_threshold: Lower threshold for edge detection
    - high_threshold: Higher threshold for edge detection
    
    Returns:
    - Binary edge map
    """
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    
    return edges

def laplacian_detector(image, kernel_size=3, scale=1):
    """
    Apply Laplacian edge detection to an image
    
    Parameters:
    - image: Input grayscale image
    - kernel_size: Size of the Laplacian kernel
    - scale: Scale factor for the Laplacian
    
    Returns:
    - Binary edge map
    """
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Apply Laplacian
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=kernel_size)
    
    # Take absolute value and convert to 8-bit
    laplacian = np.uint8(np.absolute(laplacian) * scale)
    
    # Apply threshold to create binary edge map
    _, binary_output = cv2.threshold(laplacian, 30, 255, cv2.THRESH_BINARY)
    
    return binary_output

def compare_edge_detectors(image):
    """
    Compare different edge detection methods on the same image
    
    Parameters:
    - image: Input grayscale image
    
    Returns:
    - Dictionary containing edge maps from different methods
    """
    # Apply different edge detection methods
    sobel_edges = sobel_detector(image)
    canny_edges = canny_detector(image)
    laplacian_edges = laplacian_detector(image)
    
    # Return a dictionary of results
    return {
        'sobel': sobel_edges,
        'canny': canny_edges,
        'laplacian': laplacian_edges
    }