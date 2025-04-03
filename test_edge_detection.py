import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.edge_detection import sobel_detector, canny_detector, laplacian_detector, compare_edge_detectors
from src.utils.image_processing import preprocess_image

def test_edge_detection_on_image(image_path):
    """
    Test edge detection methods on a single image
    
    Parameters:
    - image_path: Path to the test image
    """
    # Read the image
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not read image file {image_path}")
        return
    
    # Preprocess the image
    preprocessed = preprocess_image(image)
    
    # Apply different edge detection methods
    sobel_edges = sobel_detector(preprocessed)
    canny_edges = canny_detector(preprocessed)
    laplacian_edges = laplacian_detector(preprocessed)
    
    # Display the results
    plt.figure(figsize=(15, 10))
    
    # Original image
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    # Sobel edges
    plt.subplot(2, 2, 2)
    plt.imshow(sobel_edges, cmap='gray')
    plt.title('Sobel Edge Detection')
    plt.axis('off')
    
    # Canny edges
    plt.subplot(2, 2, 3)
    plt.imshow(canny_edges, cmap='gray')
    plt.title('Canny Edge Detection')
    plt.axis('off')
    
    # Laplacian edges
    plt.subplot(2, 2, 4)
    plt.imshow(laplacian_edges, cmap='gray')
    plt.title('Laplacian Edge Detection')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Compare the methods
    print("Edge Detection Comparison:")
    print(f"Sobel: {np.sum(sobel_edges > 0)} edge pixels")
    print(f"Canny: {np.sum(canny_edges > 0)} edge pixels")
    print(f"Laplacian: {np.sum(laplacian_edges > 0)} edge pixels")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Edge Detection Methods')
    parser.add_argument('--image', type=str, required=True, help='Path to test image')
    
    args = parser.parse_args()
    
    test_edge_detection_on_image(args.image)