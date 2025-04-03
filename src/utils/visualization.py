import cv2
import numpy as np
import matplotlib.pyplot as plt

def draw_lanes(image, lanes, color=(0, 255, 0), thickness=5):
    """
    Draw lane lines on an image
    
    Parameters:
    - image: Input image
    - lanes: Dictionary containing left and right lane lines
    - color: Color of the lane lines (BGR format)
    - thickness: Thickness of the lane lines
    
    Returns:
    - Image with lane lines drawn
    """
    # Create a copy of the image
    result = np.copy(image)
    
    # Draw left lane line
    if lanes['left_lane'] is not None:
        x1, y1, x2, y2 = lanes['left_lane']
        cv2.line(result, (x1, y1), (x2, y2), color, thickness)
    
    # Draw right lane line
    if lanes['right_lane'] is not None:
        x1, y1, x2, y2 = lanes['right_lane']
        cv2.line(result, (x1, y1), (x2, y2), color, thickness)
    
    return result

def draw_objects(image, objects):
    """
    Draw detected objects on an image
    
    Parameters:
    - image: Input image
    - objects: Dictionary containing detected objects
    
    Returns:
    - Image with objects drawn
    """
    # Create a copy of the image
    result = np.copy(image)
    
    # Draw cars
    for (x, y, w, h) in objects.get('cars', []):
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(result, 'Car', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Draw pedestrians
    for (x, y, w, h) in objects.get('pedestrians', []):
        cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(result, 'Pedestrian', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return result

def visualize_edge_detection(original, sobel, canny, laplacian):
    """
    Create a visualization comparing different edge detection methods
    
    Parameters:
    - original: Original image
    - sobel: Sobel edge detection result
    - canny: Canny edge detection result
    - laplacian: Laplacian edge detection result
    
    Returns:
    - Visualization image
    """
    # Convert original to grayscale if it's not already
    if len(original.shape) == 3:
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    else:
        gray = original
    
    # Create a figure with subplots
    plt.figure(figsize=(12, 8))
    
    # Original grayscale image
    plt.subplot(2, 2, 1)
    plt.imshow(gray, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # Sobel edge detection
    plt.subplot(2, 2, 2)
    plt.imshow(sobel, cmap='gray')
    plt.title('Sobel Edge Detection')
    plt.axis('off')
    
    # Canny edge detection
    plt.subplot(2, 2, 3)
    plt.imshow(canny, cmap='gray')
    plt.title('Canny Edge Detection')
    plt.axis('off')
    
    # Laplacian edge detection
    plt.subplot(2, 2, 4)
    plt.imshow(laplacian, cmap='gray')
    plt.title('Laplacian Edge Detection')
    plt.axis('off')
    
    plt.tight_layout()
    
    # Convert the plot to an image
    fig = plt.gcf()
    fig.canvas.draw()
    
    # Convert canvas to image
    img = np.array(fig.canvas.renderer.buffer_rgba())
    
    # Close the plot to free memory
    plt.close()
    
    # Convert RGBA to BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    
    return img

def visualize_results(original, sobel_edges, canny_edges, laplacian_edges, lanes, objects):
    """
    Create a comprehensive visualization of all results
    
    Parameters:
    - original: Original image
    - sobel_edges: Sobel edge detection result
    - canny_edges: Canny edge detection result
    - laplacian_edges: Laplacian edge detection result
    - lanes: Detected lane lines
    - objects: Detected objects
    
    Returns:
    - Visualization image
    """
    # Create a copy of the original image
    result = np.copy(original)
    
    # Draw lane lines
    result = draw_lanes(result, lanes)
    
    # Draw detected objects
    result = draw_objects(result, objects)
    
    # Create a smaller version of the edge detection comparison
    edge_comparison = visualize_edge_detection(original, sobel_edges, canny_edges, laplacian_edges)
    edge_comparison = cv2.resize(edge_comparison, (result.shape[1] // 3, result.shape[0] // 3))
    
    # Place the edge comparison in the top-right corner
    h, w = edge_comparison.shape[:2]
    result[10:10+h, result.shape[1]-w-10:result.shape[1]-10] = edge_comparison
    
    # Add a border around the edge comparison
    cv2.rectangle(result, (result.shape[1]-w-10-1, 10-1), (result.shape[1]-10+1, 10+h+1), (0, 0, 0), 2)
    
    # Add title and information
    cv2.putText(result, 'Autonomous Vehicle Vision System', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    return result