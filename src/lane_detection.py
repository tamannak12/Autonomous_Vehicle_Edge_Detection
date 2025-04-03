import cv2
import numpy as np

def region_of_interest(img, vertices):
    """
    Apply a mask to only keep the region of interest defined by the vertices
    
    Parameters:
    - img: Input image
    - vertices: Vertices of the polygon defining the region of interest
    
    Returns:
    - Masked image
    """
    # Define a blank mask
    mask = np.zeros_like(img)
    
    # Define a 3 channel or 1 channel color to fill the mask with
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    
    # Fill the polygon defined by vertices with the mask color
    cv2.fillPoly(mask, [vertices], ignore_mask_color)
    
    # Apply the mask
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def detect_lines(edge_image, rho=1, theta=np.pi/180, threshold=50, min_line_length=40, max_line_gap=20):
    """
    Detect lines in an edge image using Hough Transform
    
    Parameters:
    - edge_image: Binary edge image
    - rho: Distance resolution of the accumulator in pixels
    - theta: Angle resolution of the accumulator in radians
    - threshold: Accumulator threshold parameter
    - min_line_length: Minimum line length
    - max_line_gap: Maximum allowed gap between points on the same line
    
    Returns:
    - List of detected lines
    """
    lines = cv2.HoughLinesP(edge_image, rho, theta, threshold, 
                           minLineLength=min_line_length, maxLineGap=max_line_gap)
    return lines

def separate_lines(lines, img_shape):
    """
    Separate lines into left and right lane lines
    
    Parameters:
    - lines: List of detected lines
    - img_shape: Shape of the image
    
    Returns:
    - Tuple of left and right lane lines
    """
    left_lines = []
    right_lines = []
    
    if lines is None:
        return left_lines, right_lines
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 == x1:
                continue  # Skip vertical lines
            
            slope = (y2 - y1) / (x2 - x1)
            
            # Filter based on slope
            if abs(slope) < 0.5:  # Ignore horizontal lines
                continue
                
            if slope < 0:  # Negative slope = left lane
                left_lines.append(line)
            else:  # Positive slope = right lane
                right_lines.append(line)
    
    return left_lines, right_lines

def average_lines(lines, img_shape):
    """
    Average multiple lines into a single line
    
    Parameters:
    - lines: List of lines to average
    - img_shape: Shape of the image
    
    Returns:
    - Tuple of (x1, y1, x2, y2) representing the averaged line
    """
    if not lines:
        return None
    
    # Extract points and calculate slopes and intercepts
    slopes = []
    intercepts = []
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 != x1:  # Avoid division by zero
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                slopes.append(slope)
                intercepts.append(intercept)
    
    if not slopes:
        return None
    
    # Calculate average slope and intercept
    avg_slope = np.mean(slopes)
    avg_intercept = np.mean(intercepts)
    
    # Calculate line endpoints
    y1 = img_shape[0]  # Bottom of the image
    y2 = int(img_shape[0] * 0.6)  # Slightly above the middle
    
    x1 = int((y1 - avg_intercept) / avg_slope)
    x2 = int((y2 - avg_intercept) / avg_slope)
    
    return (x1, y1, x2, y2)

def detect_lanes(edge_image):
    """
    Detect lane lines in an edge image
    
    Parameters:
    - edge_image: Binary edge image
    
    Returns:
    - Dictionary containing left and right lane lines
    """
    # Define region of interest
    height, width = edge_image.shape
    vertices = np.array([[(0, height), (width/2, height*0.6), (width, height)]], dtype=np.int32)
    
    # Apply region of interest mask
    masked_edges = region_of_interest(edge_image, vertices)
    
    # Detect lines using Hough Transform
    lines = detect_lines(masked_edges)
    
    # Separate lines into left and right lanes
    left_lines, right_lines = separate_lines(lines, edge_image.shape)
    
    # Average the lines
    left_lane = average_lines(left_lines, edge_image.shape)
    right_lane = average_lines(right_lines, edge_image.shape)
    
    return {
        'left_lane': left_lane,
        'right_lane': right_lane
    }