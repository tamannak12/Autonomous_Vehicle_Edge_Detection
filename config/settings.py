import numpy as np
"""
Configuration settings for the autonomous vehicle vision system
"""

# Edge detection parameters
EDGE_DETECTION_PARAMS = {
    # Sobel parameters
    'sobel_kernel_size': 3,
    'sobel_threshold': 50,
    
    # Canny parameters
    'canny_low_threshold': 50,
    'canny_high_threshold': 150,
    
    # Laplacian parameters
    'laplacian_kernel_size': 3,
    'laplacian_scale': 1
}

# Lane detection parameters
LANE_DETECTION_PARAMS = {
    'hough_rho': 1,
    'hough_theta': 0.017453292519943295,  # np.pi/180
    'hough_threshold': 50,
    'hough_min_line_length': 40,
    'hough_max_line_gap': 20,
    
    'min_slope': 0.5,  # Minimum slope to consider a line as a lane
    'roi_vertices_ratio': {
        'bottom_left': (0.0, 1.0),
        'top_left': (0.4, 0.6),
        'top_right': (0.6, 0.6),
        'bottom_right': (1.0, 1.0)
    }
}

# Object detection parameters
OBJECT_DETECTION_PARAMS = {
    'car_scale_factor': 1.1,
    'car_min_neighbors': 5,
    'car_min_size': (30, 30),
    
    'pedestrian_scale_factor': 1.1,
    'pedestrian_min_neighbors': 5,
    'pedestrian_min_size': (30, 30),
    
    'max_safe_distance': 50  # Maximum distance considered safe
}

# Visualization parameters
VISUALIZATION_PARAMS = {
    'lane_color': (0, 255, 0),  # Green
    'lane_thickness': 5,
    
    'car_color': (0, 0, 255),  # Red
    'pedestrian_color': (255, 0, 0),  # Blue
    
    'text_color': (0, 0, 0),  # Black
    'text_size': 0.5,
    'text_thickness': 2
}

# Camera calibration parameters
# These would typically be determined through a calibration process
CAMERA_CALIBRATION = {
    'camera_matrix': np.array([
        [1000, 0, 640],
        [0, 1000, 480],
        [0, 0, 1]
    ]),
    'dist_coeffs': np.array([0.1, 0.01, 0.001, 0.001, 0.01])
}

# Performance parameters
PERFORMANCE_PARAMS = {
    'target_fps': 30,
    'processing_resolution': (640, 480),  # Process at lower resolution for speed
    'display_resolution': (1280, 720)     # Display at higher resolution
}