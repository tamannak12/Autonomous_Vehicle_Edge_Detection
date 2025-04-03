import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from src.edge_detection import sobel_detector, canny_detector, laplacian_detector
from src.lane_detection import detect_lanes
from src.object_detection import detect_objects
from src.utils.visualization import visualize_results
from src.utils.image_processing import preprocess_image
from config.settings import EDGE_DETECTION_PARAMS

def process_frame(frame):
    """Process a single frame through the autonomous vehicle vision pipeline"""
    try:
        # Preprocess the image
        preprocessed = preprocess_image(frame)
        
        # Apply different edge detection methods
        sobel_edges = sobel_detector(preprocessed, 
                                     EDGE_DETECTION_PARAMS['sobel_kernel_size'],
                                     EDGE_DETECTION_PARAMS['sobel_threshold'])
        
        canny_edges = canny_detector(preprocessed,
                                    EDGE_DETECTION_PARAMS['canny_low_threshold'],
                                    EDGE_DETECTION_PARAMS['canny_high_threshold'])
        
        laplacian_edges = laplacian_detector(preprocessed,
                                            EDGE_DETECTION_PARAMS['laplacian_kernel_size'],
                                            EDGE_DETECTION_PARAMS['laplacian_scale'])
        
        # Detect lanes using the edge detection results
        lanes = detect_lanes(canny_edges)
        
        # Detect objects in the frame
        objects = detect_objects(frame)
        
        # Visualize the results
        result = visualize_results(frame, sobel_edges, canny_edges, laplacian_edges, lanes, objects)
        
        return result
    except Exception as e:
        print(f"Error processing frame: {e}")
        return frame

def process_video(video_path, output_path=None):
    """Process a video file through the autonomous vehicle vision pipeline"""
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} not found.")
        return
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        result = process_frame(frame)
        
        if output_path:
            out.write(result)
        
        cv2.imshow('Autonomous Vehicle Vision', result)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

def process_image(image_path, output_path=None):
    """Process a single image through the autonomous vehicle vision pipeline"""
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found.")
        return
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image file {image_path}")
        return
    
    result = process_frame(image)
    
    if output_path:
        cv2.imwrite(output_path, result)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))  # Fix for Matplotlib
    plt.title('Autonomous Vehicle Vision')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Autonomous Vehicle Vision System')
    parser.add_argument('--input', type=str, required=True, help='Path to input image or video')
    parser.add_argument('--output', type=str, help='Path to output image or video (optional)')
    parser.add_argument('--type', type=str, choices=['image', 'video'], required=True, 
                        help='Type of input (image or video)')
    
    args = parser.parse_args()
    
    if args.type == 'image':
        process_image(args.input, args.output)
    else:
        process_video(args.input, args.output)
    
    print("Processing complete!")
