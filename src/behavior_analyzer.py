# import os
# import cv2
# import numpy as np
# import pandas as pd
# from ultralytics import YOLO
# from datetime import datetime
# import matplotlib.pyplot as plt

# # Load a pre-trained YOLOv8 model (or train one on synthetic fish images)
# MODEL_PATH = "yolov8n.pt"  # Pre-trained model for object detection

# #MODEL_PATH = "D:/AQ1/Feeding Optimization System/data/yolo_dataset/runs/detect/train/weights/best.pt"
# model = YOLO(MODEL_PATH)

# def detect_fish_in_image(image_path):
#     """
#     Detect fish in a single image using YOLOv8.
#     Returns bounding boxes of detected fish.
#     """
#     try:
#         results = model(image_path)
#         detections = []
#         for result in results:
#             boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
#             detections.extend(boxes)
#         return detections
#     except Exception as e:
#         print(f"Error detecting fish in {image_path}: {e}")
#         return []

# def calculate_feeding_activity(detections, img_size=(300, 300)):
#     """
#     Calculate feeding activity score based on fish clustering and movement.
#     """
#     if not detections:
#         return 0.0
    
#     # Convert detections to numpy array
#     detections = np.array(detections)
    
#     # Compute centroid of all detections
#     centroids = detections[:, :2] + (detections[:, 2:] - detections[:, :2]) / 2
#     mean_centroid = np.mean(centroids, axis=0)
    
#     # Clustering score: average distance from mean centroid
#     distances = np.linalg.norm(centroids - mean_centroid, axis=1)
#     clustering_score = np.mean(distances)
    
#     # Normalize clustering score to [0, 1]
#     max_possible_distance = np.sqrt((img_size[0] / 2) ** 2 + (img_size[1] / 2) ** 2)  # Max distance from center
#     normalized_score = 1 - (clustering_score / max_possible_distance)
    
#     # Clamp the score to [0, 1]
#     normalized_score = max(0, min(1, normalized_score))
    
#     return normalized_score

# def visualize_detections(image_path, detections):
#     """
#     Visualize bounding boxes on an image.
#     """
#     img = cv2.imread(image_path)
#     if img is None:
#         print(f"Error: Unable to load image {image_path}")
#         return
    
#     for box in detections:
#         x1, y1, x2, y2 = map(int, box)
#         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green bounding box
    
#     cv2.imshow("Detections", img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# def analyze_feeding_behavior(image_folder="../data/synthetic/fish_images", output_csv="../data/feeding_events.csv", visualize=False):
#     """
#     Analyze feeding behavior across a sequence of images.
#     Outputs a CSV with timestamps and feeding activity scores.
#     """
#     if not os.path.exists(image_folder):
#         print(f"Error: Image folder {image_folder} does not exist.")
#         return
    
#     image_files = sorted(os.listdir(image_folder))
#     feeding_events = []
    
#     for i, image_file in enumerate(image_files):
#         image_path = os.path.join(image_folder, image_file)
        
#         # Detect fish in the image
#         detections = detect_fish_in_image(image_path)
        
#         # Visualize detections if enabled
#         if visualize:
#             visualize_detections(image_path, detections)
        
#         # Simulate timestamp based on image index
#         timestamp = datetime.strptime(f"2024-01-01 08:{i % 60:02d}", "%Y-%m-%d %H:%M")
        
#         # Calculate feeding activity score
#         activity_score = calculate_feeding_activity(detections)
        
#         # Log feeding event
#         feeding_events.append({"timestamp": timestamp, "activity_score": activity_score})
    
#     # Save results to CSV
#     df = pd.DataFrame(feeding_events)
#     df.to_csv(output_csv, index=False)
#     print(f"Feeding events saved to {output_csv}")
    
#     # Plot feeding activity scores
#     plt.figure(figsize=(10, 5))
#     plt.plot(df["timestamp"], df["activity_score"], marker='o', linestyle='-', color='b')
#     plt.title("Feeding Activity Over Time")
#     plt.xlabel("Timestamp")
#     plt.ylabel("Activity Score")
#     plt.grid(True)
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     analyze_feeding_behavior(visualize=True)  # Set visualize=False to skip visualization



# behavior_analyzer.py

import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from datetime import datetime
import matplotlib.pyplot as plt

#Load a pre-trained YOLOv8 model (or train one on synthetic fish images)
MODEL_PATH = "D:/AQ1/Feeding Optimization System/data/yolo_dataset/runs/detect/train/weights/best.pt"  # Path to your fine-tuned model
model = YOLO(MODEL_PATH)

def detect_fish_in_image(image_path):
    """
    Detect fish in a single image using YOLOv8.
    Returns bounding boxes of detected fish.
    """
    try:
        results = model(image_path)
        detections = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
            detections.extend(boxes)
        return detections
    except Exception as e:
        print(f"Error detecting fish in {image_path}: {e}")
        return []

def calculate_feeding_activity(detections, img_size=(300, 300)):
    if not detections:
        return 0.0
    
    # Convert detections to numpy array
    detections = np.array(detections)
    
    # Compute centroid of all detections
    centroids = detections[:, :2] + (detections[:, 2:] - detections[:, :2]) / 2
    mean_centroid = np.mean(centroids, axis=0)
    
    # Clustering score: average distance from mean centroid
    distances = np.linalg.norm(centroids - mean_centroid, axis=1)
    clustering_score = np.mean(distances)
    
    # Normalize clustering score to [0, 1]
    max_distance = np.sqrt(sum(np.array(img_size) ** 2))  # Maximum possible distance
    clustering_score = 1 - (clustering_score / max_distance)
    
    return clustering_score

def visualize_detections(image_path, detections):
    """
    Visualize bounding boxes on an image.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image {image_path}")
        return
    
    for box in detections:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green bounding box
    
    cv2.imshow("Detections", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def analyze_feeding_behavior(image_folder="../data/synthetic/fish_images", output_csv="../data/feeding_events.csv", visualize=False):
    """
    Analyze feeding behavior across a sequence of images.
    Outputs a CSV with timestamps and feeding activity scores.
    """
    if not os.path.exists(image_folder):
        print(f"Error: Image folder {image_folder} does not exist.")
        return
    
    image_files = sorted(os.listdir(image_folder))
    feeding_events = []
    
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(image_folder, image_file)
        
        # Detect fish in the image
        detections = detect_fish_in_image(image_path)
        
        # Visualize detections if enabled
        if visualize:
            visualize_detections(image_path, detections)
        
        # Simulate timestamp based on image index
        timestamp = datetime.strptime(f"2024-01-01 08:{i % 60:02d}", "%Y-%m-%d %H:%M")
        
        # Calculate feeding activity score
        activity_score = calculate_feeding_activity(detections)
        
        # Log feeding event
        feeding_events.append({"timestamp": timestamp, "activity_score": activity_score})
    
    # Save results to CSV
    df = pd.DataFrame(feeding_events)
    df.to_csv(output_csv, index=False)
    print(f"Feeding events saved to {output_csv}")
    
    # Plot feeding activity scores
    plt.figure(figsize=(10, 5))
    plt.plot(df["timestamp"], df["activity_score"], marker='o', linestyle='-', color='b')
    plt.title("Feeding Activity Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Activity Score")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Debugging Section: Test the calculate_feeding_activity function
if __name__ == "__main__":
    # Example: Single fish
    detections_single = [[50, 50, 70, 70]]
    print("Single fish score:", calculate_feeding_activity(detections_single))  # Should be 0.5

    # Example: Fish are tightly clustered
    detections_tight = [[50, 50, 70, 70], [60, 60, 80, 80], [55, 55, 75, 75]]
    print("Tightly clustered fish score:", calculate_feeding_activity(detections_tight))  # Should be close to 1.0

    # Example: Fish are spread out
    detections_spread = [[50, 50, 70, 70], [200, 200, 220, 220], [100, 100, 120, 120]]
    print("Spread-out fish score:", calculate_feeding_activity(detections_spread))  # Should be closer to 0.0

    # Run the full analysis
    analyze_feeding_behavior(visualize=True)
