
# Court dimensions in meters
BASE_LINE = 10
SIDE_LINE = 20
SERVICE_SIDE_LINE = 3
NET_SIDE_LINE = 10

# Video processing
FRAME_RATE = 25  # Frames per second for processing the video
TRACKING_CONFIDENCE = 0.5
POSE_CONFIDENCE = 0.5

# Heatmap resolution
HEATMAP_WIDTH = 1920
HEATMAP_HEIGHT = 1080

# Model paths (for any saved models, if needed)
YOLO_MODEL_PATH = "yolov8n.pt"  # Path to YOLO model (uses pre-trained weights by default)
POSE_MODEL_PATH = None           # MediaPipe is used for pose estimation
BALL_TRACKING_MODEL = None       # Placeholder for TrackNet model

# Training parameters (if fine-tuning is needed)
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10