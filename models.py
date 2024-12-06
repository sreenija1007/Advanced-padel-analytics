import torch
import torchvision.models as models
from ultralytics import YOLO

def load_yolo_model(model_name="yolov8n"):
    """
    Load the YOLO model from Ultralytics.
    Args:
        model_name (str): Name of the YOLO model (e.g., 'yolov8n').
    Returns:
        YOLO: Loaded YOLO model instance.
    """
    try:
        model = YOLO(model_name)
        print(f"Successfully loaded YOLO model: {model_name}")
        return model
    except Exception as e:
        print(f"Error loading YOLO model {model_name}: {e}")
        return None

def load_pose_model(pretrained=True):
    """
    Load a ResNet-based model for pose estimation with 13 degrees of freedom.
    Args:
        pretrained (bool): Whether to load a pretrained ResNet backbone.
    Returns:
        Torchvision model object.
    """
    model = models.resnet50(pretrained=pretrained)
    # Modify the model for pose estimation (13 keypoints * 2 coordinates)
    model.fc = torch.nn.Linear(model.fc.in_features, 26)  # 13 DOF
    return model

def load_tracknet_model(model_path=None):
    """
    Load a TrackNet model for ball tracking.
    Args:
        model_path (str): Path to a custom-trained TrackNet model. If None, uses a placeholder model.
    Returns:
        Pretrained TrackNet model object or placeholder.
    """
    if model_path:
        try:
            model = torch.load(model_path)
            print(f"Loaded custom TrackNet model from {model_path}")
        except Exception as e:
            print(f"Error loading TrackNet model: {e}")
            model = None
    else:
        print("Using placeholder TrackNet model. Please replace with actual implementation.")
        model = None
    return model

def load_models(yolo_path="yolov8n.pt", tracknet_path=None, pose_pretrained=True):
    """
    Load all models required for tracking and analytics.
    Args:
        yolo_path (str): Path to YOLO model.
        tracknet_path (str): Path to TrackNet model.
        pose_pretrained (bool): Whether to use a pretrained ResNet for pose estimation.
    Returns:
        dict: Dictionary containing all models.
    """
    models = {
        "yolo_model": load_yolo_model(yolo_path),
        "tracknet_model": load_tracknet_model(tracknet_path),
        "pose_model": load_pose_model(pretrained=pose_pretrained),
    }
    return models