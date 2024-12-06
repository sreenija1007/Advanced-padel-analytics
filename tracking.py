import cv2
import torch
import torchvision
from torchvision.transforms import functional as F
from utils import draw_bounding_boxes


class Tracker:
    def __init__(self):
        """
        Initialize the tracker with a Faster R-CNN model for player detection.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.player_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.player_model.to(self.device)
        self.player_model.eval()

    def preprocess_frame(self, frame):
        """
        Preprocess a video frame for Faster R-CNN.
        Args:
            frame (numpy.ndarray): The input frame.
        Returns:
            torch.Tensor: Preprocessed frame as a tensor.
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
        tensor = F.to_tensor(frame_rgb).to(self.device)  # Convert to PyTorch tensor
        return tensor

    def track_players_and_ball(self, video_path, output_path="output/tracked_video.mp4"):
        """
        Track players and the ball using Faster R-CNN and save the annotated output video.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")

        # Video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

        output_video = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (frame_width, frame_height)
        )

        print("Starting tracking process...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess frame
            tensor_frame = [self.preprocess_frame(frame)]

            # Player detection (Faster R-CNN)
            with torch.no_grad():
                predictions = self.player_model(tensor_frame)

            # Extract bounding boxes with high confidence scores
            boxes = predictions[0]["boxes"].cpu().numpy()
            scores = predictions[0]["scores"].cpu().numpy()
            player_boxes = boxes[scores > 0.5]  # Filter by confidence threshold

            # Ball tracking placeholder
            ball_coords = None  # Replace with ball tracking logic if needed

            # Draw bounding boxes on the frame
            annotated_frame = draw_bounding_boxes(frame, player_boxes, ball_coords)

            # Write annotated frame to output video
            output_video.write(annotated_frame)

        cap.release()
        output_video.release()
        print(f"Tracking completed. Output saved to {output_path}")


if __name__ == "__main__":
    tracker = Tracker()
    input_video = "data/rally.mp4"
    output_video = "output/tracked_rally.mp4"

    tracker.track_players_and_ball(input_video, output_video)