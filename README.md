Overview
This project explores advanced AI techniques to analyze Padel gameplay by accurately tracking players and the ball. Leveraging state-of-the-art deep learning models such as YOLO, ResNet, and TrackNet, the framework aims to overcome challenges like occlusions, fast-moving objects, and irrelevant detections outside the court. The project further integrates innovative methodologies such as masking, pose estimation, and court keypoint projections for enhanced tracking precision.

Methodology
1. ResNet for Player Tracking:
Implemented ResNet for detecting and tracking players.
Observed challenges with extraneous detections outside the court.
2. ResNet with Masking:
Applied masking techniques to limit detection to the court area.
Improved detection accuracy by focusing exclusively on relevant regions.
3. YOLOv8 Integration:
Used YOLOv8 for efficient player detection with masking.
Achieved faster inference times while maintaining accuracy.
4. TrackNet Integrated with YOLOv8:
Combined YOLOv8 with TrackNet for ball tracking.
Leveraged pretrained weights from a tennis dataset for initial implementation.
Addressed challenges in tracking fast-moving balls and handling court boundary variations.
5. Court Keypoints for 2D Projections:
Introduced manual selection of court keypoints to define 2D boundaries.
Enhanced tracking accuracy by addressing occlusions and misdetections.
6. MediaPipe for Pose Estimation:
Integrated MediaPipe for pose estimation to identify player-ball interactions.
Enabled tracking of player actions such as hitting the ball.

Evaluation Metrics:

Detection Accuracy: Measures the precision of object detection.
Inference Time: Assesses real-time processing capability.
Tracking Error: Quantifies deviations in object tracking.
Heatmap Precision: Evaluates spatial analytics accuracy using court keypoints.
