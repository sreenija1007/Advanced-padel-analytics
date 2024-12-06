import numpy as np
import cv2
import matplotlib.pyplot as plt
from constants import SIDE_LINE, BASE_LINE

def calculate_distance(coord1, coord2):
    """
    Calculate Euclidean distance between two points.
    Args:
        coord1 (tuple): First point (x, y).
        coord2 (tuple): Second point (x, y).
    Returns:
        float: Euclidean distance.
    """
    return np.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

def calculate_velocity(prev_coord, curr_coord, frame_rate):
    """
    Calculate velocity based on consecutive coordinates and frame rate.
    Args:
        prev_coord (tuple): Previous position (x, y).
        curr_coord (tuple): Current position (x, y).
        frame_rate (int): Frame rate of the video.
    Returns:
        float: Velocity in units per second.
    """
    if prev_coord is None or curr_coord is None:
        return 0
    distance = calculate_distance(prev_coord, curr_coord)
    return distance * frame_rate

def draw_bounding_boxes(frame, player_boxes, ball_coords=None):
    """
    Draws bounding boxes for players and the ball on a video frame.
    Args:
        frame (numpy.ndarray): The video frame to annotate.
        player_boxes (list): List of player bounding boxes [(x1, y1, x2, y2), ...].
        ball_coords (tuple): Ball coordinates as (x, y) or None if not detected.
    Returns:
        numpy.ndarray: The annotated frame.
    """
    # Draw player bounding boxes
    for box in player_boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "Player", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw ball coordinates
    if ball_coords:
        ball_x, ball_y = map(int, ball_coords)
        cv2.circle(frame, (ball_x, ball_y), 5, (0, 0, 255), -1)
        cv2.putText(frame, "Ball", (ball_x, ball_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return frame

def generate_heatmap(positions, output_path, title, xlabel="X-axis (meters)", ylabel="Y-axis (meters)"):
    """
    Generate and save a heatmap from positions.
    Args:
        positions (list): List of positions as (x, y) tuples.
        output_path (str): Path to save the heatmap image.
        title (str): Title for the heatmap.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
    """
    x_coords, y_coords = zip(*positions)
    heatmap, xedges, yedges = np.histogram2d(
        x_coords, y_coords, bins=(50, 50), range=[[0, SIDE_LINE], [0, BASE_LINE]]
    )
    plt.imshow(heatmap.T, origin="lower", cmap="hot", interpolation="nearest")
    plt.colorbar(label="Frequency")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(output_path)
    plt.close()
    print(f"Heatmap saved to {output_path}")