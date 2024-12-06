import cv2
import numpy as np

class CourtProjection:
    def __init__(self, court_corners, court_dimensions):
        """
        Initialize the projection module.
        Args:
            court_corners (list of tuples): Coordinates of the court corners in the image frame.
            court_dimensions (tuple): Real-world dimensions of the court (width, height).
        """
        self.court_corners = np.array(court_corners, dtype=np.float32)
        self.court_dimensions = np.array([
            [0, 0],
            [court_dimensions[0], 0],
            [court_dimensions[0], court_dimensions[1]],
            [0, court_dimensions[1]],
        ], dtype=np.float32)

        # Compute homography matrix
        self.homography_matrix, _ = cv2.findHomography(self.court_corners, self.court_dimensions)

    def project_to_court(self, image_coords):
        """
        Project image coordinates to court coordinates.
        Args:
            image_coords (list of tuples): List of (x, y) points in image frame.
        Returns:
            list of tuples: List of (x, y) points in court coordinates.
        """
        if len(image_coords) == 0:
            return []

        image_coords = np.array(image_coords, dtype=np.float32).reshape(-1, 1, 2)
        court_coords = cv2.perspectiveTransform(image_coords, self.homography_matrix)
        return court_coords.reshape(-1, 2).tolist()

    def project_to_image(self, court_coords):
        """
        Project court coordinates back to image coordinates.
        Args:
            court_coords (list of tuples): List of (x, y) points in court frame.
        Returns:
            list of tuples: List of (x, y) points in image coordinates.
        """
        if len(court_coords) == 0:
            return []

        court_coords = np.array(court_coords, dtype=np.float32).reshape(-1, 1, 2)
        image_coords = cv2.perspectiveTransform(court_coords, np.linalg.inv(self.homography_matrix))
        return image_coords.reshape(-1, 2).tolist()