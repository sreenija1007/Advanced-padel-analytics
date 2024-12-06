import numpy as np
from utils import calculate_distance, calculate_velocity
from constants import FRAME_RATE, BASE_LINE, SIDE_LINE
from projection import CourtProjection


class Analytics:
    def __init__(self, court_corners, court_dimensions):
        """
        Initialize the Analytics module with a projection model.
        Args:
            court_corners (list of tuples): Coordinates of the court corners in the image frame.
            court_dimensions (tuple): Real-world court dimensions (width, height).
        """
        self.tracking_data = []  # Stores tracking data (players, ball) for analytics
        self.projection = CourtProjection(court_corners, court_dimensions)  # Projection for real-world coordinates

    def add_tracking_data(self, frame_number, players, ball):
        """
        Add tracking data for a single frame and project coordinates to court dimensions.
        Args:
            frame_number (int): The frame number.
            players (list): List of dictionaries containing player coordinates and IDs.
            ball (tuple): Coordinates of the ball (x, y) or None if not detected.
        """
        # Project players to court coordinates
        projected_players = self.projection.project_to_court([player["coords"] for player in players])
        player_data = [{"id": player["id"], "coords": proj_coords} for player, proj_coords in zip(players, projected_players)]

        # Project ball to court coordinates (if detected)
        projected_ball = None
        if ball:
            projected_ball = self.projection.project_to_court([ball])[0]

        # Add to tracking data
        self.tracking_data.append({
            "frame": frame_number,
            "players": player_data,
            "ball": projected_ball
        })

    def calculate_velocities(self):
        """
        Calculate player and ball velocities using frame-by-frame tracking data.
        Returns:
            dict: Velocities of players and ball across frames.
        """
        velocities = {"players": {}, "ball": []}

        for i in range(1, len(self.tracking_data)):
            prev_data = self.tracking_data[i - 1]
            curr_data = self.tracking_data[i]

            # Player velocities
            for player in curr_data["players"]:
                player_id = player["id"]
                prev_coords = next((p["coords"] for p in prev_data["players"] if p["id"] == player_id), None)
                curr_coords = player["coords"]
                if player_id not in velocities["players"]:
                    velocities["players"][player_id] = []
                velocities["players"][player_id].append(
                    calculate_velocity(prev_coords, curr_coords, FRAME_RATE)
                )

            # Ball velocity
            if prev_data["ball"] and curr_data["ball"]:
                velocities["ball"].append(
                    calculate_velocity(prev_data["ball"], curr_data["ball"], FRAME_RATE)
                )
            else:
                velocities["ball"].append(0)  # Ball not detected in consecutive frames

        return velocities

    def calculate_error_rate(self):
        """
        Calculate error rate based on player movements and ball position.
        Returns:
            float: Error rate for the game.
        """
        errors = 0
        total_frames = len(self.tracking_data)

        for frame_data in self.tracking_data:
            ball = frame_data["ball"]
            players = frame_data["players"]
            if ball:
                distances = [calculate_distance(ball, player["coords"]) for player in players]
                if min(distances) > 5:  # Example threshold for error (adjust as needed)
                    errors += 1

        return errors / total_frames

    def generate_summary(self):
        """
        Generate a summary of the analytics, including velocities and error rates.
        Returns:
            dict: Summary of analytics.
        """
        velocities = self.calculate_velocities()
        error_rate = self.calculate_error_rate()

        return {
            "player_velocities": velocities["players"],
            "ball_velocity": velocities["ball"],
            "error_rate": error_rate,
        }