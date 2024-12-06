import numpy as np
import matplotlib.pyplot as plt
from constants import SIDE_LINE, BASE_LINE


class HeatmapGenerator:
    def __init__(self, player_positions, ball_positions):
        """
        Initialize with tracking data.
        Args:
            player_positions (list): List of player positions as tuples (x, y).
            ball_positions (list): List of ball positions as tuples (x, y).
        """
        self.player_positions = player_positions
        self.ball_positions = ball_positions

    def generate_player_heatmap(self, output_path="output/player_heatmap.png"):
        """
        Generate a heatmap for player movements.
        Args:
            output_path (str): Path to save the heatmap image.
        """
        x_coords, y_coords = zip(*self.player_positions)
        heatmap, xedges, yedges = np.histogram2d(
            x_coords, y_coords, bins=(50, 50), range=[[0, SIDE_LINE], [0, BASE_LINE]]
        )

        plt.imshow(heatmap.T, origin="lower", cmap="hot", interpolation="nearest")
        plt.colorbar(label="Frequency")
        plt.title("Player Movement Heatmap")
        plt.xlabel("Court X-axis (meters)")
        plt.ylabel("Court Y-axis (meters)")
        plt.savefig(output_path)
        plt.close()
        print(f"Player heatmap saved to {output_path}")

    def generate_ball_heatmap(self, output_path="output/ball_heatmap.png"):
        """
        Generate a heatmap for ball movements.
        Args:
            output_path (str): Path to save the heatmap image.
        """
        x_coords, y_coords = zip(*self.ball_positions)
        heatmap, xedges, yedges = np.histogram2d(
            x_coords, y_coords, bins=(50, 50), range=[[0, SIDE_LINE], [0, BASE_LINE]]
        )

        plt.imshow(heatmap.T, origin="lower", cmap="cool", interpolation="nearest")
        plt.colorbar(label="Frequency")
        plt.title("Ball Movement Heatmap")
        plt.xlabel("Court X-axis (meters)")
        plt.ylabel("Court Y-axis (meters)")
        plt.savefig(output_path)
        plt.close()
        print(f"Ball heatmap saved to {output_path}")


# Example usage
if __name__ == "__main__":
    # Example positions (Replace with actual tracking data)
    player_positions = [(2, 3), (4, 5), (7, 8), (6, 7), (2, 3), (1, 2)]
    ball_positions = [(3, 4), (5, 6), (7, 8), (6, 5), (3, 4), (2, 2)]

    heatmap_generator = HeatmapGenerator(player_positions, ball_positions)
    heatmap_generator.generate_player_heatmap("output/example_player_heatmap.png")
    heatmap_generator.generate_ball_heatmap("output/example_ball_heatmap.png")