from tracking import Tracker
from analytics import Analytics
from heatmap import HeatmapGenerator
from constants import SIDE_LINE, BASE_LINE

def main():
    # Input and output paths
    input_video = "data/rally.mp4"
    output_video = "output/tracked_rally.mp4"
    player_heatmap_path = "output/player_heatmap.png"
    ball_heatmap_path = "output/ball_heatmap.png"

    # Create output directory
    import os
    os.makedirs("output", exist_ok=True)

    # Court dimensions
    court_corners = [(100, 50), (1180, 50), (1180, 650), (100, 650)]  # Example corners in image space
    court_dimensions = (SIDE_LINE, BASE_LINE)  # Real-world dimensions

    # Step 1: Track players and ball
    print("Tracking players and ball...")
    tracker = Tracker()
    tracker.track_players_and_ball(input_video, output_video)

    # Example: Tracking data format from tracker
    tracking_data = [
        {"frame": 1, "players": [{"id": 1, "coords": (2, 3)}, {"id": 2, "coords": (4, 5)}], "ball": (3, 4)},
        {"frame": 2, "players": [{"id": 1, "coords": (3, 4)}, {"id": 2, "coords": (5, 6)}], "ball": (4, 5)},
    ]

    # Step 2: Analyze tracking data
    print("Analyzing tracking data...")
    analytics = Analytics(court_corners, court_dimensions)
    for data in tracking_data:
        analytics.add_tracking_data(
            frame_number=data["frame"], players=data["players"], ball=data["ball"]
        )

    summary = analytics.generate_summary()
    print("Analytics Summary:")
    print(summary)

    # Step 3: Generate heatmaps
    print("Generating heatmaps...")
    player_positions = [player["coords"] for frame in analytics.tracking_data for player in frame["players"]]
    ball_positions = [frame["ball"] for frame in analytics.tracking_data if frame["ball"]]

    heatmap_generator = HeatmapGenerator(player_positions, ball_positions)
    heatmap_generator.generate_player_heatmap(player_heatmap_path)
    heatmap_generator.generate_ball_heatmap(ball_heatmap_path)

    print("Pipeline completed successfully!")
    print(f"Tracked video saved to: {output_video}")
    print(f"Player heatmap saved to: {player_heatmap_path}")
    print(f"Ball heatmap saved to: {ball_heatmap_path}")

if __name__ == "__main__":
    main()