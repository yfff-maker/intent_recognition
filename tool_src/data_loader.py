import json
import os
import glob


class DataLoader:
    def __init__(self, dataset_root):
        self.dataset_root = dataset_root

    def get_participants(self):
        """Returns a list of participant IDs (e.g., ['P1', 'P2'])"""
        return [
            d
            for d in os.listdir(self.dataset_root)
            if os.path.isdir(os.path.join(self.dataset_root, d)) and d.startswith("P")
        ]

    def load_behavior_sequence(self, participant_id):
        """Loads the behavior_sequences.json for a given participant"""
        path = os.path.join(
            self.dataset_root, participant_id, "behavior_sequences.json"
        )
        if not os.path.exists(path):
            print(f"Warning: No behavior sequence found for {participant_id}")
            return []

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_video_path(self, participant_id):
        """Returns the path to the task recording video"""
        pattern = os.path.join(
            self.dataset_root, participant_id, f"task_recording_{participant_id}.mp4"
        )
        files = glob.glob(pattern)
        return files[0] if files else None
