import cv2
import os


class ContextBuilder:
    def __init__(self, frame_cache_dir):
        self.frame_cache_dir = frame_cache_dir
        if not os.path.exists(self.frame_cache_dir):
            os.makedirs(self.frame_cache_dir)

    def extract_frame(self, video_path, timestamp_ms, output_filename):
        """Extracts a frame from the video at the given timestamp"""
        if not video_path or not os.path.exists(video_path):
            return None

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
        success, image = cap.read()

        if success:
            output_path = os.path.join(self.frame_cache_dir, output_filename)
            cv2.imwrite(output_path, image)
            cap.release()
            return output_path

        cap.release()
        return None

    def build_prompt(self, task_info, anomaly, frame_path):
        """
        Constructs the prompt based on Table III in the paper.
        """

        # In a real implementation with GPT-4V, we would pass the image.
        # For GPT-3.5 (as in paper), we might need a textual description of the GUI.
        # Here we simulate the GUI description or assume the user provides it.

        gui_description = f"[GUI Screenshot extracted at {frame_path}]"

        prompt = f"""
You are a Requirement Analyst. Your goal is to analyze user behavior data to identify usability issues and elicit evolutionary requirements.

### Goal Context
- User Goal: {task_info['objective']}
- Expected Actions: {task_info['expected_actions']}

### GUI Information
- Interface Context: The user is currently interacting with the system.
- Visual Reference: {gui_description}
(Note: In a full multi-modal pipeline, the image content would be analyzed here.)

### Abnormal Pattern
- Type: {anomaly['type']}
- Description: {anomaly['description']}

### Task Instruction
Please analyze the causes of the abnormal behavior pattern based on the given information.
1. Identify reasons for the detected anomaly.
2. Provide specific improvement suggestions, such as optimizations for interface design or interaction mechanisms.
3. Output format: A numbered list of requirements. Each item must include:
   - Requirement Statement
   - Target UI Element
   - Rationale
"""
        return prompt
