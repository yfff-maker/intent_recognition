from config import LONG_DURATION_THRESHOLD, REPETITIVE_CLICK_THRESHOLD


class AnomalyDetector:
    def __init__(self):
        pass

    def detect_anomalies(self, behavior_sequence):
        """
        Analyzes a behavior sequence to find anomalous patterns based on the paper's rules.
        Returns a list of anomaly objects:
        {
            "type": "Repetitive Clicks",
            "timestamp": 12345,
            "description": "...",
            "involved_events": [...]
        }
        """
        anomalies = []

        # 1. Detect Repetitive Clicks (Rule: (Click, _, RO)+ or (Click, _, IO)+)
        # We look for consecutive events on the same widget
        i = 0
        while i < len(behavior_sequence):
            current_event = behavior_sequence[i]
            # Assuming 'Click' is implicit if widget is present or operationId suggests interaction
            # Adjust logic based on actual json structure.
            # Here we assume if 'widget' is not None/None, it's an interaction.

            widget = current_event.get("widget", "None")
            page = current_event.get("page", "None")

            if widget != "None":
                repetition_count = 1
                j = i + 1
                while j < len(behavior_sequence):
                    next_event = behavior_sequence[j]
                    if (
                        next_event.get("widget") == widget
                        and next_event.get("page") == page
                    ):
                        repetition_count += 1
                        j += 1
                    else:
                        break

                if repetition_count >= REPETITIVE_CLICK_THRESHOLD:
                    anomalies.append(
                        {
                            "type": "Repetitive Interaction",
                            "timestamp": current_event.get("startTimeTick", 0),
                            "description": f"User interacted with widget '{widget}' on page '{page}' {repetition_count} times in a row.",
                            "context_event": current_event,
                        }
                    )
                    i = j  # Skip processed events
                else:
                    i += 1
            else:
                i += 1

        # 2. Detect Long Duration (Hesitation)
        # Rule: Duration > Threshold
        for event in behavior_sequence:
            duration = event.get("duration", 0)
            if duration > LONG_DURATION_THRESHOLD:
                anomalies.append(
                    {
                        "type": "Long Duration / Hesitation",
                        "timestamp": event.get("startTimeTick", 0),
                        "description": f"User stayed on page '{event.get('page')}' for {duration}ms without effective progress.",
                        "context_event": event,
                    }
                )

        return anomalies
