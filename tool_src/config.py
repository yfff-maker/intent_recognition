import os

# Configuration for the Requirements Elicitation Pipeline

# Paths
DATASET_ROOT = r"../anonymous_data"
OUTPUT_DIR = r"./output"
FRAME_CACHE_DIR = r"./output/frames"

# LLM Configuration (OpenRouter only)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
# Recommended by OpenRouter for attribution/analytics (can be any URL / app name)
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "http://localhost")
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "RequirementsElicitation")

# OpenRouter model name examples:
# - "openai/gpt-4o-mini"
# - "anthropic/claude-3.5-sonnet"
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")

# Interaction Mode: 'API' (Automated) or 'WEB_UI' (Manual via ChatGPT Website)
# Set to 'WEB_UI' to generate a guide for manual interaction instead of calling the API.
LLM_INTERACTION_MODE = "API"

# LLM Task
# - "REQUIREMENTS": reproduce original paper-style requirements elicitation
# - "INTENT": long-sequence intent inference with evidence tracing (this project extension)
LLM_TASK = "INTENT"

# Anomaly Detection Thresholds
REPETITIVE_CLICK_THRESHOLD = 3  # Number of clicks to consider repetitive
LONG_DURATION_THRESHOLD = 5000  # ms

# Long-Sequence Experiment Settings (A/B/C strategies)
# We control ONLY the context length; prompt skeleton, labels, retrieval top-k remain fixed.
WINDOW_MODE = "events"  # "events" (recommended) or "time"
STRATEGY_WINDOWS = {
    # Short (baseline): almost no history
    "A": {"k_left": 2, "k_right": 2, "time_left_ms": 3000, "time_right_ms": 3000},
    # Meso: pattern-level evidence
    "B": {"k_left": 20, "k_right": 20, "time_left_ms": 30000, "time_right_ms": 30000},
    # Long: global-in-sequence (still capped by key-event selection/compression)
    "C": {"k_left": 200, "k_right": 50, "time_left_ms": 180000, "time_right_ms": 30000},
}

# Key-event selection to control tokens
KEY_EVENT_TARGET_K = 600  # select up to K key events from full sequence (increased from 300)
KEY_EVENT_NUM_BINS = 12   # coverage bins across time
KEY_EVENT_TOP_M_PER_BIN = 40
KEY_EVENT_NEAR_DT_MS = 300  # de-duplicate near-duplicate key events by time proximity

# Prompt compression
COMPRESS_MERGE_CONSECUTIVE = True
PROMPT_MAX_EVENT_LINES = 120  # hard cap to avoid token explosion

# Memory (LTM) parameters
MEMORY_CHUNK_SIZE = 30  # reduced from 60 for finer granularity (30 events ≈ 1min activity)
MEMORY_MAX_ITEMS = 50
MEMORY_RETRIEVE_TOP_K = 5  # retrieves 5 out of ~20 chunks (25% selection rate)

# Intent label set (closed-set recommended for evaluation)
INTENT_LABELS = [
    "Login",
    "Navigate",
    "Search/Explore",
    "FillForm",
    "Upload/Download",
    "Submit/Confirm",
    "ErrorRecovery",
    "Waiting/NoFeedback",
    "Hesitation/Uncertainty",
    "Other",
]

# Task Context (Simulated for the purpose of reproduction)
# In a real scenario, this might come from a task log or experiment design doc
TASK_DEFINITIONS = {
    "Task1": {
        "objective": "Upload a courseware file to the system",
        "expected_actions": "Login -> Navigate to Course -> Click Upload -> Select File -> Confirm",
    },
    "Task2": {
        "objective": "Create a new student account",
        "expected_actions": "Navigate to User Management -> Click Add User -> Fill Form -> Submit",
    },
}
