# Goal-aware User Behavior Modeling via Multi-modal Data Fusion for Evolutionary Requirements Elicitation

This repository contains the replication package for the paper **"Goal-aware User Behavior Modeling via Multi-modal Data Fusion for Evolutionary Requirements Elicitation"**. It includes the anonymized multi-modal user behavior dataset and the source code for the LLM-based requirements elicitation approach.

## Repository Structure

- **`anonymous_data/`**: The anonymized dataset containing multi-modal user behavior data (logs, videos, interviews) from 20 participants.
- **`tool_src/`**: The source code implementing the proposed approach, including anomaly detection, context extraction, and LLM-based inference.

## 1. Multi-modal User Behavior Dataset

The dataset supports the empirical study and evaluation presented in the paper. It is located in the `anonymous_data/` directory.

### Content Overview
- **Participants**: Data from 20 anonymized participants (P1-P20).
- **Data Types**:
    - **Behavior Sequences (`behavior_sequences.json`)**: Formalized user action sequences (Action, Time, Object) used for pattern mining.
    - **Task Recordings (`task_recording_P{ID}.mp4`)**: Screen and facial recordings used for ground-truth verification and GUI context extraction.
    - **Raw Data (`raw_data/`)**: Original logs from eye-tracking and interaction logging tools.
    - **Interview Logs (`interview_log.txt`)**: Transcribed post-task interviews serving as the ground truth for user requirements.

For detailed documentation on the dataset structure and usage, please refer to [anonymous_data/README.md](anonymous_data/README.md).

## 2. Approach Reproduction Source Code

The `tool_src/` directory contains the implementation of the **LLM-based Evolutionary Requirements Elicitation** framework described in Section III of the paper.

### Key Modules
- **Anomaly Detection**: Implements the rule-based detection of behavioral anomalies (e.g., repetitive clicks, hesitation) as defined in the paper's "Anomalous Behavior Pattern Mining" section.
- **Context Builder**: Extracts visual context (GUI screenshots) from task recordings to construct multi-modal prompts.
- **LLM Inference**: Provides an interface to interact with Large Language Models (e.g., GPT-3.5/4) to infer evolutionary requirements from detected anomalies.

### Usage
The tool supports two modes of operation:
1.  **Automated API Mode**: Directly connects to OpenAI API for batch processing.
2.  **Manual Web UI Mode**: Generates a guide with prompts and screenshots for manual experimentation via the ChatGPT web interface.

For detailed instructions on how to configure and run the code, please refer to [tool_src/README.md](tool_src/README.md).
