# AppWorld White Agent

This repository contains the White Agent development branch, designed to solve complex multi-app tasks within the AppWorld simulated environment.

[!IMPORTANT] Version Note:

I will upload the complete execution logs for train, dev, test-n, and test-c datasets under the log/ directory to facilitate result verification and reproducibility.

Current Version (White Agent): This repository is dedicated to the core development and debugging of the White Agent. This specific version is not yet adapted for the AgentBeats framework.

AgentBeats Compatible Version (Green Agent): If you require the version that is fully compatible with AgentBeats for standard evaluation, please visit the Green Agent Repository.

---

## 🚀 Overview

The White Agent utilizes a Reasoning–Action loop powered by GPT-4o-mini and features:

* **Dynamic App Identification**: Analyzes task instructions to load only the necessary app tool schemas.
* **Stateful Token Management**: Intelligently persists `access_token` to avoid redundant logins unless a 401 error occurs.
* **Debugging Optimization**: Provides detailed `agent_trace.jsonl` logs capturing every reasoning step, action, and environment observation.

---

## 🛠️ Setup & Configuration

### 1. Prerequisites

* Python 3.10+
* OpenAI API Key

### 2. Install AppWorld

Since the AppWorld environment involves complex Docker containers or local database configurations, please refer to the official documentation for complete installation:

👉 **[AppWorld Official Installation Guide](https://github.com/StonyBrookNLP/appworld/tree/main?tab=readme-ov-file#trophy-leaderboard)**

### 3. Install Dependencies

Once the AppWorld environment is configured, install the required Python packages:

```bash
pip install openai appworld
```

### 4. Environment Variables

Ensure the necessary paths and keys are configured before running:

```bash
# Set your OpenAI API Key
export OPENAI_API_KEY='your-api-key-here'

# Set the AppWorld root directory path
export APPWORLD_ROOT='/path/to/your/appworld'
```

---

## 📋 Running & Evaluation Guide

### Execute White Agent

Run the evaluation script to initialize the environment and invoke the White Agent:

```bash
python run_agent.py
```

### Reproduce Evaluation Results

The `run_agent.py` script performs automated testing on the datasets. You can customize which datasets to test by modifying the `run_benchmark_suite` function:

```python
def run_benchmark_suite(task_limit_per_set=None):
    Config.setup()
    
    # You can toggle datasets here:
    # datasets = ["train", "dev"]
    datasets = ["test_normal", "test_challenge"]  # Default datasets
    ...
```

### Output and Monitoring

* **Report Generation**: A detailed JSON report (`benchmark_report_YYYYMMDD_HHMMSS.json`) is generated upon completion.
* **Summary Statistics**: The terminal will display a summary table featuring TGC (Task Goal Completion) and SGC (Scenario Goal Completion).
* **Trace Logs**: The agent's decision-making process is saved to `agent_trace.jsonl` for debugging.
