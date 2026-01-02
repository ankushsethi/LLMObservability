# LLMObservability (TinyLlama Local Setup - CPU Only)

This repository documents a minimal, reproducible setup for running the **TinyLlama 1.1B Chat** model locally on a CPU-only machine. The setup uses Python, Hugging Face tooling, and a virtual environment. No GPU or CUDA dependencies are required.

---

## Overview

- Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0  
- Runtime: CPU only  
- Platform: Windows  
- Environment isolation via Python virtual environment  

---

## System Requirements

- Windows 10 or later  
- Python 3.9 or newer  
- Visual Studio Code  
- Git  
- Stable internet connection (initial model download only)  
- Minimum 8 GB RAM recommended  

---

## Setup Steps

### 1. VS Code Preparation

- Install the **Python** extension in Visual Studio Code.
- Open this repository as a workspace.

---

### 2. Create a Virtual Environment

Open the VS Code terminal and run:

```powershell
python -m venv llm_env
```

### 3. Enable PowerShell Script Execution

Open PowerShell as Administrator and execute:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

This step is required once per user account.

### 4. Activate the Virtual Environment

Return to the VS Code terminal and run:

```powershell
llm_env\Scripts\activate
```

### 5. Install Dependencies

With the virtual environment active, install required packages:

```powershell
pip install torch transformers accelerate huggingface_hub
```

All dependencies are installed for CPU execution only.

### 6. Download the TinyLlama Model

Download the model locally using the Hugging Face CLI:
```powershell
huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --local-dir tinyllama
```

The model files will be stored in the tinyllama directory.

### 7. Run the Model Locally for Testing

Execute the inference script:
```powershell
python .\llm_env\run.py
```

The first run may take additional time due to model loading and initialization.

### 8. Running the FastAPI WebApp to validate Instrumentation

Install Openetelemetry dependencies
```powershell
pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp opentelemetry-instrumentation opentelemetry-instrumentation-fastapi  opentelemetry-instrumentation-asgi opentelemetry-instrumentation-requests
```

### 9. Run Model as an API with single /generate endpoint

```powershell
uvicorn app:app --host 0.0.0.0 --port 8000
``` 

Notes: 
No GPU, CUDA, or vendor-specific acceleration is used.
Performance depends entirely on CPU speed and available memory.
This setup is intended for experimentation, testing, and learning purposes.
