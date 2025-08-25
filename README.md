# MLOps-Project - End-to-End ML System with Ray, MLflow, and FastAPI

This project demonstrates an **end-to-end MLOps pipeline** for training, tracking, and deploying a machine learning model using:


- **Ray** for distributed training and data processing  
- **PyTorch** for model development  
- **MLflow** for experiment tracking and model registry  
- **FastAPI + Ray Serve** for serving models via an API

---

## Features

- Distributed model training with Ray
- Configurable hyperparameters via CLI
- MLflow logging of metrics, parameters, and checkpoints
- Model checkpointing and retrieval from MLflow
- Inference and evaluation API built with FastAPI + Ray Serve
- Preprocessing pipeline with custom preprocessor
- JSON-based model predictions and probability outputs

---

## Project Structure

```
MLOps-Project/
├── data.py # Data loading, splitting, preprocessing
├── models.py # Finetuned LLM definition
├── train.py # Training pipeline with Ray + MLflow
├── predict.py # CLI-based inference and checkpoint loading
├── evaluate.py # Model evaluation functions
├── serve_app.py # API server for deployment (FastAPI + Ray Serve)
├── config.py # Configuration constants and logging
├── utils.py # Utility functions (saving, loading, formatting)
├── requirements.txt
└── README.md
```

---

## Requirements

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## 1. Train the Model

Run distributed training using Ray and MLflow:

```bash
python train.py \
  --experiment-name "llm" \
  --dataset-loc <data-set-location> \
  --train-loop-config "{\"dropout_p\": 0.5, \"lr\": 1e-4, \"lr_factor\": 0.8, \"lr_patience\": 3}" \
  --num-workers 1 \
  --cpu-per-worker 3 \
  --gpu-per-worker 0 \
  --num-epochs 10 \
  --batch-size 256 \
  --results-fp results/training_results.json
```

---

## 2. Predict Using the Trained Model

```bash
python predict.py predict \
  --run-id <MLFLOW_RUN_ID> \
  --title "My ML Project" \
  --description "A transformer-based architecture to classify tabular data"
```

---

## 3. Serve Model via API

```bash
python serve_app.py \
  --run_id <MLFLOW_RUN_ID> \
  --threshold 0.9
```
