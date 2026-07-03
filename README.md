# EEG-to-Text Translation Project

This repository contains a deep learning pipeline developed during my internship to decode electroencephalogram (EEG) brain signals directly into natural language text using a sequence-to-sequence transformer model.

# Project Structure

* **`eegdata.py`** – Handles data pipeline management, including EEG signal loading, preprocessing, tokenization, and dataset mapping.
* **`training.py`** – Implements the main training loop, hyperparameter scheduling, and model checkpoint saving.
* **`Inference.py`** – Loads the trained model weights to run predictions on raw/new EEG inputs and generate text sequences.
* **`evaluation.py`** – Evaluates model performance using sequence-to-sequence evaluation metrics like Word Error Rate (WER).

# Tech Stack & Architecture

* **Language:** Python 3
* **Frameworks:** PyTorch (`torch`), NumPy, SciPy, Pandas
* **Model Architecture:** Hugging Face Transformers (`BartForConditionalGeneration`, `BartTokenizer`)

# Training Hyperparameters

The model is configured with the following training parameters:
* **EEG Channels:** 128
* **Sequence Length:** 512
* **Model Dimension ($D_{model}$):** 256
* **BART Dimension:** 768
* **Batch Size:** 8
* **Epochs:** 300
* **Learning Rate:** 3e-4

# Model Artifacts
The final trained weights are exported and saved locally as `eeg_bart_model.pt` along with training logs saved to `training_log.csv`.

