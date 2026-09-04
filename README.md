# IMDb Sentiment Analysis with LSTM

A PyTorch-based Natural Language Processing project for binary sentiment classification on the IMDb movie review dataset. The project builds a vocabulary from the training reviews, converts text into indexed sequences, dynamically pads batches, trains an LSTM-based recurrent neural network, and provides a separate inference pipeline for classifying new movie reviews as **positive** or **negative**.

## Overview

The project implements an end-to-end sentiment-analysis pipeline using PyTorch without relying on a high-level text-classification framework.

The pipeline includes:

- Text tokenization and basic normalization
- Vocabulary construction from the training set
- Handling unknown words with `<unk>`
- Padding variable-length sequences with `<pad>`
- Word embeddings using PyTorch's `Embedding` layer
- Sequence modeling with LSTM
- Binary classification with a linear output layer
- Training with cross-entropy loss and the Adam optimizer
- Gradient clipping for training stability
- GPU acceleration when CUDA is available
- Saving the trained model together with the training vocabulary and model configuration
- A separate prediction interface for classifying new reviews

## Model Architecture

The default training configuration is:

| Component | Configuration |
|---|---|
| Embedding | 128 dimensions |
| Recurrent layer | LSTM |
| Hidden size | 128 |
| Number of layers | 1 |
| Dropout | 0.5 |
| Output classes | 2 |
| Batch size | 32 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Loss function | CrossEntropyLoss |
| Epochs | 100 |

The model also supports a standard RNN through the `cell_type` parameter. The default is `lstm`.

## Dataset

This project uses the **IMDb Large Movie Review Dataset**, which contains labeled movie reviews for binary sentiment classification.

Dataset reference:

- Stanford AI Lab: https://ai.stanford.edu/~amaas/data/sentiment/
- Original paper: Maas et al., *Learning Word Vectors for Sentiment Analysis*, ACL 2011.

The expected directory structure is:

```text
aclImdb/
├── train/
│   ├── pos/
│   └── neg/
└── test/
    ├── pos/
    └── neg/
```

> **Important:** The dataset is intentionally not included in this GitHub repository. Download and extract it separately from the official source above.

## Project Structure

```text
imdb-sentiment-analysis/
├── models/
│   ├── __init__.py
│   ├── model.py                 # LSTM/RNN sentiment classifier
│   └── preprocessing.py         # Tokenization, vocabulary, dataset, batching
│
├── train/
│   ├── __init__.py
│   ├── trainer.py               # Training and evaluation loop
│   └── main_trainer.py          # Training entry point
│
├── predicts/
│   ├── __init__.py
│   ├── predict.py               # Prediction logic
│   └── main_predict.py          # Interactive prediction interface
│
├── aclImdb/                     # Local dataset directory
├── sentiment_model.pth          # Trained checkpoint 
├── requirements.txt             # Python dependencies
├── .gitignore                   # Files excluded from Git
└── README.md                    # Project documentation
```

## Requirements

- Python 3.9+
- PyTorch
- pandas
- tqdm

For GPU training, install the appropriate PyTorch build for your CUDA version from the official PyTorch website:

https://pytorch.org/get-started/locally/

## Installation and Setup

Clone the repository:

```bash
git clone https://github.com/mahandhgh/IMDB-Sentiment-Analysis-LSTM.git
cd IMDB-Sentiment-Analysis-LSTM
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download and extract the IMDb dataset. The `aclImdb` directory should be located in the project root:

```text
imdb-sentiment-analysis/
├── aclImdb/
│   ├── train/
│   └── test/
├── models/
├── train/
└── predicts/
```

## Training

From the project root, run:

```bash
python -m train.main_trainer
```

The training pipeline:

1. Loads the IMDb training and test reviews.
2. Builds the vocabulary **only from the training reviews**.
3. Converts reviews to indexed sequences.
4. Dynamically pads each batch.
5. Trains the LSTM classifier.
6. Evaluates the model during training.
7. Saves the trained checkpoint as `sentiment_model.pth`.

### Model Checkpoint

The final `sentiment_model.pth` is saved as a checkpoint containing:

```text
model_state_dict
word2idx
model_config
```

This means the exact vocabulary mapping used during training is stored inside the checkpoint itself.

**No separate `vocab.json` file is required.**

Because the vocabulary is stored with the model, the prediction pipeline does not need to rebuild the vocabulary from the IMDb dataset.

## Prediction

After training, place the generated `sentiment_model.pth` in the project root:

```text
imdb-sentiment-analysis/
├── sentiment_model.pth
├── models/
├── train/
└── predicts/
```

The prediction pipeline does **not** require the `aclImdb` dataset.

Run the interactive predictor from the project root:

```bash
python predicts/main_predict.py
```

## How the Code Works

### 1. Data preprocessing

`models/preprocessing.py` reads positive and negative reviews from the IMDb folder structure. Reviews are converted to lowercase and tokenized using a simple regular-expression-based tokenizer.

### 2. Vocabulary

The vocabulary is created from the training reviews with `min_freq=2`.

Two special tokens are always included:

```text
<pad> -> 0
<unk> -> 1
```

Words that appear fewer than two times are excluded, while unseen words are mapped to `<unk>`.

### 3. Batch preparation

Reviews have different lengths, so each batch is padded to the length of its longest sequence. The original sequence lengths are also returned so that packed sequences can ignore padding during recurrent processing.

### 4. LSTM model

`models/model.py` maps token IDs to dense word embeddings. The embedded sequences are packed and processed by the recurrent layer. The final hidden representation is passed through dropout and a linear classifier to produce two logits:

```text
0 -> Negative
1 -> Positive
```

### 5. Training

`train/trainer.py` uses:

- Cross-entropy loss
- Adam optimization
- Gradient clipping with a maximum norm of `1.0`
- Accuracy as the main evaluation metric

### 6. Prediction

`predicts/predict.py` loads:

- The trained model weights
- The exact `word2idx` mapping used during training
- The stored model configuration

The input review is processed with the same tokenizer used during training, converted to the stored word indices, and passed to the LSTM. The predictor returns the predicted sentiment, confidence, and probabilities for both classes.

`predicts/main_predict.py` provides a simple interactive command-line interface.

## Current Experimental Setup

In the current implementation, the IMDb **test split** is passed to the trainer as `val_loader`. Therefore, the reported `Val Accuracy` corresponds to performance on the test split rather than on a separate validation split.

For a stricter machine-learning evaluation, a validation subset should be created from the training data and the test set should remain untouched until the final evaluation.

## Limitations and Possible Improvements

The current project uses a lightweight preprocessing pipeline and a relatively simple recurrent architecture. Possible improvements include:

- Creating a dedicated validation split from the training data
- Tracking precision, recall, and F1-score in addition to accuracy
- Adding reproducibility settings such as fixed random seeds
- Saving the best checkpoint instead of only the final trained checkpoint
- Using pretrained word embeddings
- Replacing the LSTM with a transformer-based model
- Adding a graphical or web-based interface for inference
