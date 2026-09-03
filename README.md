# IMDb Sentiment Analysis with LSTM

A PyTorch-based Natural Language Processing project for binary sentiment classification on the IMDb movie review dataset. The project builds a vocabulary from the training reviews, converts text into indexed sequences, pads batches dynamically, and uses an LSTM-based recurrent neural network to classify reviews as **positive** or **negative**.

## Overview

The main goal of this project is to implement a complete sentiment-analysis pipeline without relying on a high-level text-classification framework.

The pipeline includes:

- Text tokenization and basic normalization
- Vocabulary construction from the training set
- Handling unknown words with an `<unk>` token
- Padding variable-length sequences with a `<pad>` token
- Word embeddings using PyTorch's `Embedding` layer
- Sequence modeling with LSTM
- Binary classification with a linear output layer
- Training with cross-entropy loss and the Adam optimizer
- Gradient clipping for training stability
- GPU acceleration when CUDA is available
- Saving the trained model weights to `sentiment_model.pth`

## Model Architecture

The default configuration in `main.py` is:

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

> **Important:** The dataset is intentionally not included in this GitHub repository. It is large and can be downloaded separately from the official source above.

## Project Structure

```text
imdb-sentiment-analysis/
├── data.py                  # Dataset loading, tokenization, vocabulary, batching
├── model.py                 # LSTM/RNN sentiment classifier
├── trainer.py               # Training and evaluation loop
├── main.py                  # Project entry point
├── sentiment_model.pth      # Trained model weights
├── requirements.txt         # Python dependencies
├── .gitignore               # Files excluded from Git
└── README.md                # Project documentation
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
git clone https://github.com/mahandhgh/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download and extract the IMDb dataset, then place the `aclImdb` directory so that it is available next to `main.py`.

For example:

```text
imdb-sentiment-analysis/
├── aclImdb/
│   ├── train/
│   └── test/
├── data.py
├── model.py
├── trainer.py
└── main.py
```

## Training

Run:

```bash
python main.py
```

During training, the program reports the loss and accuracy for each epoch. At the end, the best validation accuracy is displayed and the trained weights are saved as:

```text
sentiment_model.pth
```

## How the Code Works

### 1. Data preprocessing

`data.py` reads the positive and negative reviews from the IMDb folder structure. Reviews are converted to lowercase and tokenized using a simple regular-expression-based tokenizer.

### 2. Vocabulary

The vocabulary is created only from the training reviews. Words that appear fewer than `min_freq=2` times are excluded, while unseen words are mapped to `<unk>`.

### 3. Batch preparation

Reviews have different lengths, so each batch is padded to the length of its longest sequence. The original sequence lengths are also returned so that `pack_padded_sequence` can ignore padding during recurrent processing.

### 4. LSTM model

`model.py` first maps token IDs to dense word embeddings. The embedded sequences are packed and processed by the LSTM. The final hidden state is passed through dropout and a linear classifier to produce two logits: negative and positive.

### 5. Training

`trainer.py` uses:

- Cross-entropy loss
- Adam optimization
- Gradient clipping with a maximum norm of `1.0`
- Accuracy as the main evaluation metric

## Current Experimental Setup

In the current implementation, `main.py` passes the IMDb **test split** to the trainer as `val_loader`. Therefore, the reported `Val Accuracy` corresponds to performance on the test split rather than on a separate validation split.

For a stricter machine-learning evaluation, a validation subset should be created from the training data and the test set should remain untouched until the final evaluation.

## Trained Model

The repository can include the trained weights:

```text
sentiment_model.pth
```

The file contains the model's `state_dict`, not a complete serialized model object. To load it later, recreate the same `SentimentRNN` architecture and then call:

```python
model.load_state_dict(torch.load("sentiment_model.pth", map_location=device))
model.eval()
```

Because the model architecture depends on the vocabulary, the same vocabulary-building process must be used when reconstructing the model.

## Limitations and Possible Improvements

The current project uses a lightweight preprocessing pipeline and a relatively simple recurrent architecture. Possible improvements include:

- Creating a dedicated validation split from the training data
- Saving the vocabulary alongside the model
- Adding an inference script for predicting new reviews
- Using pretrained word embeddings or a transformer-based model
- Tracking precision, recall, and F1-score in addition to accuracy
- Adding reproducibility settings such as fixed random seeds
- Saving the best model checkpoint instead of only the final epoch weights


