import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from pathlib import Path
import pandas as pd
import re
import glob


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    tokens = text.split()
    return tokens


class Vocabulary:
    def __init__(self, word2idx):
        self.word2idx = word2idx
        self.idx2word = {idx: word for word, idx in word2idx.items()}

        self.pad_idx = word2idx[PAD_TOKEN]
        self.unk_idx = word2idx[UNK_TOKEN]

    def __len__(self):
        return len(self.word2idx)

    def encode(self, tokens):
        return [
            self.word2idx.get(token, self.unk_idx)
            for token in tokens
        ]


class IMDBDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = tokenize(self.texts[idx])

        encoded = self.vocab.encode(tokens)

        sequence = torch.tensor(encoded, dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return sequence, label
    

def read_imdb_folder(folder_path, label):
    texts = []
    labels = []

    files = glob.glob(str(Path(folder_path) / "*.txt"))

    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        texts.append(text)
        labels.append(label)

    return texts, labels


def load_imdb_data(data_dir):
    data_dir = Path(data_dir)

    train_pos_texts, train_pos_labels = read_imdb_folder(
        data_dir / "train" / "pos", 1
    )

    train_neg_texts, train_neg_labels = read_imdb_folder(
        data_dir / "train" / "neg", 0
    )

    test_pos_texts, test_pos_labels = read_imdb_folder(
        data_dir / "test" / "pos", 1
    )

    test_neg_texts, test_neg_labels = read_imdb_folder(
        data_dir / "test" / "neg", 0
    )

    train_texts = train_pos_texts + train_neg_texts
    train_labels = train_pos_labels + train_neg_labels

    test_texts = test_pos_texts + test_neg_texts
    test_labels = test_pos_labels + test_neg_labels

    return train_texts, train_labels, test_texts, test_labels
    


def build_vocabulary(train_texts, min_freq=2):
    counter = Counter()

    for text in train_texts:
        tokens = tokenize(text)
        counter.update(tokens)

    word2idx = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1
    }

    for word, freq in counter.items():
        if freq >= min_freq:
            word2idx[word] = len(word2idx)

    vocab = Vocabulary(word2idx)

    return vocab

def collate_batch(batch):
    sequences = []
    labels = []

    for sequence, label in batch:
        sequences.append(sequence)
        labels.append(label)

    lengths = torch.tensor(
        [len(sequence) for sequence in sequences],
        dtype=torch.long
    )

    padded_sequences = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=0
    )

    labels = torch.tensor(labels, dtype=torch.long)

    return padded_sequences, lengths, labels

def create_imdb_dataloaders(data_dir=".", batch_size=32, min_freq=2):
    train_texts, train_labels, test_texts, test_labels = load_imdb_data(data_dir)

    vocab = build_vocabulary(train_texts, min_freq=min_freq)

    train_dataset = IMDBDataset(train_texts, train_labels, vocab)
    test_dataset = IMDBDataset(test_texts, test_labels, vocab)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch
    )

    return train_loader, test_loader, vocab


