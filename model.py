import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, num_layers=1, dropout=0.5, padding_idx=0, cell_type="lstm"):
       
        super().__init__()

        self.cell_type = cell_type.lower()
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )

        if self.cell_type == "rnn":
            self.rnn = nn.RNN(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        
        elif self.cell_type == "lstm":
            self.lstm = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )

        else:
            raise ValueError(f"Unsupported cell_type: {cell_type}")
        
        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, sequences, lenghts):
        embedded = self.embedding(sequences)

        packed = pack_padded_sequence(
            embedded,
            lenghts.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        if self.cell_type == "lstm":
            _, (hidden, _) = self.lstm(packed)

        else:
            _,hidden = self.rnn(packed)
        
        hidden = hidden[-1]
        hidden = self.dropout(hidden)

        logits = self.classifier(hidden)

        return logits
