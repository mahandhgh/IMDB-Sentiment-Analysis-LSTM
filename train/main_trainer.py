import torch
from pathlib import Path
from models.preprocessing import create_imdb_dataloaders
from models.model import SentimentRNN
from train.trainer import Trainer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "aclImdb"

train_loader, test_loader, vocab = create_imdb_dataloaders(
    data_dir=DATA_DIR,
    batch_size=32
)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

model = SentimentRNN(
    vocab_size=len(vocab),
    embedding_dim=128,
    hidden_dim=128,
    num_layers=1,
    dropout=0.5,
    padding_idx=vocab.pad_idx,
    cell_type="lstm" # or rnn
)

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=test_loader,
    device=device
)

trainer.train(num_epochs=100)

print("Best Validation Accuracy:", trainer.best_val_accuracy)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "word2idx": vocab.word2idx,
        "model_config": {
            "embedding_dim": 128,
            "hidden_dim": 128,
            "num_layers": 1,
            "dropout": 0.5,
            "padding_idx": vocab.pad_idx,
            "cell_type": "lstm"
        }
    },
    "sentiment_model.pth"
)

print("Model and vocabulary saved successfully.")
