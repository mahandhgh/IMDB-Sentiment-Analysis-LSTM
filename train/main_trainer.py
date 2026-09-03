import torch
from model.preprocessing import create_imdb_dataloaders
from model.model import SentimentRNN
from train.trainer import Trainer


train_loader, test_loader, vocab = create_imdb_dataloaders(
    data_dir=".",
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
    model.state_dict(),
    "sentiment_model.pth"
)

print("Model saved successfully.")
