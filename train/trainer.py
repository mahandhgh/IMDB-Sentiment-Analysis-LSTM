import torch
from torch import nn
from tqdm import tqdm
from models.preprocessing import create_imdb_dataloaders
from models.model import SentimentRNN

class Trainer:
    def __init__(self, model, train_loader, val_loader, device, learning_rate=1e-3):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.loss_fn = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.train_losses = []
        self.train_accuracy = []

        self.val_losses = []
        self.val_accuracy = []

        self.best_val_accuracy = 0.0

    def run_epoch(self, loader, training=True):

        if training:
            self.model.train()
        else:
            self.model.eval()

        running_loss = 0.0
        correct_predictions = 0
        total_examples = 0

        context = torch.enable_grad() if training else torch.no_grad()

        with context:

            for sequences, lengths, labels in tqdm(loader):

                sequences = sequences.to(self.device)

                lengths = lengths.to(self.device)

                labels = labels.to(self.device)

                outputs = self.model(sequences, lengths)

                loss = self.loss_fn(outputs, labels)

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    self.optimizer.step()

                predictions = torch.argmax(outputs, dim=1)

                correct_predictions += (predictions == labels).sum().item()

                total_examples += labels.size(0)

                running_loss += loss.item()

        epoch_loss = (running_loss / len(loader))

        epoch_accuracy = (correct_predictions / total_examples)

        return epoch_loss, epoch_accuracy
    
    def train(self, num_epochs):

        print("Training started...")

        for epoch in range(num_epochs):
            train_loss, train_acc = self.run_epoch(self.train_loader, training=True)
            val_loss, val_acc = self.run_epoch(self.val_loader, training=False)

            self.train_losses.append(train_loss)
            self.train_accuracy.append(train_acc)

            self.val_losses.append(val_loss)
            self.val_accuracy.append(val_acc)

            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc

            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f}"
            )
