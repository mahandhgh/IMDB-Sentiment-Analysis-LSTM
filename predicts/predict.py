import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.model import SentimentRNN
from models.preprocessing import tokenize

class SentimentPredictor:
    def __init__(self, model_path=None, device=None):
        self.model_path = (
            Path(model_path)
            if model_path is not None
            else PROJECT_ROOT / "sentiment_model.pth"
        )

        if not self.model_path.exists():
            raise FileNotFoundError(
                "Trained model was not found. Expected:\n"
                f"{self.model_path}\n\n"
                "Make sure sentiment_model.pth is in the project root."
            )

        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        try:
            checkpoint = torch.load(
                self.model_path,
                map_location=self.device,
                weights_only=True
            )
        except TypeError:
            checkpoint = torch.load(
                self.model_path,
                map_location=self.device
            )

        if not isinstance(checkpoint, dict):
            raise ValueError(
                "Invalid model checkpoint format.\n"
                "The model must contain model_state_dict, word2idx, "
                "and optionally model_config.\n"
                "Please retrain the model using the updated saving code."
            )

        if "model_state_dict" not in checkpoint:
            raise ValueError(
                "The checkpoint does not contain 'model_state_dict'.\n"
                "Please retrain the model using the updated saving code."
            )

        if "word2idx" not in checkpoint:
            raise ValueError(
                "The checkpoint does not contain 'word2idx'.\n"
                "The prediction code cannot reproduce the training vocabulary.\n"
                "Please retrain the model using the updated saving code."
            )

        self.word2idx = {
            str(word): int(index)
            for word, index in checkpoint["word2idx"].items()
        }

        self.pad_idx = self.word2idx.get("<pad>", 0)
        self.unk_idx = self.word2idx.get("<unk>", 1)

        model_config = checkpoint.get("model_config", {})

        embedding_dim = int(model_config.get("embedding_dim", 128))
        hidden_dim = int(model_config.get("hidden_dim", 128))
        num_layers = int(model_config.get("num_layers", 1))
        dropout = float(model_config.get("dropout", 0.5))
        padding_idx = int(model_config.get("padding_idx", self.pad_idx))
        cell_type = str(model_config.get("cell_type", "lstm"))

        self.model = SentimentRNN(
            vocab_size=len(self.word2idx),
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            padding_idx=padding_idx,
            cell_type=cell_type
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def encode(self, text):
        tokens = tokenize(text)

        if not tokens:
            raise ValueError(
                "The review does not contain any valid English words."
            )

        encoded = [
            self.word2idx.get(token, self.unk_idx)
            for token in tokens
        ]
        return torch.tensor(encoded, dtype=torch.long)

    def predict(self, text):
        sequence = self.encode(text)
        sequence = sequence.unsqueeze(0).to(self.device)

        lengths = torch.tensor([sequence.size(1)], dtype=torch.long)

        with torch.no_grad():
            logits = self.model(sequence, lengths)
            probabilities = torch.softmax(logits, dim=1)[0]

            predicted_class = int(torch.argmax(probabilities).item())

        sentiment = "Positive" if predicted_class == 1 else "Negative"
        confidence = float(probabilities[predicted_class].item() * 100)

        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "negative_probability": float(probabilities[0].item() * 100),
            "positive_probability": float(probabilities[1].item() * 100),
        }
