import sys
from pathlib import Path
from predict import SentimentPredictor

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_PATH = PROJECT_ROOT / "sentiment_model.pth"


def main():
    print("=" * 60)
    print("IMDb Sentiment Analysis - Prediction")
    print("=" * 60)
    print("Enter a movie review to classify it as Positive or Negative.")
    print("Type 'quit' or 'exit' to stop.\n")

    try:
        predictor = SentimentPredictor(model_path=MODEL_PATH)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"Error while loading the model:\n{error}")
        return

    print(f"Model loaded successfully on: {predictor.device}")
    print(f"Vocabulary size: {len(predictor.word2idx)}\n")

    while True:
        try:
            review = input("Review: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nPrediction stopped.")
            break

        if review.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        if not review:
            print("Please enter a movie review.\n")
            continue

        try:
            result = predictor.predict(review)

            print(f"\nSentiment: {result['sentiment']}")
            print(f"Confidence: {result['confidence']:.2f}%")
            print(
                f"Negative: {result['negative_probability']:.2f}% | "
                f"Positive: {result['positive_probability']:.2f}%\n"
            )

        except ValueError as error:
            print(f"Error: {error}\n")
        except RuntimeError as error:
            print(f"Prediction error: {error}\n")


if __name__ == "__main__":
    main()
