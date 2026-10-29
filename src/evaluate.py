import argparse
import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from model_scratch import PlantDiseaseCNN
from model_transfer import get_resnet_model

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

def evaluate_model(model, valid_loader, classes, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("Classification Report:\n")
    print(classification_report(all_labels, all_preds, target_names=classes))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "confusion_matrix.png")
    plt.show()

if __name__ == "__main__":
    from dataset import get_dataloaders

    parser = argparse.ArgumentParser(description="Evaluate a trained plant disease model.")
    parser.add_argument(
        "--model",
        choices=["scratch", "transfer"],
        default="scratch",
        help="Model checkpoint to evaluate (default: scratch)",
    )
    args = parser.parse_args()
    train_loader, valid_loader, classes = get_dataloaders()
    model_name = "cnn_scratch" if args.model == "scratch" else "resnet_transfer"
    checkpoint = MODELS_DIR / f"{model_name}_best.pth"
    if not checkpoint.exists():
        parser.error(f"Checkpoint not found: {checkpoint}")

    if args.model == "scratch":
        model = PlantDiseaseCNN(num_classes=len(classes))
    else:
        model = get_resnet_model(num_classes=len(classes), freeze_base=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )

    evaluate_model(model, valid_loader, classes, device=device)