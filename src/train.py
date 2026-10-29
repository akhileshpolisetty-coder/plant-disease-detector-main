import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import os
import matplotlib
matplotlib.use("Agg")  # headless — no display required
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

def train_model(model, train_loader, valid_loader, num_epochs=5, lr=0.001, device=None, model_name="model"):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    best_val_acc = 0.0
    history = []  # list of dicts, one per epoch
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Halve LR if val_acc doesn't improve for 3 epochs — prevents overshooting
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    for epoch in range(num_epochs):
        start_time = time.time()

        # --- Training phase ---
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # --- Validation phase ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.2e} | Time: {elapsed:.1f}s")

        scheduler.step(val_acc)  # adjust LR based on validation accuracy

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss,   "val_acc": val_acc,
        })

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODELS_DIR / f"{model_name}_best.pth")
            print(f"  -> Saved new best model (val_acc: {val_acc:.4f})")

    # ── save history CSV ────────────────────────────────────────────────────
    csv_path = MODELS_DIR / f"{model_name}_history.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        writer.writeheader()
        writer.writerows(history)
    print(f"History saved -> {csv_path}")

    # ── plot training curves ─────────────────────────────────────────────────
    epochs_range = [h["epoch"] for h in history]
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs_range, [h["train_loss"] for h in history], label="Train", marker="o")
    ax1.plot(epochs_range, [h["val_loss"]   for h in history], label="Val",   marker="o")
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend(); ax1.grid(True)

    ax2.plot(epochs_range, [h["train_acc"] for h in history], label="Train", marker="o")
    ax2.plot(epochs_range, [h["val_acc"]   for h in history], label="Val",   marker="o")
    ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.legend(); ax2.grid(True)

    plt.suptitle(f"{model_name} — Training Curves", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plot_path = MODELS_DIR / f"{model_name}_training_curve.png"
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"Training curve saved -> {plot_path}")

    return model