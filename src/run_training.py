import argparse
import json
import torch
from pathlib import Path
from dataset import get_dataloaders
from model_scratch import PlantDiseaseCNN
from model_transfer import get_resnet_model
from train import train_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["scratch", "transfer"], required=True,
                        help="Which model to train: 'scratch' (CNN from scratch) or 'transfer' (pretrained ResNet)")
    parser.add_argument("--epochs",         type=int,   default=15,     help="Total training epochs (default: 15)")
    parser.add_argument("--lr",             type=float, default=0.001,  help="Initial learning rate (default: 0.001)")
    parser.add_argument("--finetune-after", type=int,   default=5,
                        help="[transfer only] Unfreeze all layers after this many epochs for fine-tuning (default: 5). Set 0 to skip.")
    args = parser.parse_args()

    print("Loading data...")
    train_loader, valid_loader, classes = get_dataloaders()
    num_classes = len(classes)
    print(f"Found {num_classes} classes.")

    models_dir = Path(__file__).resolve().parent.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    with (models_dir / "class_names.json").open("w", encoding="utf-8") as class_file:
        json.dump(classes, class_file, indent=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    # ── CNN from scratch ──────────────────────────────────────────────────────
    if args.model == "scratch":
        model = PlantDiseaseCNN(num_classes=num_classes)
        train_model(
            model, train_loader, valid_loader,
            num_epochs=args.epochs, lr=args.lr,
            device=device, model_name="cnn_scratch"
        )

    # ── ResNet18 transfer + fine-tuning ───────────────────────────────────────
    else:
        finetune_after = args.finetune_after

        if finetune_after > 0 and finetune_after < args.epochs:
            # ── Phase 1: frozen backbone, train head only ──
            phase1_epochs = finetune_after
            print(f"\n{'='*55}")
            print(f" Phase 1 — Frozen backbone ({phase1_epochs} epochs, lr={args.lr})")
            print(f"{'='*55}\n")

            model = get_resnet_model(num_classes=num_classes, freeze_base=True)
            model = train_model(
                model, train_loader, valid_loader,
                num_epochs=phase1_epochs, lr=args.lr,
                device=device, model_name="resnet_transfer_phase1"
            )

            # ── Phase 2: unfreeze everything, fine-tune with lower LR ──
            phase2_epochs = args.epochs - finetune_after
            fine_lr = args.lr / 10
            print(f"\n{'='*55}")
            print(f" Phase 2 — Full fine-tuning ({phase2_epochs} epochs, lr={fine_lr})")
            print(f"{'='*55}\n")

            for param in model.parameters():
                param.requires_grad = True          # unfreeze all layers

            model = train_model(
                model, train_loader, valid_loader,
                num_epochs=phase2_epochs, lr=fine_lr,
                device=device, model_name="resnet_transfer"
            )

        else:
            # single-phase: just train the head (or full if finetune_after=0 and freeze_base=False)
            model = get_resnet_model(num_classes=num_classes, freeze_base=(finetune_after > 0))
            train_model(
                model, train_loader, valid_loader,
                num_epochs=args.epochs, lr=args.lr,
                device=device, model_name="resnet_transfer"
            )

    print("\nTraining complete.")

if __name__ == "__main__":
    main()