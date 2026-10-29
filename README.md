#  Plant Disease Detection

A deep learning pipeline that classifies plant leaf diseases from images using **PyTorch**.  
Implements two architectures — a **custom CNN from scratch** and **ResNet18 transfer learning** — with a full inference UI built in **Streamlit**.

---

##  Project Structure

```
plant-disease-detection/
├── app/
│   └── app.py                 # Inference web app
├── data/
│   ├── raw/                   # Original PlantVillage dataset
│   └── processed/             # Subset used for training (auto-generated)
├── models/                    # Saved .pth checkpoints + plots
├── notebooks/
│   └── exploration.ipynb      # EDA notebook
├── src/
│   ├── dataset.py             # DataLoader setup
│   ├── make_subset.py         # Creates train/valid subset from raw data
│   ├── model_scratch.py       # Custom CNN architecture
│   ├── model_transfer.py      # ResNet18 transfer learning
│   ├── train.py               # Training loop (saves history + plots)
│   ├── evaluate.py            # Metrics, classification report, confusion matrix
│   └── run_training.py        # CLI entry point
└── requirements.txt
```

---

##  Dataset

[New Plant Diseases Dataset (Augmented)](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) — a subset of the **PlantVillage** dataset containing 87,000+ RGB leaf images across 38 disease/healthy classes.

The default workflow samples up to **300 training and 75 validation images per class** from all 38 classes (up to 11,400 train / 2,850 valid images). This includes tomato bacterial spot, healthy tomatoes, and other tomato diseases, so the classifier can compare these alternatives instead of forcing every tomato leaf into the bacterial-spot class.

---

##  Models

### CNN from Scratch (`model_scratch.py`)
A 4-block convolutional network built with PyTorch:
- 4x `Conv2d -> BatchNorm -> ReLU -> MaxPool` blocks (32 -> 64 -> 128 -> 256 filters)
- Fully connected classifier with dropout (0.5)
- Input: 224x224 RGB — Output: N class logits

### ResNet18 Transfer Learning (`model_transfer.py`)
- Pretrained on ImageNet via `torchvision.models`
- Trains the final FC layer first, then fine-tunes the backbone
- Typically converges faster than training a CNN from scratch

---

## Setup

```bash
# 1. Install dependencies
pip install torch torchvision scikit-learn matplotlib seaborn streamlit

# 2. Place the dataset at:
#    data/raw/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/

# 3. Create the processed subset (all 38 classes)
python src/make_subset.py --num-classes 38
# If the dataset is stored elsewhere, pass its root (the folder containing train/ and valid/):
# For the KaggleHub extraction used on Windows:
# python src\make_subset.py --num-classes 38 --source-base "C:\pd\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)"

# 4. Train a model
python src/run_training.py --model transfer --epochs 10 --lr 0.001

# 5. Evaluate the matching checkpoint
python src/evaluate.py --model transfer

# 6. Launch the web app
python -m streamlit run app.py
```

---

##  Results

Results are saved automatically to `models/<model_name>_history.csv` and plotted to `models/<model_name>_training_curve.png` after each training run. Previously reported scores were from a smaller 10-class experiment and should not be compared to the new 38-class run or treated as field accuracy.

---

##  Web App

Upload a leaf photo to compare the top-5 candidate classes. The app highlights possible tomato bacterial spot and provides general scouting guidance when that is the top match.

```bash
python -m streamlit run app.py
```

- Displays candidate scores and marks low-scoring predictions as uncertain
- Shows photo-quality tips and a clear screening-only warning
- A model score is not a calibrated probability. The dataset uses controlled images; validate results on field photos before relying on them.

---
