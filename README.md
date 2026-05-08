<div align="center">

# 🎭 Real-Time Emotion Detection System

**7-class facial emotion recognition — from scratch CNN · transfer learning · ensemble · live webcam · web app**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.55-red?logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

</div>

---

## Overview

A full deep-learning pipeline for **real-time facial emotion recognition**, built as a university AI/Computer Vision project. The system trains a custom multi-scale CNN with Squeeze-and-Excitation attention from scratch, fine-tunes MobileNetV2 via transfer learning, and combines them into a 4-model soft-voting ensemble — beating a DeepFace pretrained baseline by **+17.65%** accuracy.

**Key results:**

| Model | Test Accuracy |
|---|---|
| DeepFace pretrained (zero-shot baseline) | 52.80% |
| MobileNetV2 SWA + TTA | 68.19% |
| Custom CNN v1 Best + TTA | 69.42% |
| Custom CNN v2 Fine-tuned + TTA | 69.73% |
| **4-Model Ensemble + TTA** | **70.45%** |

---

## Emotions Detected

| 😠 Angry | 🤢 Disgust | 😨 Fear | 😄 Happy | 😢 Sad | 😲 Surprise | 😐 Neutral |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|

---

## Demo

### Web App (Streamlit)

```bash
streamlit run app/app.py
```

The web app offers four tabs:

- **📷 Live Demo** — upload an image or take a webcam snapshot; see emotion labels, probability bars, and Grad-CAM heatmaps per face. Optional side-by-side DeepFace comparison.
- **🎬 Video Analysis** — upload any video file; get an emotion timeline, stacked area chart, dominant-emotion colour strip, and frame-level distribution breakdown.
- **📊 Model Comparison** — benchmark bar chart of all models vs the DeepFace baseline, with a sortable accuracy table.
- **📈 Training Curves** — live-loaded accuracy and loss curves for every training phase from CSV logs.

### Real-Time Webcam

```bash
# Standard mode — Grad-CAM overlay + probability sidebar
python3 realtime/realtime_webcam.py

# Compare mode — our model vs DeepFace side-by-side
python3 realtime/realtime_webcam.py --compare

# Choose model or detector
python3 realtime/realtime_webcam.py --model saved_models/custom_cnn_v2_best.keras --detector mtcnn
```

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `G` | Toggle Grad-CAM overlay |

---

## Architecture

### Custom CNN — MultiScale SE-CNN

```
Input 48×48×1
│
├── Branch 1 (3×3 convolutions)
├── Branch 2 (5×5 convolutions)  → Concatenate → Residual blocks + SE attention
└── Branch 3 (7×7 convolutions)
                                            │
                             GlobalAveragePooling → Dense → Softmax (7)
```

- **Squeeze-and-Excitation (SE)** attention blocks — learned channel recalibration
- **Residual connections** throughout to prevent gradient vanishing
- **He initialisation** + **Batch Normalisation** on every conv layer
- ~3.2M parameters · input: 48 × 48 × 1 (grayscale)

### Training Pipeline

```
Raw Data
   │
   ├── Per-image standardisation (zero-mean, unit-variance → [0,1])
   ├── Geometric augmentation only (flip, rotate, zoom, translate)
   │    └── ⚠️ Colour augmentations deliberately excluded — incompatible
   │         with per-image standardisation (causes training collapse)
   ├── Oversampling minority classes → 12,000 samples each
   │
   ├── Phase 1 — 100 epochs CosineDecay LR (1e-3 → ~0)
   ├── Phase 2 — 30-epoch fine-tune from best checkpoint (lr = 2e-4)
   └── Phase 3 — 8-epoch SWA pass (weight averaging for better generalisation)
                        │
            Test-Time Augmentation (8 random passes)
                        │
            Soft-voting ensemble (4 models)
                        │
                   70.45% accuracy
```

---

## Project Structure

```
cv-project/
├── app/
│   └── app.py                   Streamlit web application (4 tabs)
├── data/
│   ├── dataloader.py            Dataset loading, augmentation, TTA, oversampling
│   └── processed/               Cached preprocessed arrays (auto-generated)
├── detection/
│   ├── face_detector.py         Haar cascade + MTCNN face detection
│   ├── face_tracker.py          Multi-face tracker with smoothing
│   ├── detect_image.py          CLI: emotion detection on a single image
│   └── detect_video.py          CLI: emotion detection on a video file
├── evaluation/
│   ├── gradcam.py               Grad-CAM computation and overlay
│   ├── evaluate.py              Per-model evaluation + confusion matrix
│   └── plots/                   Saved training curve PNGs
├── models/
│   ├── custom_cnn.py            MultiScale SE-CNN architecture
│   └── ensemble.py              Soft-voting ensemble + evaluate_ensemble()
├── training/
│   ├── train.py                 Main training script (custom CNN + MobileNetV2)
│   └── callbacks.py             ModelCheckpoint, EarlyStopping, SWA, CSVLogger
├── realtime/
│   └── realtime_webcam.py       Live webcam demo (Grad-CAM, timeline, compare mode)
├── analysis/
│   └── meeting_analyzer.py      Video mood analytics
├── logs/                        CSV training history files
├── saved_models/                Trained .keras model files
├── finetune_cnn.py              Fine-tune from best checkpoint + SWA
├── evaluate_final.py            Full benchmark: all models + TTA + ensemble + DeepFace
├── benchmark_deepface.py        Standalone DeepFace accuracy benchmark
├── requirements.txt
└── train.sh                     One-shot training shell script
```

---

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/<your-username>/emotion-detection.git
cd emotion-detection
pip install -r requirements.txt
pip install tf-keras deepface   # for DeepFace comparison features
```

### 2. Prepare datasets

Place the RAF-DB and/or FER2013 data in the expected directories:

```
data/
├── raf/          RAF-DB images (aligned, 100×100 JPEGs + EmoLabel/)
└── extra/        Any supplementary dataset (same folder structure as RAF-DB)
```

The dataloader auto-detects and preprocesses on first run, then caches to `data/processed/dataset_cache.npz` for fast subsequent loads.

### 3. Train

```bash
# Full training pipeline (custom CNN 100 epochs + MobileNetV2 transfer learning)
python3 training/train.py

# Fine-tune custom CNN from best checkpoint (30 more epochs + SWA)
python3 finetune_cnn.py
```

Or use the one-shot script:

```bash
bash train.sh
```

### 4. Evaluate

```bash
# Full benchmark: all 4 models + TTA + 8 ensemble combos + DeepFace comparison
python3 evaluate_final.py

# DeepFace accuracy only (fast, 500-sample estimate)
python3 benchmark_deepface.py
```

---

## Key Technical Decisions

### Why no brightness/contrast augmentation?

Per-image standardisation (zero-mean, unit-variance) is applied during preprocessing. Adding `RandomBrightness` or `RandomContrast` augmentation on top of already-standardised images creates contradictory gradients that prevent learning entirely — the model gets stuck at the 7-class random baseline (14.3%) for the entire training run.

**Fix:** geometric augmentations only (flip, rotate, zoom, translate).

### Why SWA?

Stochastic Weight Averaging averages model weights from the flat region of the loss landscape rather than using a single minimum. This consistently improves generalisation by 0.5–1% with no extra training data.

### Why TTA?

Running 8 random augmentation passes at inference time and averaging the softmax outputs reduces variance. Each model gains +1–2.5% accuracy with TTA enabled at no training cost.

### Why an ensemble?

The 4 models (MobileNetV2 SWA, CNN v1 Best, CNN v2 Best, CNN v2 SWA) make partially independent errors due to different architectures and training trajectories. Soft-voting their TTA outputs yields **70.45%** — higher than any single model.

---

## Dependencies

| Package | Purpose |
|---|---|
| `tensorflow >= 2.21` | Model training and inference |
| `opencv-python` | Face detection, frame capture, image processing |
| `streamlit >= 1.55` | Web application |
| `deepface` | Pretrained comparison baseline |
| `tf-keras` | Required by DeepFace under TF 2.21 |
| `scikit-learn` | Metrics, classification report |
| `mtcnn` | Accurate face detection (alternative to Haar) |
| `matplotlib / pandas` | Visualisation and data handling |

Full list: [`requirements.txt`](requirements.txt)

---

## License

MIT — see [`LICENSE`](LICENSE) for details.

---

<div align="center">
Built for a university AI / Computer Vision course · TensorFlow 2.21 · RTX 4050
</div>
