<div align="center">

# 🎭 Real-Time Emotion Detection System

**7-class facial emotion recognition — custom SE-CNN · EfficientNetB2 transfer learning · ensemble · Grad-CAM · live webcam · Streamlit web app**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.55-red?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Best Accuracy](https://img.shields.io/badge/Best%20Accuracy-80.02%25-brightgreen)](#results)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

</div>

---

## Overview

A complete deep-learning pipeline for **real-time facial emotion recognition**, built as a university AI/Computer Vision project. Starting from a DeepFace zero-shot baseline at 52.80%, the system progressively improves through a custom multi-scale SE-CNN trained from scratch, MobileNetV2 transfer learning, a 4-model soft-voting ensemble, and finally EfficientNetB2 fine-tuned with progressive layer unfreezing — reaching **80.02% test accuracy** on RAF-DB, a **+27.22%** improvement over the baseline.

---

## Results

| Model | Test Accuracy |
|---|---|
| DeepFace pretrained (zero-shot baseline) | 52.80% |
| MobileNetV2 SWA + TTA | 68.19% |
| Custom SE-CNN v1 Best + TTA | 69.42% |
| Custom SE-CNN v2 Fine-tuned + TTA | 69.73% |
| 4-Model Soft-Voting Ensemble + TTA | 70.45% |
| **EfficientNetB2 Fine-tuned + TTA** | **80.02%** |

All custom models evaluated on the full RAF-DB test set (3,068 images) with 8-pass Test-Time Augmentation.

**EfficientNetB2 per-class breakdown:**

| Emotion | Precision | Recall | F1 |
|---|---|---|---|
| Happy | 0.91 | 0.92 | 0.91 |
| Surprise | 0.80 | 0.76 | 0.78 |
| Angry | 0.80 | 0.64 | 0.71 |
| Neutral | 0.72 | 0.80 | 0.76 |
| Sad | 0.74 | 0.81 | 0.77 |
| Fear | 0.63 | 0.49 | 0.55 |
| Disgust | 0.60 | 0.33 | 0.42 |

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

Five tabs:

- **📷 Live Demo** — upload an image or take a webcam snapshot; emotion labels, probability bars, and Grad-CAM heatmaps per detected face. Optional side-by-side DeepFace comparison.
- **🎬 Video Analysis** — upload any video file; emotion timeline, stacked area chart, dominant-emotion colour strip, and frame-level distribution.
- **📊 Model Comparison** — benchmark bar chart of all models vs the DeepFace baseline with a sortable accuracy table.
- **📈 Training Curves** — accuracy and loss curves for every training phase loaded live from CSV logs.
- **ℹ️ About** — architecture overview, dataset details, and command reference.

### Real-Time Webcam

```bash
# Standard mode — Grad-CAM overlay + probability sidebar
python3 realtime/realtime_webcam.py

# Side-by-side comparison with DeepFace
python3 realtime/realtime_webcam.py --compare

# Choose a specific model or face detector
python3 realtime/realtime_webcam.py --model saved_models/efficientnetb2_best.keras --detector mtcnn
```

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `G` | Toggle Grad-CAM overlay |

---

## Architecture

### Custom SE-CNN — MultiScale Squeeze-and-Excitation CNN

```
Input 48×48×1 (grayscale)
│
├── Branch 1 (3×3 convolutions)
├── Branch 2 (5×5 convolutions)  ──► Concatenate ──► Residual blocks + SE attention
└── Branch 3 (7×7 convolutions)
                                               │
                                GlobalAveragePooling
                                               │
                                    Dense(256) + Dropout
                                               │
                                       Softmax (7 classes)
```

- **Squeeze-and-Excitation (SE)** attention — learned per-channel recalibration
- **Residual connections** throughout to prevent gradient vanishing
- **He initialisation** + **Batch Normalisation** on every conv layer
- ~3.2M parameters · input: 48 × 48 × 1 (grayscale)

### EfficientNetB2 — Best Model (80.02%)

ImageNet-pretrained backbone fine-tuned on RAF-DB at 96×96 RGB with a 4-phase progressive unfreezing schedule:

| Phase | Backbone layers unfrozen | Epochs | LR | Best val acc |
|---|---|---|---|---|
| 1 — Head only | 0% (frozen) | 15 | 1e-3 | 57.41% |
| 2 — Top 40% | 40% | 20 | 2e-4 | 72.72% |
| 3 — Full fine-tune | 100% | 10 | 5e-5 | 74.59% |
| 4 — SWA | 100% | 8 | 1e-5 | 75.16% |

With 8-pass TTA: **80.02% test accuracy** — +9.57% over the 4-model ensemble.

- Mixed precision training (float16) for speed on RTX 4050
- uint8 in-memory storage (~300 MB RAM) with lazy float32 casting per batch
- Gradient clipping (clipnorm=1.0) throughout all phases

### Training Pipeline (Custom CNN)

```
RAF-DB + FER2013 + Extra
         │
         ├── Per-image standardisation (zero-mean, unit-variance)
         ├── Geometric augmentation (flip, rotate, zoom, translate)
         │    └── ⚠ Colour augmentation excluded — incompatible with
         │         per-image standardisation (causes training collapse)
         ├── Minority class oversampling → 12,000 samples per class
         │
         ├── Phase 1 — 100 epochs, CosineDecay LR (1e-3 → ~0)
         ├── Phase 2 — 30-epoch fine-tune from best checkpoint (lr=2e-4)
         └── Phase 3 — 8-epoch SWA pass
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
│   └── app.py                   Streamlit web app (5 tabs)
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
│   ├── custom_cnn.py            MultiScale SE-CNN architecture definition
│   └── ensemble.py              Soft-voting ensemble + evaluate_ensemble()
├── training/
│   ├── train.py                 Main training script (custom CNN + MobileNetV2)
│   └── callbacks.py             ModelCheckpoint, EarlyStopping, SWA, CSVLogger
├── realtime/
│   └── realtime_webcam.py       Live webcam demo (Grad-CAM, timeline, compare mode)
├── analysis/
│   └── meeting_analyzer.py      Video mood analytics
├── logs/                        CSV training history files (all phases)
├── saved_models/                Trained .keras model files (git-ignored, ~35–73 MB each)
├── finetune_cnn.py              Fine-tune custom CNN from best checkpoint + SWA
├── finetune_efficientnet.py     4-phase EfficientNetB2 fine-tune on RAF-DB → 80.02%
├── evaluate_final.py            Full benchmark: all models + TTA + ensemble + DeepFace
├── benchmark_deepface.py        Standalone DeepFace accuracy benchmark
├── requirements.txt
└── train.sh                     One-shot training shell script
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/charbelmezeraani0-star/facial-emotion-recognition.git
cd facial-emotion-recognition
pip install -r requirements.txt
pip install tf-keras deepface   # DeepFace comparison features
```

### 2. Prepare datasets

```
data/
├── raf/          RAF-DB (aligned 100×100 JPEGs, folder structure: train/1..7, test/1..7)
└── extra/        Any supplementary dataset (same folder structure as RAF-DB)
```

The custom CNN dataloader caches to `data/processed/dataset_cache.npz` on first run. EfficientNetB2 fine-tuning loads RAF-DB directly from raw files.

### 3. Train

```bash
# Custom CNN — 100 epochs from scratch + MobileNetV2 transfer learning
python3 training/train.py

# Fine-tune custom CNN from best checkpoint (30 epochs + SWA)
python3 finetune_cnn.py

# Fine-tune EfficientNetB2 — 4-phase progressive unfreezing (~2h on RTX 4050)
python3 finetune_efficientnet.py
```

### 4. Evaluate

```bash
# Full benchmark: all models + TTA + ensemble combos + DeepFace comparison
python3 evaluate_final.py

# DeepFace accuracy only (fast, 500-sample estimate)
python3 benchmark_deepface.py
```

---

## Key Technical Decisions

### Why no brightness/contrast augmentation?

Per-image standardisation (zero-mean, unit-variance) is applied during preprocessing. Adding `RandomBrightness` or `RandomContrast` on top of already-standardised images creates contradictory gradients — the model gets stuck at the 7-class random baseline (14.3%) for the entire training run.

**Fix:** geometric augmentations only (flip, rotate, zoom, translate).

### Why SWA?

Stochastic Weight Averaging averages weights from the flat region of the loss landscape instead of a single minimum. This consistently improves generalisation by 0.5–1% with no extra training data.

### Why TTA?

Running 8 random augmentation passes at inference time and averaging the softmax outputs reduces prediction variance. Each model gains +1–2.5% accuracy with TTA at zero training cost.

### Why an ensemble?

The 4 models (MobileNetV2 SWA, SE-CNN v1 Best, SE-CNN v2 Best, SE-CNN v2 SWA) make partially independent errors due to different architectures and training trajectories. Soft-voting their TTA outputs yields **70.45%** — higher than any single model alone.

### Why does EfficientNetB2 outperform the ensemble?

EfficientNetB2 pretrained on ImageNet provides far richer feature representations than a 3.2M-parameter CNN trained from scratch on ~11k samples. ImageNet pretraining supplies texture, edge, and shape detectors that transfer well to facial expression recognition. Progressive layer unfreezing (head → top 40% → full backbone) prevents catastrophic forgetting while adapting to emotion-specific features. The result: **80.02% with TTA** — beating the 4-model ensemble by **+9.57%**.

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
| `mtcnn` | Accurate face detection (alternative to Haar cascade) |
| `matplotlib / pandas` | Visualisation and data handling |

Full list: [`requirements.txt`](requirements.txt)

---

## License

MIT — see [`LICENSE`](LICENSE) for details.

---

<div align="center">
Built for a university AI / Computer Vision course · TensorFlow 2.21 · RTX 4050 Laptop GPU
</div>
