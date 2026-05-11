# Real-Time Facial Emotion Recognition
## A Progressive Deep Learning Approach: From Baseline to EfficientNetB2

---

| | |
|:---|:---|
| **Course** | Artificial Intelligence and Computer Vision |
| **University** | [University Name] |
| **Student(s)** | [Student Name(s)] |
| **Instructor** | [Instructor Name] |
| **Submission Date** | May 2026 |

---

> *All experiments were conducted on an NVIDIA RTX 4050 Laptop GPU using TensorFlow 2.21 and Python 3.10.*

\newpage

## Abstract

Facial Emotion Recognition (FER) is a core problem in affective computing that requires automatically inferring emotional state from facial imagery. This report presents the design, implementation, evaluation, and deployment of a complete FER pipeline trained on the RAF-DB dataset across seven emotion classes: angry, disgust, fear, happy, sad, surprise, and neutral. The project follows a principled progression: a zero-shot DeepFace baseline (52.80%) establishes the task difficulty, followed by MobileNetV2 transfer learning (68.19%), a purpose-built Multi-Scale Squeeze-and-Excitation CNN (SE-CNN, 69.73%), a 4-model soft-voting ensemble (70.45%), and finally EfficientNetB2 fine-tuned with a 4-phase progressive unfreezing schedule, reaching **80.02%** on the 3,068-image RAF-DB test set — a **+27.22%** absolute improvement. Key contributions include the SE-CNN architecture combining three convolutional scales with learned channel attention; the identification of a critical training instability caused by applying photometric augmentation after per-image standardization; a systematic fine-tuning protocol with Stochastic Weight Averaging (SWA) and 8-pass Test-Time Augmentation (TTA); Grad-CAM interpretability confirming anatomically meaningful attention; and a full-stack deployment comprising a Streamlit web application, real-time webcam inference, video analysis tools, and TFLite edge export.

\newpage

## Table of Contents

1. Introduction
2. Dataset and Preprocessing
3. Proposed System Architecture
4. Model Architectures
5. Training Protocol and Experimental Setup
6. Results and Analysis
7. Grad-CAM Interpretability
8. Deployment and Real-Time Application
9. Problems Faced, Solutions, and Lessons Learned
10. Limitations and Future Work
11. Conclusion
12. References

\newpage

## 1. Introduction

### 1.1 Problem and Motivation

Facial expressions are among the most information-dense channels of human communication, carrying emotional content that is consistent across cultures and immediately interpretable. Automating the recognition of these expressions has direct application in human-computer interaction (adaptive interfaces, tutoring systems), healthcare (pain monitoring, affective disorder assessment), driver monitoring (fatigue and distraction detection), smart classrooms (engagement tracking), and meeting analytics (mood and participation reporting).

Despite these applications, automated FER is technically difficult. The primary challenges are: illumination and pose variation that alter facial appearance independently of expression; partial occlusion by hair, glasses, or hands; severe class imbalance (happy and neutral dominate naturalistic data while disgust and fear are rare); low inter-class variance (fear and surprise share raised brows and widened eyes); and annotation ambiguity at low expression intensity. These challenges make raw accuracy a necessary but insufficient evaluation metric.

### 1.2 Project Objectives

This project sets out to build a complete FER system that is both high-performing and genuinely understood. The four concrete objectives are:

1. Establish a zero-shot baseline to quantify task difficulty without training.
2. Progress through multiple model architectures — lightweight transfer learning, a custom architecture, and a high-capacity fine-tuned model — comparing each objectively.
3. Analyze results at the per-class level to understand what the system does and does not handle well.
4. Deploy the system as a real-time, interpretable application.

**Table 1 — Project Contribution Summary**

| Contribution | Description | Why It Matters |
|:------------|:------------|:---------------|
| DeepFace zero-shot baseline | Evaluated directly on RAF-DB without training | Establishes task difficulty; reference for all improvements |
| MobileNetV2 transfer learning | ImageNet pretrained backbone adapted to RAF-DB | Shows value of pretrained features over random init |
| Custom Multi-Scale SE-CNN | 3×3/5×5/7×7 branches + SE attention, trained from scratch | Purpose-built for FER; demonstrates architectural reasoning |
| 4-model soft-voting ensemble | Aggregates MobileNetV2 + SE-CNN variants | Exploits model diversity to reduce prediction variance |
| EfficientNetB2 fine-tuning | 4-phase progressive unfreezing, 80.02% | Best result; shows power of large-scale pretraining |
| Augmentation instability fix | Removed photometric augmentation conflicting with standardization | Critical engineering insight; generalizable lesson |
| Grad-CAM interpretability | Visualizes model attention on facial regions | Validates that model attends to emotionally relevant areas |
| Full deployment stack | Streamlit, webcam, CLI, video, TFLite export | Proves the system is usable beyond a training experiment |

\newpage

## 2. Dataset and Preprocessing

### 2.1 RAF-DB Dataset

The **Real-world Affective Faces Database (RAF-DB)** is collected from the internet under unconstrained real-world conditions, covering natural variation in illumination, pose, age, ethnicity, and image quality. Aligned facial images are provided at 100×100 pixels. The seven basic emotion categories are listed in Table 2 alongside their primary facial muscle actions.

**Table 2 — Seven Emotion Classes in RAF-DB**

| Emotion | Primary Visual Cue | Training Frequency |
|:--------|:------------------|:-------------------|
| Happy | Raised cheeks, visible teeth, crow's feet | High |
| Neutral | Absence of distinctive deformations | High |
| Sad | Inner brow raise, lip corner depression | Moderate |
| Angry | Lowered brows, tightened lips | Moderate |
| Surprise | Raised brows, widened eyes, dropped jaw | Moderate |
| Fear | Raised brows, widened eyes, pulled-back lips | Low |
| Disgust | Nose wrinkle, raised upper lip | Very low |

The official test partition contains **3,068 images**, used for all model evaluation. Training data combines RAF-DB's training partition with FER2013 and a supplementary dataset. Disgust and fear are the most underrepresented classes, a fact directly reflected in their lower per-class metrics.

[**Recommended Figure 2 if available:** A grid of RAF-DB sample images showing one representative face per emotion class.  
*Figure 2: Example aligned face images from the seven RAF-DB emotion categories.*  
This figure concretely shows the visual similarity between some classes (fear vs. surprise) and the subtlety of others (disgust, neutral), motivating the per-class analysis in Section 6.]

### 2.2 Preprocessing Pipeline

Two separate preprocessing configurations are applied depending on the model, summarized in Table 3.

**Table 3 — Preprocessing Comparison: SE-CNN vs. EfficientNetB2**

| Parameter | SE-CNN | EfficientNetB2 |
|:----------|:-------|:---------------|
| Input size | 48×48 px | 96×96 px |
| Color mode | Grayscale (1 channel) | RGB (3 channels) |
| Standardization | Per-image (zero-mean, unit-variance) | Per-image (zero-mean, unit-variance) |
| Augmentation | Flip, rotate, zoom, translate | Flip, rotate, zoom, translate |
| Photometric augmentation | **Excluded** | **Excluded** |
| Oversampling | 12,000 samples per class | Loaded from raw RAF-DB files |
| Data caching | NPZ cache on first run | No cache |

**Grayscale for the SE-CNN** is chosen because emotion is encoded in facial geometry, not color, and single-channel input reduces dimensionality and overfitting risk. **RGB for EfficientNetB2** is required because its pretrained ImageNet weights encode color information from the first convolution.

**Per-image standardization** normalizes each image independently to zero mean and unit variance, making the model robust to global illumination shifts without relying on dataset-wide statistics.

### 2.3 Augmentation: A Critical Engineering Finding

Geometric augmentation (horizontal flip, ±15° rotation, ±10% zoom, ±10% translation) is applied stochastically during training. Photometric augmentations (brightness, contrast) were initially included but caused a complete training failure: accuracy remained at **14.3%** — the 7-class random baseline (1/7) — for the entire training run.

The mechanism is a pipeline conflict: per-image standardization, applied first, removes all global brightness and contrast information. Applying `RandomBrightness` or `RandomContrast` afterwards introduces shifts that the subsequent normalization immediately cancels, producing contradictory gradients. The model cannot learn a consistent mapping and collapses. Removing photometric augmentation entirely resolved the collapse. This finding is not documented in standard deep learning tutorials and constitutes a generalizable engineering lesson (see also Section 9).

### 2.4 Minority Class Oversampling

To correct class imbalance, minority classes are replicated with augmentation until each class has **12,000 training samples** (84,000 total). This ensures equal gradient contribution from all classes and is critical for preventing the classifier from ignoring disgust and fear. EfficientNetB2 loads raw RAF-DB files directly and applies class-aware sampling at the batch level.

\newpage

## 3. Proposed System Architecture

The system is designed as a complete computer vision pipeline, not just a model training experiment. Figure 1 shows the full data flow from raw input to application output.

```
Video / Image Input
        │
        ▼
Face Detection  (Haar Cascade  or  MTCNN)
        │
        ▼
Face Crop and Alignment
        │
        ▼
Preprocessing  (Resize · Grayscale or RGB · Per-image Standardize)
        │
        ▼
Model Inference  (EfficientNetB2 / SE-CNN / Ensemble)
        │
        ▼
Softmax Probability Vector  (7 classes)
        │
        ├──► Grad-CAM Heatmap  (facial region visualization)
        │
        ▼
Application Output
    ├── Streamlit Web App  (Live Demo · Video · Model Comparison · Training Curves)
    ├── Real-Time Webcam  (overlay · probability bar · Grad-CAM toggle)
    ├── CLI Tools  (detect_image.py · detect_video.py)
    ├── Meeting Analyzer  (PDF report with emotion timeline)
    └── TFLite Export  (INT8 quantized · Android · Raspberry Pi)
```

*Figure 1: Complete system pipeline from image/video input to face detection, preprocessing, model inference, Grad-CAM visualization, and multi-mode deployment output.*

**Face detection** uses two backends. Haar cascade (OpenCV) provides fast CPU inference suitable for real-time processing. MTCNN offers higher accuracy on non-frontal or partially occluded faces at the cost of additional compute. The face tracker applies temporal smoothing to prevent bounding box jitter across video frames.

**Inference pipeline:** the cropped face region is resized and standardized, passed through the selected model, and converted to a 7-dimensional softmax probability vector. For offline evaluation, 8-pass TTA averages predictions over random augmented views. For real-time inference, single-pass mode is used to maintain acceptable latency.

**Grad-CAM** is integrated directly in the pipeline: the gradient of the top predicted class score with respect to the final convolutional feature maps is computed, spatially pooled to importance weights, and overlaid as a heatmap on the original face image.

\newpage

## 4. Model Architectures

### 4.1 DeepFace — Zero-Shot Baseline

DeepFace is evaluated without any training on RAF-DB. Its pretrained emotion classifier is applied directly to the RAF-DB test set, yielding **52.80%** accuracy. This result quantifies the cross-domain transfer gap: even a state-of-the-art pretrained face analysis system achieves only moderate performance when applied out-of-the-box to a new dataset with different image statistics and label distributions. Every subsequent improvement is measured against this reference.

### 4.2 MobileNetV2 — Lightweight Transfer Learning

MobileNetV2 uses depthwise separable convolutions and inverted residual blocks to achieve efficient inference while maintaining strong representational capacity. Its ImageNet-pretrained backbone is adapted by freezing the backbone and training a new classification head (Global Average Pooling → Dense → Dropout → 7-class Softmax), then fine-tuning end-to-end at a reduced learning rate. SWA and 8-pass TTA are applied at the final stages.

**Achieved: 68.19%** — a +15.39% gain over DeepFace, demonstrating that task-specific fine-tuning with a pretrained backbone substantially outperforms zero-shot application.

### 4.3 Custom Multi-Scale SE-CNN

The SE-CNN is designed specifically for the spatial scale diversity inherent in facial expressions. Different emotion cues appear at different scales: subtle muscle tension (e.g., a nasal wrinkle for disgust) requires fine-grained receptive fields, while gross facial configuration (e.g., a wide smile) requires larger ones. No single convolution kernel size can optimally capture both simultaneously.

**Architecture:**

```
Input  48×48×1 (grayscale)
    │
    ├── Branch 1: Conv 3×3  (fine-grained texture and edges)
    ├── Branch 2: Conv 5×5  (facial part shapes)          ──► Concatenate
    └── Branch 3: Conv 7×7  (holistic facial configuration)
                                        │
                              Residual Blocks + BatchNorm
                                        │
                              SE Attention (Squeeze → Excite → Recalibrate)
                                        │
                              Global Average Pooling
                                        │
                              Dense(256) + Dropout
                                        │
                              Softmax  (7 classes)
```

*Figure 3: Multi-branch SE-CNN architecture. Three parallel convolution branches capture features at different spatial scales, concatenated and refined by residual blocks with SE channel attention.*

The **Squeeze-and-Excitation (SE) block** performs channel-wise attention: a global average pool (squeeze) reduces each feature map to a scalar; a two-layer MLP with sigmoid activation (excite) produces per-channel scaling weights; these weights recalibrate the feature map by element-wise multiplication. This allows the network to adaptively suppress identity or illumination channels and emphasize expression-discriminative ones.

Additional design choices: He initialization for stable gradient flow through ReLU activations; Batch Normalization on every convolutional layer; residual connections to enable deeper training without gradient vanishing; and Global Average Pooling instead of spatial flattening to reduce parameter count and overfitting risk. The network has approximately **3.2M parameters**, trained from scratch.

**Achieved:** SE-CNN v1 (best checkpoint + TTA): **69.42%**; SE-CNN v2 (fine-tuned from best checkpoint + SWA + TTA): **69.73%**.

### 4.4 EfficientNetB2 — Progressive Fine-Tuning

EfficientNetB2 applies compound scaling — jointly increasing depth, width, and input resolution — to maximize accuracy per FLOP. Pretrained on 14 million ImageNet images, it encodes rich visual primitives (edges, textures, spatial configurations) that transfer directly to facial expression analysis.

**Input:** 96×96 RGB, matching the pretrained architecture's expected resolution and color encoding.

**Progressive unfreezing** is used to avoid catastrophic forgetting — the overwriting of pretrained representations by large gradients from an untrained head. The protocol follows four phases:

**Table 4 — EfficientNetB2 Progressive Fine-Tuning Schedule**

| Phase | Layers Unfrozen | Epochs | Learning Rate | Val Accuracy |
|:-----:|:---------------|:------:|:-------------:|:------------:|
| 1 — Head only | 0% (frozen backbone) | 15 | 1×10⁻³ | 57.41% |
| 2 — Top 40% | 40% (deepest layers) | 20 | 2×10⁻⁴ | 72.72% |
| 3 — Full fine-tune | 100% | 10 | 5×10⁻⁵ | 74.59% |
| 4 — SWA refinement | 100% | 8 | 1×10⁻⁵ | 75.16% |
| **Final (8-pass TTA)** | — | — | — | **80.02%** |

Phase 1 stabilizes the classification head before any backbone modifications. Phase 2 unfreezes the deepest layers — those encoding the most task-specific, high-level features — and produces the largest single gain (+15.31%). Phase 3 extends adaptation to the full backbone at a very low learning rate. Phase 4 applies SWA to bias toward flat loss landscape regions with better generalization. The final +4.86% from TTA reflects variance reduction across 8 augmented inference passes.

Mixed-precision training (float16), uint8 in-memory image storage (~300 MB RAM), and gradient clipping (clipnorm=1.0) were applied throughout to make training feasible on an RTX 4050 Laptop GPU.

**Achieved: 80.02%** — the best result of the project.

### 4.5 Ensemble, SWA, and TTA

**Ensemble (soft voting):** four models are combined — MobileNetV2 SWA, SE-CNN v1 best, SE-CNN v2 best, and SE-CNN v2 SWA. Their 7-dimensional softmax probability vectors are averaged, and the class with highest mean probability is selected. Architectural diversity (MobileNetV2 vs. SE-CNN) and trajectory diversity (different checkpoints, SWA variants) produce partially uncorrelated errors, reducing variance. **Achieved: 70.45%.**

**SWA** averages model weights from multiple checkpoints in the flat region of the loss landscape rather than selecting a single minimum. Flat minima generalize better to unseen data; SWA consistently yields +0.5–1% improvement without additional data or architecture changes.

**TTA** runs 8 random augmentation passes at inference time and averages the softmax outputs. Each model gains approximately +1–2.5% accuracy through TTA. For real-time webcam inference, single-pass mode is used to maintain latency; TTA is reserved for offline evaluation.

\newpage

## 5. Training Protocol and Experimental Setup

**Table 5 — Experimental Configuration Summary**

| Parameter | SE-CNN | EfficientNetB2 |
|:----------|:-------|:---------------|
| Framework | TensorFlow 2.21 | TensorFlow 2.21 |
| Hardware | RTX 4050 Laptop GPU | RTX 4050 Laptop GPU |
| Loss function | Categorical cross-entropy | Categorical cross-entropy |
| Optimizer | Adam | Adam |
| LR schedule | Cosine decay (1×10⁻³ → ~0) | Manual per-phase (see Table 4) |
| Phase 1 epochs | 100 (cosine decay) | 15 |
| Phase 2 epochs | 30 (fine-tune from best ckpt) | 20 |
| Phase 3 epochs | 8 (SWA) | 10 |
| Phase 4 epochs | — | 8 (SWA) |
| Early stopping | Patience = 15 (val loss) | Patience = 10 (val loss) |
| Gradient clipping | — | clipnorm = 1.0 |
| Mixed precision | No | float16 |
| Batch size | 64 | 32 |
| Evaluation | Full RAF-DB test set (3,068) | Full RAF-DB test set (3,068) |
| Final inference | 8-pass TTA | 8-pass TTA |

All model checkpoints are saved on every validation accuracy improvement. Training metrics (loss, accuracy) are logged per epoch to CSV files, visualized in real time in the Streamlit application's Training Curves tab.

\newpage

## 6. Results and Analysis

### 6.1 Overall Model Comparison

[**Add Figure 4 here:** `evaluation/plots/model_comparison.png` — the generated bar chart showing all 6 model configurations.]

*Figure 4: Test accuracy comparison of all evaluated model configurations on the RAF-DB test set (3,068 images, 8-pass TTA). Grey bar: zero-shot baseline. Blue bars: trained custom models. Gold bar: best model.*

Figure 4 makes the experimental progression visual. Each configuration is listed in Table 6.

**Table 6 — Model Accuracy Comparison and Improvement over Baseline**

| Model | Test Accuracy | Gain vs. DeepFace | Configuration |
|:------|:-------------:|:-----------------:|:-------------|
| DeepFace pretrained (zero-shot) | 52.80% | — | Baseline (not trained on RAF-DB) |
| MobileNetV2 SWA + TTA | 68.19% | +15.39% | Transfer learning |
| Custom SE-CNN v1 (best + TTA) | 69.42% | +16.62% | Trained from scratch |
| Custom SE-CNN v2 (fine-tuned + TTA) | 69.73% | +16.93% | Fine-tuned from v1 |
| 4-Model Soft-Voting Ensemble + TTA | 70.45% | +17.65% | Ensemble |
| **EfficientNetB2 fine-tuned + TTA** | **80.02%** | **+27.22%** | **Best model** |

### 6.2 Ablation and Contribution Analysis

Table 7 attributes each accuracy gain to its principal cause. This allows us to evaluate whether each methodological choice was justified.

**Table 7 — Ablation: Contribution of Each Stage**

| Stage | Accuracy | Marginal Gain | Principal Factor |
|:------|:--------:|:-------------:|:----------------|
| DeepFace (zero-shot) | 52.80% | — | Pretrained features, no adaptation |
| + Task-specific fine-tuning (MobileNetV2) | 68.19% | +15.39% | ImageNet pretrained weights adapted to RAF-DB |
| + Custom architecture (SE-CNN) | 69.73% | +1.54% | Multi-scale branches + SE attention |
| + Ensemble (soft voting) | 70.45% | +0.72% | Partially independent errors across 4 models |
| + High-capacity backbone (EfficientNetB2) | 75.16% | +4.71% | Richer pretrained features from ImageNet |
| + 8-pass TTA | 80.02% | +4.86% | Variance reduction at inference |
| **Total improvement** | **+27.22%** | | |

The dominant contributions are task-specific fine-tuning (+15.39%) and the transition to EfficientNetB2 (+4.71% backbone gain, +4.86% TTA). The custom SE-CNN adds +1.54% over MobileNetV2, demonstrating that the purpose-built architecture provides genuine benefit at the cost of training from scratch. The ensemble contributes +0.72%, consistent with the moderate architectural diversity among its members.

The largest insight from this ablation is that **ImageNet pretraining is the single most important factor**: EfficientNetB2's backbone, adapted through progressive unfreezing, consistently outperformed the carefully designed 3.2M-parameter SE-CNN ensemble by a substantial margin. This confirms that representational richness from 14 million pretraining images cannot be replicated by architecture design alone at this dataset scale.

### 6.3 Per-Class Performance

**Table 8 — Per-Class Precision, Recall, and F1 Score (EfficientNetB2, 80.02%)**

| Emotion | Precision | Recall | F1 Score | Difficulty |
|:--------|:---------:|:------:|:--------:|:-----------|
| Happy | 0.91 | 0.92 | 0.91 | Easiest |
| Surprise | 0.80 | 0.76 | 0.78 | Moderate |
| Angry | 0.80 | 0.64 | 0.71 | Moderate |
| Neutral | 0.72 | 0.80 | 0.76 | Moderate |
| Sad | 0.74 | 0.81 | 0.77 | Moderate |
| Fear | 0.63 | 0.49 | 0.55 | Difficult |
| Disgust | 0.60 | 0.33 | 0.42 | Hardest |

Per-class metrics reveal disparities invisible in overall accuracy. **Happy** achieves F1 = 0.91 because the Duchenne smile — raised cheeks, visible teeth, periorbital crow's feet — produces a large, distinctive, and universally consistent facial deformation. The signal is high-amplitude and unambiguous.

**Fear** (F1 = 0.55, recall = 0.49) and **surprise** share the same primary muscle activations: frontalis raises both brows, levator palpebrae widens the eyes. The discriminating signal lies in subtle differences — the inner-brow oblique lift (corrugator contraction in fear) and dropped jaw vs. pulled-back lip corners — that are difficult to resolve from still images. Over 50% of fear instances are misclassified, likely as surprise.

**Disgust** (F1 = 0.42, recall = 0.33) is the most problematic class for three compounding reasons: (1) the primary cue — nasal wrinkle and raised upper lip — is a subtle texture-level deformation requiring fine-grained feature discrimination; (2) disgust is the rarest class in naturalistic photography, meaning fewer high-quality training examples even after oversampling; (3) annotation ambiguity is highest for low-intensity disgust, introducing label noise that directly reduces achievable recall.

**Angry** shows an asymmetric precision-recall profile (P=0.80, R=0.64): the model is conservative, predicting anger only when the signal is strong and misclassifying moderate anger as neutral or disgust. This is consistent with the visual overlap between stern-neutral and mild-anger facial configurations.

This analysis confirms that overall accuracy is an insufficient evaluation criterion for FER. A system achieving 80% overall accuracy still misclassifies the majority of fear and disgust expressions — a critical limitation for any deployment involving these emotions.

### 6.4 Confusion Patterns

[**Recommended Figure 5 if available:** The confusion matrix generated by `evaluation/evaluate.py`.  
*Figure 5: Confusion matrix of the EfficientNetB2 model on the RAF-DB test set (rows: true label, columns: predicted label).*  
The matrix would confirm the expected fear↔surprise and disgust↔angry/neutral confusion patterns discussed above. If the confusion matrix file is not available, this figure can be omitted without affecting the analysis above.]

Based on the per-class metrics, the principal confusion patterns are:
- **Fear → Surprise:** shared raised brows and widened eyes; mouth configuration is the key discriminator.
- **Disgust → Angry or Neutral:** overlapping brow-lowering muscle activation at low intensity.
- **Angry → Neutral:** low-intensity anger resembles a stern neutral face.
- **Sad → Neutral:** low-intensity sadness overlaps with neutral at the inner brow.

### 6.5 Failure Case Analysis

Table 9 summarizes the most likely failure patterns based on the per-class analysis. No fabricated image examples are cited; the table reflects systematic behavior inferred from the precision-recall profiles.

**Table 9 — Representative Failure Patterns**

| True Label | Predicted Label | Probable Cause |
|:-----------|:---------------|:---------------|
| Fear | Surprise | Shared brows/eyes; discriminating cue (inner brow oblique lift) not resolved |
| Disgust | Angry | Low-intensity disgust with brow lowering misread as anger |
| Disgust | Neutral | Very subtle nasal wrinkle below model's discrimination threshold |
| Angry | Neutral | Moderate anger (not extreme) resembles concentrated neutral |
| Sad | Neutral | Low-intensity sadness; inner brow raise insufficient |

[**Recommended Figure 6 if available:** Side-by-side examples of misclassified faces with Grad-CAM overlays showing where the model incorrectly focused.  
*Figure 6: Failure case examples with Grad-CAM overlays showing model attention on misclassified inputs.*  
This figure is optional and should only be included if actual misclassified examples with Grad-CAM can be extracted from the system. It should not exceed one figure of 4–6 examples to stay within the page budget.]

\newpage

## 7. Grad-CAM Interpretability

Gradient-weighted Class Activation Mapping (Grad-CAM) is integrated throughout the system to provide spatial explanations of model decisions. The gradient of the predicted class score with respect to the final convolutional feature maps is computed, globally average-pooled to per-channel importance weights, and linearly combined to produce a coarse spatial heatmap upsampled to the input resolution.

Interpretability serves two functions in this project: (1) as a debugging tool during development, confirming whether the model attends to facial regions rather than background; and (2) as a transparency feature in deployment, allowing end users to verify the model's reasoning.

[**Add Figure 7 here:** Grad-CAM screenshots from the Streamlit Live Demo tab or webcam output — use the Streamlit screenshots you provided showing happy and angry detections. Select the pair where the Grad-CAM heatmap is visible alongside the prediction.

*Figure 7: Grad-CAM visualizations from the deployed Streamlit application. Left: input face with detection bounding box and predicted emotion label. Centre: cropped face fed to the model. Right: probability bar chart showing class-wise confidence.*

**Placement:** This is the most valuable deployment screenshot. Use the **happy detection pair** (yellow bounding box + Streamlit probability bars) as the primary figure, as it shows the clearest 100% confidence prediction. The angry pair is a strong complement but may be included as a subfigure only if space permits.

**Mandatory or optional:** Mandatory — this is the primary visual evidence of deployment.]

Observed attention patterns are consistent with the expected facial anatomy of each emotion. For happy predictions, the heatmap concentrates on the lower face — the mouth, cheek, and periorbital region — corresponding to the Duchenne smile deformation. For angry predictions, attention falls on the brow and periorbital region, consistent with brow lowering and furrowing. These anatomically meaningful patterns confirm that EfficientNetB2 is solving the classification task through genuine expression features rather than background correlates.

For difficult classes (disgust, fear), heatmaps are more diffuse and less concentrated, reflecting the model's lower certainty and the fine-grained nature of the discriminative signal. In cases of correct disgust classification, attention occasionally concentrates around the nasal region — the expected locus of disgust cues. These observations qualitatively validate the model's representational strategy and provide actionable insight for further improvement (e.g., architectures that produce sharper attention on specific facial regions).

\newpage

## 8. Deployment and Real-Time Application

The trained models are integrated into a complete deployment stack that demonstrates the system is functional beyond training experiments.

**Streamlit Web Application** (`app/app.py`) provides five interactive tabs:

| Tab | Functionality |
|:----|:-------------|
| Live Demo | Image upload or webcam snapshot; per-face emotion label, confidence, probability bar chart, and Grad-CAM heatmap; optional side-by-side DeepFace comparison |
| Video Analysis | Frame-by-frame emotion processing; temporal timeline; stacked area chart; dominant-emotion strip |
| Model Comparison | Benchmark bar chart; sortable accuracy table for all six configurations |
| Training Curves | Accuracy and loss curves loaded live from CSV logs for all training phases |
| About | Architecture overview, dataset description, command reference |

[**Add Figure 8 here:** Use the two Streamlit screenshots you provided — the happy detection (yellow bounding box image + Streamlit probability bar showing 100%) and the angry detection (red bounding box + probability bar showing 99.9%).

*Figure 8: Deployed Streamlit application output. Top row: happy expression detected with 100% confidence (yellow bounding box) and probability distribution. Bottom row: angry expression detected with 99.9% confidence (red bounding box).*

**Placement:** Section 8, after the Streamlit tab table. Use both pairs side by side if layout permits; otherwise use the happy pair as the primary and note the angry result in the caption.

**Mandatory or optional:** Mandatory — direct evidence of system deployment and functionality.]

**Real-Time Webcam** (`realtime/realtime_webcam.py`) overlays emotion labels, probability bars, and an optional Grad-CAM heatmap on live webcam frames. The `--compare` flag displays predictions from both the custom model and DeepFace side-by-side. Face detection supports both Haar cascade (low latency) and MTCNN (higher accuracy).

**CLI Tools** provide batch processing capability: `detect_image.py` annotates single images; `detect_video.py` processes video files frame-by-frame; `meeting_analyzer.py` generates a PDF report with emotion timelines and statistical summaries from meeting recordings.

**TFLite Export** with INT8 post-training quantization produces a 4× smaller model (approximately 300 KB) deployable on Android smartphones and Raspberry Pi edge hardware. This extends the system to resource-constrained settings without modifying the training pipeline.

**Deployment tradeoff:** 8-pass TTA (used for benchmark evaluation) is not suitable for real-time inference because it multiplies latency by a factor of 8. The deployed webcam system uses single-pass inference, accepting a modest accuracy reduction in exchange for acceptable frame rates. A practical improvement would be multi-scale or sliding-window TTA with reduced pass count (e.g., 2–3 passes) to better balance accuracy and latency.

[**Add Figure 9 here:** `evaluation/plots/efficientnetb2_fine_curves.png` — the actual EfficientNetB2 training curves generated by the system.

*Figure 9: EfficientNetB2 training and validation accuracy/loss curves across the four fine-tuning phases, loaded from experiment CSV logs. Each phase transition is visible as a step change in the validation accuracy trajectory.*

**Placement:** Either end of Section 8 or as an appendix if space is constrained.

**Mandatory or optional:** Strongly recommended — the most direct evidence of the experimental process. Use `evaluation/plots/mobilenetv2_fine_curves.png` and `evaluation/plots/custom_cnn_curves.png` as optional supplementary figures only if space permits.]

The training curves confirm the expected behavior: Phase 2 (top-40% unfreezing) produces the steepest accuracy gain, Phase 3 continues improvement at a slower rate, and Phase 4 (SWA) stabilizes the model at its best generalization point. The convergence behavior validates the progressive unfreezing design.

\newpage

## 9. Problems Faced, Solutions, and Lessons Learned

**Table 10 — Engineering Challenges, Root Causes, Solutions, and Lessons**

| Problem | Root Cause | Solution | Lesson Learned |
|:--------|:----------|:---------|:---------------|
| Training collapse to 14.3% (random baseline) | `RandomBrightness`/`RandomContrast` applied after per-image standardization — normalization cancels the photometric shift, creating contradictory gradients | Remove all photometric augmentation; use geometric augmentation only | Preprocessing and augmentation steps are not independent — their interaction can destroy training. Always validate the full pipeline on a small run before committing to full training |
| Class imbalance (disgust, fear underrepresented) | Naturalistic photography skews heavily toward happy and neutral expressions | Oversample minority classes to 12,000 samples each | Oversampling is simple but effective; class-aware loss (focal loss, class weights) is a complementary alternative worth exploring |
| Catastrophic forgetting during fine-tuning | High learning rate fine-tuning overwrites pretrained representations with noisy gradients from an untrained head | Progressive unfreezing: freeze backbone, train head, then unfreeze incrementally at decreasing LR | The order of adaptation matters as much as the learning rate — always warm up the task-specific layers before touching pretrained weights |
| Fear/surprise visual confusion | Both emotions activate the same primary muscles (frontalis, levator palpebrae); discriminative cues are subtle and still-image-dependent | No architectural fix applied; addressed partially by TTA and ensemble diversity | Temporal modeling (video-based FER) or audio fusion may be necessary to reliably separate fear from surprise |
| Disgust consistently under-recalled (R=0.33) | Subtle texture-level cue, low training frequency even after oversampling, high annotation ambiguity | Accepted as a limitation; harder negative mining or synthetic augmentation are future options | Per-class metrics should drive dataset curation decisions — the lowest-recall class reveals where more data is most valuable |
| Real-time latency vs. accuracy tradeoff | 8-pass TTA is 8× more expensive than single-pass inference | Use single-pass for real-time; TTA reserved for offline evaluation | Benchmark accuracy and deployment accuracy should be reported separately when TTA is used |
| EfficientNetB2 outperforming custom SE-CNN by ~10% | The SE-CNN trains only on ~84,000 samples; ImageNet pretraining exposes EfficientNetB2 to 14M images | Adopted EfficientNetB2 as primary model | For FER at typical academic dataset scales, a well-chosen pretrained backbone will almost always outperform a custom architecture trained from scratch |

\newpage

## 10. Limitations and Future Work

**Current limitations:**

RAF-DB images are aligned and pre-cropped at 100×100 pixels. Real webcam footage presents faces at varying resolutions, angles, and occlusion levels. The system's benchmark accuracy (80.02%) likely overstates real-world performance, particularly for edge cases with non-frontal poses or partial occlusion. Demographic bias in web-collected datasets (RAF-DB) may affect performance across different age groups, ethnicities, and cultural expression norms.

The fear and disgust classes remain substantially under-performing (F1 = 0.55 and 0.42, respectively). Oversampling partially compensates for class imbalance but cannot address the fundamental scarcity of high-quality disgust and fear examples in the training data. Annotation ambiguity at low expression intensity is an irreducible source of label noise.

Real-time deployment uses single-pass inference, sacrificing approximately 4–5% accuracy relative to the benchmark TTA configuration.

**Future directions:**

- **Vision Transformers (ViT / Swin):** self-attention mechanisms can model long-range dependencies between distant facial regions (brow configuration and mouth posture), which may improve fear/surprise discrimination and other inter-class confusions.
- **Temporal modeling (LSTM / TCN):** processing sequences of video frames captures expression dynamics (onset, apex, offset) that are invisible in still images, likely improving fear recognition and reducing ambiguity.
- **Audio-visual fusion:** combining facial expression features with vocal prosody (pitch, energy, speech rate) provides complementary modalities that resolve cases where visual evidence is ambiguous.
- **Class-balanced loss functions:** focal loss or class-frequency-weighted cross-entropy as alternatives or complements to oversampling, with direct gradient-level control over minority class emphasis.
- **Larger and more diverse datasets:** AffectNet (1M+ images) or EmotioNet would provide more disgust and fear examples and greater demographic diversity, expected to narrow the inter-class performance gap.
- **Mobile-optimized inference:** knowledge distillation from EfficientNetB2 to a compact student network, or neural architecture search targeting mobile NPU inference constraints, for deployment without quantization quality loss.

\newpage

## 11. Conclusion

This project designed, implemented, evaluated, and deployed a complete facial emotion recognition system that progresses from a zero-shot DeepFace baseline at **52.80%** to a fine-tuned EfficientNetB2 model at **80.02%** — a +27.22% absolute improvement — through a principled sequence of methodological choices on the RAF-DB dataset.

The key technical findings are:

1. **Transfer learning from ImageNet is decisive.** EfficientNetB2 outperforms a 4-model ensemble of purpose-built SE-CNNs by +9.57%, confirming that 14 million pretraining images provide representational richness that architectural design alone cannot replicate at academic dataset scales.

2. **Preprocessing and augmentation interact non-trivially.** Combining per-image standardization with photometric augmentation causes complete training collapse. Geometric-only augmentation resolves the instability and is the correct choice for standardized FER pipelines.

3. **Overall accuracy is an insufficient evaluation metric.** EfficientNetB2's 80.02% overall accuracy coexists with F1 = 0.42 for disgust and F1 = 0.55 for fear — both below chance-level usefulness for clinical applications involving these emotions. Per-class analysis is essential.

4. **Grad-CAM confirms anatomically meaningful reasoning.** The model attends to the mouth and cheek region for happy, the brow region for angry and surprised, and produces more diffuse attention for ambiguous classes — validating that the learned representations are expression-grounded.

5. **A functional deployment is achievable on modest hardware.** The complete pipeline — Streamlit web app, real-time webcam, video analysis, meeting analytics, and TFLite edge export — runs on a laptop GPU and demonstrates that deep learning FER is ready for practical integration.

The project demonstrates a complete computer vision and deep learning engineering process: dataset curation, preprocessing design, architecture comparison, systematic experimentation, result analysis, failure diagnosis, and real-time deployment. It also surfaces concrete research directions — temporal modeling, audio-visual fusion, Vision Transformers — where the current system's limitations motivate further investigation.

\newpage

## 12. References

[1] M. Tan and Q. Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks," *Proc. ICML*, vol. 97, pp. 6105–6114, 2019.

[2] J. Hu, L. Shen, and G. Sun, "Squeeze-and-Excitation Networks," *Proc. CVPR*, pp. 7132–7141, 2018.

[3] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization," *Proc. ICCV*, pp. 618–626, 2017.

[4] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, "MobileNetV2: Inverted Residuals and Linear Bottlenecks," *Proc. CVPR*, pp. 4510–4520, 2018.

[5] S. Li, W. Deng, and J. Du, "Reliable Crowdsourcing and Deep Locality-Preserving Learning for Expression Recognition in the Wild," *Proc. CVPR*, pp. 2852–2861, 2017. [RAF-DB]

[6] S. Serengil and A. Ozpinar, "LightFace: A Hybrid Deep Face Recognition Framework," *ASYU*, 2020. [DeepFace]

[7] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," *Proc. CVPR*, pp. 770–778, 2016.

[8] P. Izmailov, D. Podoprikhin, T. Garipov, D. Vetrov, and A. G. Wilson, "Averaging Weights Leads to Wider Optima and Better Generalization," *Proc. UAI*, 2018. [SWA]

[9] D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization," *ICLR*, 2015.

[10] S. Ioffe and C. Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift," *Proc. ICML*, pp. 448–456, 2015.

[11] I. J. Goodfellow et al., "Challenges in Representation Learning: FER2013 Dataset," *ICONIP*, pp. 117–124, 2013.

[12] P. Viola and M. Jones, "Rapid Object Detection Using a Boosted Cascade of Simple Features," *Proc. CVPR*, pp. 511–518, 2001.

[13] O. Russakovsky et al., "ImageNet Large Scale Visual Recognition Challenge," *IJCV*, vol. 115, no. 3, pp. 211–252, 2015.

---

*Report generated from experimental results produced on an RTX 4050 Laptop GPU, TensorFlow 2.21, Python 3.10.*
