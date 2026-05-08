"""
Best-in-class preprocessing pipeline for facial emotion recognition.

Pipeline per image:
  1. Upsample  → 96×96 (LANCZOS4)  — bigger canvas = better CLAHE & denoise
  2. Adaptive gamma correction       — normalises dark / overexposed faces
  3. CLAHE                           — local contrast enhancement
  4. Non-local means denoising       — removes noise, keeps edges
  5. Unsharp masking                 — sharpens eyes / mouth / brow edges
  6. Downsample → 48×48 (INTER_AREA) — clean anti-aliased result
  7. Normalise [0, 1]
  8. Per-image standardisation       — zero-mean unit-variance per face
                                       (better than dataset-wide for lighting diversity)
  9. Clip + rescale → [0, 1]

For RAF-DB / colour images we also run:
  10. Face alignment via Haar eye detector (rotation correction)
"""

import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

IMG_SIZE  = 48
UPSAMPLE  = 96          # intermediate resolution for processing

# Dataset-wide statistics (fallback when per-image std is near zero)
GLOBAL_MEAN = 0.5077
GLOBAL_STD  = 0.2550


# ────────────────────────────────────────────────────────────────────────────
# Individual enhancement functions
# ────────────────────────────────────────────────────────────────────────────

def adaptive_gamma(img_uint8: np.ndarray) -> np.ndarray:
    """
    Computes gamma so that the output mean brightness ≈ 0.5.
    Formula: gamma = log(0.5) / log(mean_brightness)
    Clamped to [0.4, 2.5] to avoid extreme corrections on near-black images.
    """
    mean_val = img_uint8.mean() / 255.0
    if mean_val < 0.01 or mean_val > 0.99:
        return img_uint8
    gamma     = np.log(0.5) / np.log(mean_val + 1e-7)
    gamma     = float(np.clip(gamma, 0.4, 2.5))
    inv_gamma = 1.0 / gamma
    table     = (np.arange(256) / 255.0) ** inv_gamma * 255.0
    table     = np.clip(table, 0, 255).astype(np.uint8)
    return cv2.LUT(img_uint8, table)


def apply_clahe(img_uint8: np.ndarray,
                clip_limit: float = 2.5,
                tile_size: int = 6) -> np.ndarray:
    """
    CLAHE on the upsampled (96×96) image.
    Larger tile_size=6 is better matched to 96×96 than the old (4,4) on 48×48.
    clip_limit=2.5 gives stronger but still artefact-free enhancement.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                             tileGridSize=(tile_size, tile_size))
    return clahe.apply(img_uint8)


def nlmeans_denoise(img_uint8: np.ndarray, h: float = 4) -> np.ndarray:
    """
    Non-local means denoising.
    Superior to bilateral filter: uses non-adjacent patches → preserves
    facial texture (pores, fine wrinkles) while removing compression noise.
    h=4 is conservative — removes noise without blurring emotion features.
    """
    return cv2.fastNlMeansDenoising(
        img_uint8, h=h,
        templateWindowSize=7,
        searchWindowSize=15)


def unsharp_mask(img_uint8: np.ndarray,
                 sigma: float = 1.2,
                 strength: float = 0.9) -> np.ndarray:
    """
    Unsharp masking: sharpened = original + strength × (original − blurred)
    Enhances discriminative edges: eye corners, lip line, brow ridge.
    strength=0.9 is strong enough to be useful without creating haloing.
    """
    blurred   = cv2.GaussianBlur(img_uint8.astype(np.float32), (0, 0), sigma)
    sharpened = img_uint8.astype(np.float32) + strength * (img_uint8.astype(np.float32) - blurred)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def align_face_rotation(gray_uint8: np.ndarray) -> np.ndarray:
    """
    Detects eyes with Haar cascade and rotates the image to make them horizontal.
    Only for larger input images (≥ 80px); skips if fewer than 2 eyes found.
    """
    if gray_uint8.shape[0] < 80:
        return gray_uint8
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(gray_uint8, 1.05, 3, minSize=(10, 10))
    if len(eyes) < 2:
        return gray_uint8
    eyes  = sorted(eyes, key=lambda e: e[0])[:2]
    cx    = [e[0] + e[2] // 2 for e in eyes]
    cy    = [e[1] + e[3] // 2 for e in eyes]
    angle = np.degrees(np.arctan2(cy[1] - cy[0], cx[1] - cx[0]))
    if abs(angle) < 2.0:        # already nearly horizontal
        return gray_uint8
    h, w = gray_uint8.shape
    M    = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(gray_uint8, M, (w, h),
                          flags=cv2.INTER_LANCZOS4,
                          borderMode=cv2.BORDER_REFLECT)


def per_image_standardise(img_float: np.ndarray) -> np.ndarray:
    """
    Zero-mean, unit-variance normalisation per individual image.
    Better than dataset-wide stats because faces have wildly different
    lighting conditions — per-image normalisation removes that variation.
    Falls back to global stats if the image has near-zero variance (rare).
    """
    mean = img_float.mean()
    std  = img_float.std()
    if std < 1e-6:
        # Degenerate image (solid colour) — use global stats
        std  = GLOBAL_STD
        mean = GLOBAL_MEAN
    img = (img_float - mean) / std
    img = np.clip(img, -3.0, 3.0)
    img = (img + 3.0) / 6.0          # rescale to [0, 1]
    return img


# ────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ────────────────────────────────────────────────────────────────────────────

def preprocess_image(img_uint8: np.ndarray,
                     align: bool = False) -> np.ndarray:
    """
    Full preprocessing pipeline.

    Args:
        img_uint8 : grayscale uint8 array, any size (will be handled)
        align     : whether to attempt eye-based rotation correction
                    (use True for RAF-DB / real photos; False for FER2013)

    Returns:
        float32 array of shape (48, 48, 1), values in [0, 1]
    """
    img = img_uint8.reshape(img_uint8.shape[0], img_uint8.shape[1] if img_uint8.ndim > 1 else IMG_SIZE)

    # ── Optional: face rotation alignment ──────────────────────────────────
    if align:
        img = align_face_rotation(img)

    # ── Step 1: upsample to 96×96 ─────────────────────────────────────────
    img = cv2.resize(img, (UPSAMPLE, UPSAMPLE), interpolation=cv2.INTER_LANCZOS4)

    # ── Step 2: adaptive gamma correction ────────────────────────────────
    img = adaptive_gamma(img)

    # ── Step 3: CLAHE ────────────────────────────────────────────────────
    img = apply_clahe(img, clip_limit=2.5, tile_size=6)

    # ── Step 4: non-local means denoising ────────────────────────────────
    img = nlmeans_denoise(img, h=4)

    # ── Step 5: unsharp masking ──────────────────────────────────────────
    img = unsharp_mask(img, sigma=1.2, strength=0.9)

    # ── Step 6: downsample back to 48×48 ─────────────────────────────────
    # INTER_AREA is the correct interpolation for downsampling — no aliasing
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    # ── Step 7: float conversion ─────────────────────────────────────────
    img_f = img.astype(np.float32) / 255.0

    # ── Step 8: per-image standardisation ────────────────────────────────
    img_f = per_image_standardise(img_f)

    return img_f.reshape(IMG_SIZE, IMG_SIZE, 1)


def preprocess_from_color(img_bgr: np.ndarray, align: bool = True) -> np.ndarray:
    """
    For colour images (RAF-DB, AffectNet, webcam frames):
    converts to grayscale first, then runs the full pipeline with alignment.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMG_SIZE * 2, IMG_SIZE * 2))  # ensure enough res for alignment
    return preprocess_image(gray, align=align)


# ────────────────────────────────────────────────────────────────────────────
# Batched preprocessing (multi-threaded for speed)
# ────────────────────────────────────────────────────────────────────────────

def preprocess_batch(X_uint8: np.ndarray,
                     align: bool = False,
                     n_workers: int = 8) -> np.ndarray:
    """
    Preprocess a batch of raw uint8 images in parallel using threads.
    n_workers=8 is efficient on most CPUs without I/O contention.
    """
    flat = X_uint8.reshape(len(X_uint8), IMG_SIZE, IMG_SIZE)

    def _process(img):
        return preprocess_image(img, align=align)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        results = list(tqdm(
            pool.map(_process, flat),
            total=len(flat),
            desc='  Preprocessing',
            unit='img',
            ncols=80))

    return np.array(results, dtype=np.float32)


def preprocess_image_from_path(path: str, align: bool = True) -> np.ndarray:
    """Load an image file and preprocess it — used by RAF-DB and AffectNet loaders."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (IMG_SIZE * 2, IMG_SIZE * 2))
    return preprocess_image(img, align=align)
