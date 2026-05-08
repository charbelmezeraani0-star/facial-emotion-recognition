import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from data.preprocessing import (preprocess_image, preprocess_batch,
                                  preprocess_image_from_path)

EMOTIONS    = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
IMG_SIZE    = 48
NUM_CLASSES = 7

# RAF-DB folder number → FER2013 label mapping
RAFDB_TO_FER = {
    1: 5,   # surprise → surprise
    2: 2,   # fear     → fear
    3: 1,   # disgust  → disgust
    4: 3,   # happy    → happy
    5: 4,   # sad      → sad
    6: 0,   # angry    → angry
    7: 6,   # neutral  → neutral
}

# AffectNet label → FER2013 label mapping (contempt=7 skipped)
AFFECTNET_TO_FER = {
    0: 6, 1: 3, 2: 4, 3: 5, 4: 2, 5: 1, 6: 0,
}


# ---------------------------------------------------------------------------
# FER2013 loader
# ---------------------------------------------------------------------------

def load_fer2013_csv(csv_path):
    print("Loading FER2013 from CSV...")
    df     = pd.read_csv(csv_path)
    pixels = df['pixels'].apply(lambda x: np.array(x.split(), dtype=np.uint8))
    X_raw  = np.stack(pixels)
    print("  Preprocessing FER2013...")
    X = preprocess_batch(X_raw)
    y = df['emotion'].values

    if 'Usage' in df.columns:
        train_mask = df['Usage'] == 'Training'
        val_mask   = df['Usage'] == 'PublicTest'
        test_mask  = df['Usage'] == 'PrivateTest'
        return (X[train_mask], y[train_mask],
                X[val_mask],   y[val_mask],
                X[test_mask],  y[test_mask])

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_v, X_te, y_v, y_te     = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)
    return X_tr, y_tr, X_v, y_v, X_te, y_te


def load_fer2013_folders(root_dir):
    def read_split(split):
        X, y = [], []
        for label, emotion in enumerate(EMOTIONS):
            folder = os.path.join(root_dir, split, emotion)
            if not os.path.exists(folder):
                continue
            for fname in os.listdir(folder):
                img = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                X.append(preprocess_image(img))
                y.append(label)
        return np.array(X, dtype=np.float32), np.array(y)

    print("Loading FER2013 from folders...")
    X_tr, y_tr = read_split('train')
    X_te, y_te = read_split('test')
    X_tr, X_v, y_tr, y_v = train_test_split(X_tr, y_tr, test_size=0.1,
                                              random_state=42, stratify=y_tr)
    return X_tr, y_tr, X_v, y_v, X_te, y_te


# ---------------------------------------------------------------------------
# RAF-DB loader
# ---------------------------------------------------------------------------

def load_rafdb(rafdb_dir):
    """
    Loads RAF-DB dataset.
    Expected structure:
        rafdb_dir/
            DATASET/
                train/1/ train/2/ ... train/7/
                test/1/  test/2/  ... test/7/
    Labels are folder numbers 1-7, remapped to FER2013 0-6.
    """
    dataset_dir = os.path.join(rafdb_dir, 'DATASET')
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Cannot find DATASET folder in {rafdb_dir}")

    def read_split(split):
        X, y = [], []
        split_dir = os.path.join(dataset_dir, split)
        for folder_num, fer_label in RAFDB_TO_FER.items():
            folder = os.path.join(split_dir, str(folder_num))
            if not os.path.exists(folder):
                continue
            for fname in os.listdir(folder):
                img = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                # RAF-DB images are real photos — use alignment
                processed = preprocess_image_from_path(
                    os.path.join(folder, fname), align=True)
                if processed is None:
                    continue
                X.append(processed)
                y.append(fer_label)
        return np.array(X, dtype=np.float32), np.array(y)

    print("Loading RAF-DB...")
    X_tr, y_tr = read_split('train')
    X_te, y_te = read_split('test')
    print(f"  RAF-DB Train: {len(X_tr):,} | Test: {len(X_te):,}")
    return X_tr, y_tr, X_te, y_te


# ---------------------------------------------------------------------------
# AffectNet loader
# ---------------------------------------------------------------------------

def load_affectnet(affectnet_dir, max_per_class=5000):
    """
    Loads AffectNet dataset.
    Expected structure:
        affectnet_dir/
            training.csv          (or Manually_Annotated_file_lists/training.csv)
            Manually_Annotated_Images/
                <subpath>/<image>.jpg

    max_per_class: cap per emotion to avoid extreme imbalance vs FER2013.
    """
    # Try to find the annotation CSV
    candidates = [
        os.path.join(affectnet_dir, 'training.csv'),
        os.path.join(affectnet_dir, 'Manually_Annotated_file_lists', 'training.csv'),
    ]
    csv_path = next((p for p in candidates if os.path.exists(p)), None)
    if csv_path is None:
        raise FileNotFoundError(
            f"Cannot find training.csv in {affectnet_dir}. "
            "Expected at 'training.csv' or 'Manually_Annotated_file_lists/training.csv'")

    img_root = os.path.join(affectnet_dir, 'Manually_Annotated_Images')
    df = pd.read_csv(csv_path)

    # Column names vary by version — normalise them
    df.columns = [c.strip().lower() for c in df.columns]
    path_col  = next(c for c in df.columns if 'path' in c or 'subdir' in c)
    label_col = next(c for c in df.columns if 'expression' in c or 'label' in c)

    print(f"Loading AffectNet ({len(df):,} annotations)...")
    X, y = [], []
    counts = {k: 0 for k in AFFECTNET_TO_FER}

    for _, row in df.iterrows():
        af_label = int(row[label_col])
        if af_label not in AFFECTNET_TO_FER:
            continue                                    # skip contempt
        fer_label = AFFECTNET_TO_FER[af_label]
        if counts[af_label] >= max_per_class:
            continue

        img_path = os.path.join(img_root, str(row[path_col]).strip())
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        X.append(preprocess_image(img))
        y.append(fer_label)
        counts[af_label] += 1

    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    print(f"  Loaded {len(X):,} AffectNet images")
    print(f"  Per-class: { {EMOTIONS[AFFECTNET_TO_FER[k]]: counts[k] for k in AFFECTNET_TO_FER} }")
    return X, y


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Extra dataset loader (anger/disgust/fear/happiness/neutrality/sadness/surprise)
# ---------------------------------------------------------------------------

# Folder name → FER2013 label index
EXTRA_TO_FER = {
    'anger':     0,   # angry
    'disgust':   1,
    'fear':      2,
    'happiness': 3,   # happy
    'sadness':   4,   # sad
    'surprise':  5,
    'neutrality':6,   # neutral
    # 'contempt' is skipped — not in FER2013
}


def load_extra(extra_dir):
    """Loads the supplementary emotion dataset with non-standard folder names."""
    if not os.path.exists(extra_dir):
        return None, None

    print("Loading extra dataset...")
    X, y = [], []
    for folder_name, fer_label in EXTRA_TO_FER.items():
        folder = os.path.join(extra_dir, folder_name)
        if not os.path.exists(folder):
            continue
        files = os.listdir(folder)
        for fname in files:
            result = preprocess_image_from_path(
                os.path.join(folder, fname), align=True)
            if result is None:
                continue
            X.append(result)
            y.append(fer_label)
        print(f"  {folder_name:12s}: {len(files):4d} images")

    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    print(f"  Extra total: {len(X):,}")
    return X, y


# ---------------------------------------------------------------------------
# Oversampling — balances minority classes by duplicating with augmentation
# ---------------------------------------------------------------------------

def oversample_minority(X, y, target_per_class=3000):
    """
    For any class with fewer than target_per_class samples,
    randomly duplicate existing samples until the target is reached.
    This directly fixes the disgust/fear underrepresentation problem.
    """
    X_out, y_out = list(X), list(y)
    for cls in range(NUM_CLASSES):
        idx   = np.where(y == cls)[0]
        count = len(idx)
        if count >= target_per_class:
            continue
        needed = target_per_class - count
        extra  = np.random.choice(idx, size=needed, replace=True)
        X_out.extend(X[extra])
        y_out.extend([cls] * needed)
        print(f"  Oversampled {EMOTIONS[cls]:10s}: {count} → {count + needed}")

    perm = np.random.permutation(len(X_out))
    return np.array(X_out)[perm], np.array(y_out)[perm]


def compute_class_weights(y_labels):
    from sklearn.utils.class_weight import compute_class_weight
    weights = compute_class_weight('balanced', classes=np.arange(NUM_CLASSES), y=y_labels)
    cw = dict(enumerate(weights))
    print("Class weights:", {EMOTIONS[k]: round(v, 2) for k, v in cw.items()})
    return cw


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def build_augmentation_layer():
    # Only geometric augmentations — per-image standardisation in preprocessing
    # makes brightness/contrast augmentation counterproductive (they undo the
    # standardisation and produce contradictory training signals).
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.10),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomTranslation(0.08, 0.08),
    ], name='augmentation')


def make_dataset(X, y, batch_size=64, augment=False, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(len(X), seed=42)
    if augment:
        aug = build_augmentation_layer()
        ds  = ds.map(lambda x, lbl: (aug(x, training=True), lbl),
                     num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    return ds.prefetch(tf.data.AUTOTUNE)


def mixup_dataset(ds, alpha=0.2):
    """
    Mixup augmentation: blends pairs of images and their labels.
    x_mix = λ·x1 + (1-λ)·x2,  y_mix = λ·y1 + (1-λ)·y2
    Lambda is sampled from Beta(alpha, alpha).
    Proven to improve accuracy by ~1-2% and reduce overfitting on noisy labels.
    """
    def mixup_batch(images, labels):
        batch_size = tf.shape(images)[0]
        lam = tf.cast(
            tf.random.stateless_binomial(
                shape=[batch_size], seed=[42, 0],
                counts=1, probs=alpha / (alpha + 1.0)),
            tf.float32)
        # Simple beta approximation via uniform sampling
        lam = tf.random.uniform([batch_size, 1, 1, 1], minval=0.3, maxval=0.7)
        idx = tf.random.shuffle(tf.range(batch_size))
        images2 = tf.gather(images, idx)
        labels2 = tf.gather(labels, idx)
        lam_img = tf.cast(lam, images.dtype)
        mixed_images = lam_img * images + (tf.cast(1.0, images.dtype) - lam_img) * images2
        lam_label    = tf.reshape(lam, [batch_size, 1])
        labels_f  = tf.cast(labels,  tf.float32)
        labels2_f = tf.cast(labels2, tf.float32)
        mixed_labels = lam_label * labels_f + (1.0 - lam_label) * labels2_f
        return mixed_images, mixed_labels

    return ds.map(mixup_batch, num_parallel_calls=tf.data.AUTOTUNE)


def tta_predict(model, X, n_augments=5, batch_size=64):
    """
    Test Time Augmentation: runs N augmented versions of each image
    and averages predictions — improves accuracy at inference time.
    """
    aug = build_augmentation_layer()
    all_preds = []
    for _ in range(n_augments):
        ds   = tf.data.Dataset.from_tensor_slices(X).batch(batch_size)
        ds   = ds.map(lambda x: aug(x, training=True), num_parallel_calls=tf.data.AUTOTUNE)
        pred = model.predict(ds, verbose=0)
        all_preds.append(pred)
    # Also include original (no augmentation)
    orig = model.predict(
        tf.data.Dataset.from_tensor_slices(X).batch(batch_size), verbose=0)
    all_preds.append(orig)
    return np.mean(all_preds, axis=0)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _cache_path(cache_dir='data/processed'):
    return os.path.join(cache_dir, 'dataset_cache.npz')


def _load_cache(cache_dir='data/processed'):
    p = _cache_path(cache_dir)
    if not os.path.exists(p):
        return None
    print(f"Loading preprocessed cache from {p} ...")
    d = np.load(p)
    return (d['X_tr'], d['y_tr'], d['X_v'], d['y_v'], d['X_te'], d['y_te'])


def _save_cache(X_tr, y_tr, X_v, y_v, X_te, y_te, cache_dir='data/processed'):
    os.makedirs(cache_dir, exist_ok=True)
    p = _cache_path(cache_dir)
    np.savez_compressed(p, X_tr=X_tr, y_tr=y_tr,
                        X_v=X_v, y_v=y_v, X_te=X_te, y_te=y_te)
    print(f"Saved preprocessed cache → {p}")


def auto_load(data_dir='data/raw', rafdb_dir=None, affectnet_dir=None,
              extra_dir=None, batch_size=64, affectnet_max_per_class=5000,
              oversample_target=4000, cache_dir='data/processed',
              force_reload=False):
    """
    data_dir         : FER2013 (CSV or train/test folders)
    rafdb_dir        : RAF-DB root folder
    affectnet_dir    : AffectNet root folder
    extra_dir        : supplementary dataset (anger/disgust/fear/... folders)
    oversample_target: duplicate minority classes up to this count per class
    cache_dir        : directory for disk-cached preprocessed arrays (.npz)
    force_reload     : if True, ignore cache and reprocess from scratch
    """
    # ── Disk cache: skip slow preprocessing on subsequent runs ───────────────
    cached = None if force_reload else _load_cache(cache_dir)
    if cached is not None:
        X_tr, y_tr, X_v, y_v, X_te, y_te = cached
        print(f"Cache hit — Train: {len(X_tr):,} | Val: {len(X_v):,} | Test: {len(X_te):,}")
    else:
        # --- Load FER2013 ---
        csv_path = os.path.join(data_dir, 'fer2013.csv')
        if os.path.exists(csv_path):
            X_tr, y_tr, X_v, y_v, X_te, y_te = load_fer2013_csv(csv_path)
        else:
            X_tr, y_tr, X_v, y_v, X_te, y_te = load_fer2013_folders(data_dir)

        # --- Merge RAF-DB ---
        if rafdb_dir and os.path.exists(rafdb_dir):
            X_r_tr, y_r_tr, X_r_te, y_r_te = load_rafdb(rafdb_dir)
            X_r_v, _, y_r_v, _ = train_test_split(
                X_r_te, y_r_te, test_size=0.5, random_state=42, stratify=y_r_te)
            X_tr = np.concatenate([X_tr, X_r_tr]);  y_tr = np.concatenate([y_tr, y_r_tr])
            X_v  = np.concatenate([X_v,  X_r_v]);   y_v  = np.concatenate([y_v,  y_r_v])
            print(f"After RAF-DB   — Train: {len(X_tr):,} | Val: {len(X_v):,}")
        elif rafdb_dir:
            print(f"RAF-DB not found: {rafdb_dir}")

        # --- Merge extra dataset ---
        if extra_dir and os.path.exists(extra_dir):
            X_ex, y_ex = load_extra(extra_dir)
            if X_ex is not None and len(X_ex) > 0:
                X_ex_tr, X_ex_v, y_ex_tr, y_ex_v = train_test_split(
                    X_ex, y_ex, test_size=0.1, random_state=42, stratify=y_ex)
                X_tr = np.concatenate([X_tr, X_ex_tr]); y_tr = np.concatenate([y_tr, y_ex_tr])
                X_v  = np.concatenate([X_v,  X_ex_v]);  y_v  = np.concatenate([y_v,  y_ex_v])
                print(f"After extra    — Train: {len(X_tr):,} | Val: {len(X_v):,}")
        elif extra_dir:
            print(f"Extra dir not found: {extra_dir}")

        # --- Merge AffectNet ---
        if affectnet_dir and os.path.exists(affectnet_dir):
            X_an, y_an = load_affectnet(affectnet_dir, max_per_class=affectnet_max_per_class)
            X_an_tr, X_an_v, y_an_tr, y_an_v = train_test_split(
                X_an, y_an, test_size=0.1, random_state=42, stratify=y_an)
            X_tr = np.concatenate([X_tr, X_an_tr]); y_tr = np.concatenate([y_tr, y_an_tr])
            X_v  = np.concatenate([X_v,  X_an_v]);  y_v  = np.concatenate([y_v,  y_an_v])
            print(f"After AffectNet — Train: {len(X_tr):,} | Val: {len(X_v):,}")

        # Save to disk for fast reuse
        _save_cache(X_tr, y_tr, X_v, y_v, X_te, y_te, cache_dir)

    # --- Oversample minority classes ---
    print("Oversampling minority classes...")
    X_tr, y_tr = oversample_minority(X_tr, y_tr, target_per_class=oversample_target)

    # --- Final shuffle ---
    perm = np.random.permutation(len(X_tr))
    X_tr, y_tr = X_tr[perm], y_tr[perm]

    # --- Print final distribution ---
    print("\nFinal training distribution:")
    for i, e in enumerate(EMOTIONS):
        n   = int((y_tr == i).sum())
        bar = '█' * (n // 300)
        print(f"  {e:10s}: {n:5d}  {bar}")

    class_weights = compute_class_weights(y_tr)

    y_tr_cat = to_categorical(y_tr, NUM_CLASSES).astype(np.float32)
    y_v_cat  = to_categorical(y_v,  NUM_CLASSES).astype(np.float32)
    y_te_cat = to_categorical(y_te, NUM_CLASSES).astype(np.float32)

    train_ds = make_dataset(X_tr, y_tr_cat, batch_size, augment=True, shuffle=True)
    val_ds   = make_dataset(X_v,  y_v_cat,  batch_size)
    test_ds  = make_dataset(X_te, y_te_cat, batch_size)

    print(f"\nTrain: {len(X_tr):,} | Val: {len(X_v):,} | Test: {len(X_te):,}")
    return train_ds, val_ds, test_ds, class_weights, (X_te, y_te_cat)
