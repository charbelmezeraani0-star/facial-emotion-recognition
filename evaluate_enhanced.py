"""Enhanced evaluation: TTA + best checkpoint + weighted ensemble."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import tensorflow as tf
import keras
keras.config.enable_unsafe_deserialization()

from data.dataloader import auto_load, tta_predict
from models.ensemble import evaluate_ensemble

BATCH_SIZE = 64

print("Loading data...")
_, val_ds, test_ds, _, (X_test, y_test) = auto_load(
    rafdb_dir='data/raf', extra_dir='data/extra',
    affectnet_dir=None, batch_size=BATCH_SIZE, oversample_target=12000)

# ── Load models ──────────────────────────────────────────────────────────────
print("\nLoading models...")
mob_swa  = tf.keras.models.load_model('saved_models/mobilenetv2_swa.keras',  compile=False)
cnn_swa  = tf.keras.models.load_model('saved_models/custom_cnn_swa.keras',   compile=False)
cnn_best = tf.keras.models.load_model('saved_models/custom_cnn_best.keras',  compile=False)

for m in [mob_swa, cnn_swa, cnn_best]:
    m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ── 1. Baseline (already known) ───────────────────────────────────────────────
print("\n" + "="*60)
print("BASELINE (no TTA)")
print("="*60)
for name, model in [('MobileNetV2 SWA', mob_swa),
                    ('Custom CNN SWA',  cnn_swa),
                    ('Custom CNN Best', cnn_best)]:
    _, acc = model.evaluate(test_ds, verbose=0)
    print(f"  {name:20s}  test_acc={acc:.4f}")

# ── 2. TTA predictions ────────────────────────────────────────────────────────
print("\n" + "="*60)
print("WITH TEST-TIME AUGMENTATION (TTA, 8 passes)")
print("="*60)
print("Running TTA for MobileNetV2...")
mob_tta  = tta_predict(mob_swa,  X_test, n_augments=8, batch_size=BATCH_SIZE)
print("Running TTA for Custom CNN SWA...")
cnn_tta  = tta_predict(cnn_swa,  X_test, n_augments=8, batch_size=BATCH_SIZE)
print("Running TTA for Custom CNN Best...")
cnnb_tta = tta_predict(cnn_best, X_test, n_augments=8, batch_size=BATCH_SIZE)

y_true = np.argmax(y_test, axis=1)

for name, preds in [('MobileNetV2 SWA TTA',  mob_tta),
                    ('Custom CNN SWA TTA',    cnn_tta),
                    ('Custom CNN Best TTA',   cnnb_tta)]:
    acc = np.mean(np.argmax(preds, axis=1) == y_true)
    print(f"  {name:25s}  test_acc={acc:.4f}")

# ── 3. Ensemble variants ──────────────────────────────────────────────────────
print("\n" + "="*60)
print("ENSEMBLE VARIANTS")
print("="*60)

def ensemble_acc(pred_list, weights=None, label=""):
    if weights is None:
        weights = [1.0] * len(pred_list)
    w = np.array(weights) / sum(weights)
    avg = sum(p * wi for p, wi in zip(pred_list, w))
    acc = np.mean(np.argmax(avg, axis=1) == y_true)
    print(f"  {label:40s}  test_acc={acc:.4f}")
    return avg

# Equal weight ensembles
ensemble_acc([mob_tta, cnn_tta],          label="MobV2+CNN_SWA (equal, TTA)")
ensemble_acc([mob_tta, cnnb_tta],         label="MobV2+CNN_Best (equal, TTA)")
ensemble_acc([mob_tta, cnn_tta, cnnb_tta],label="MobV2+SWA+Best (equal, TTA)")

# Weighted: more weight to stronger model
ensemble_acc([mob_tta, cnn_tta],  weights=[0.35, 0.65], label="MobV2+CNN_SWA (35/65, TTA)")
ensemble_acc([mob_tta, cnnb_tta], weights=[0.35, 0.65], label="MobV2+CNN_Best (35/65, TTA)")

# Best overall — detailed per-class breakdown
print("\n" + "="*60)
print("BEST ENSEMBLE — Per-Class Breakdown")
print("="*60)
evaluate_ensemble([mob_swa, cnn_best], X_test, y_test,
                  precomputed_preds=[mob_tta, cnnb_tta], weights=[0.35, 0.65])
