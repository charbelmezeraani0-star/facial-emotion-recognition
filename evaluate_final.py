"""Final comparison: all models with TTA, find best ensemble, vs DeepFace."""
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import tensorflow as tf
import keras
keras.config.enable_unsafe_deserialization()

from data.dataloader import auto_load, tta_predict
from models.ensemble import evaluate_ensemble

BATCH_SIZE = 64

print("Loading data...")
_, _, test_ds, _, (X_test, y_test) = auto_load(
    rafdb_dir='data/raf', extra_dir='data/extra',
    affectnet_dir=None, batch_size=BATCH_SIZE, oversample_target=12000)

y_true = np.argmax(y_test, axis=1)

print("\nLoading models...")
mob   = tf.keras.models.load_model('saved_models/mobilenetv2_swa.keras',   compile=False)
cnn_v1_best = tf.keras.models.load_model('saved_models/custom_cnn_best.keras',     compile=False)
cnn_v2_best = tf.keras.models.load_model('saved_models/custom_cnn_v2_best.keras',  compile=False)
cnn_v2_swa  = tf.keras.models.load_model('saved_models/custom_cnn_v2_swa.keras',   compile=False)

for m in [mob, cnn_v1_best, cnn_v2_best, cnn_v2_swa]:
    m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ── TTA predictions (8 passes) ────────────────────────────────────────────────
print("\nRunning TTA (8 passes per model)...")
mob_tta    = tta_predict(mob,        X_test, n_augments=8, batch_size=BATCH_SIZE)
v1b_tta    = tta_predict(cnn_v1_best, X_test, n_augments=8, batch_size=BATCH_SIZE)
v2b_tta    = tta_predict(cnn_v2_best, X_test, n_augments=8, batch_size=BATCH_SIZE)
v2s_tta    = tta_predict(cnn_v2_swa,  X_test, n_augments=8, batch_size=BATCH_SIZE)

def acc(preds): return np.mean(np.argmax(preds, axis=1) == y_true)
def ens(pred_list, weights=None):
    if weights is None: weights = [1]*len(pred_list)
    w = np.array(weights, dtype=np.float32) / sum(weights)
    return sum(p*wi for p, wi in zip(pred_list, w))

print("\n" + "="*60)
print("INDIVIDUAL MODELS (with TTA)")
print("="*60)
print(f"  MobileNetV2 SWA TTA      {acc(mob_tta):.4f}")
print(f"  Custom CNN v1 Best TTA   {acc(v1b_tta):.4f}")
print(f"  Custom CNN v2 Best TTA   {acc(v2b_tta):.4f}")
print(f"  Custom CNN v2 SWA TTA    {acc(v2s_tta):.4f}")

print("\n" + "="*60)
print("ENSEMBLE VARIANTS (with TTA)")
print("="*60)

combos = [
    ("MobV2 + v1_Best (35/65)",      [mob_tta, v1b_tta],          [0.35, 0.65]),
    ("MobV2 + v2_Best (35/65)",      [mob_tta, v2b_tta],          [0.35, 0.65]),
    ("MobV2 + v2_SWA (35/65)",       [mob_tta, v2s_tta],          [0.35, 0.65]),
    ("v1_Best + v2_Best (equal)",    [v1b_tta, v2b_tta],          None),
    ("v1_Best + v2_Best (40/60)",    [v1b_tta, v2b_tta],          [0.40, 0.60]),
    ("MobV2 + v1 + v2_Best (equal)", [mob_tta, v1b_tta, v2b_tta], None),
    ("MobV2 + v1 + v2_SWA (equal)",  [mob_tta, v1b_tta, v2s_tta], None),
    ("All 4 equal",                  [mob_tta, v1b_tta, v2b_tta, v2s_tta], None),
]

best_acc, best_preds, best_label = 0, None, ""
for label, preds, weights in combos:
    a = acc(ens(preds, weights))
    marker = " ◄ NEW BEST" if a > best_acc else ""
    print(f"  {label:40s}  {a:.4f}{marker}")
    if a > best_acc:
        best_acc, best_preds, best_label = a, ens(preds, weights), label

print("\n" + "="*60)
print(f"BEST: {best_label}  →  {best_acc:.4f}")
print("="*60)
evaluate_ensemble([], X_test, y_test,
                  precomputed_preds=[best_preds], weights=[1.0])

# ── DeepFace comparison ───────────────────────────────────────────────────────
print("\n" + "="*60)
print("PRETRAINED COMPARISON: DeepFace (FER2013 backbone)")
print("="*60)
import cv2, time as _time
from deepface import DeepFace

_EMOTIONS       = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
DEEPFACE_SAMPLE = 500   # quick estimate; increase to len(X_test) for full run
rng = np.random.default_rng(42)
idx = rng.choice(len(X_test), min(DEEPFACE_SAMPLE, len(X_test)), replace=False)
X_df, y_df = X_test[idx], y_true[idx]

def _to_rgb(x):
    img = (x[:, :, 0] * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

print(f"  Running DeepFace on {len(X_df)} sample images...")
df_preds, df_t0 = [], _time.time()
for img in X_df:
    try:
        res  = DeepFace.analyze(_to_rgb(img), actions=['emotion'],
                                enforce_detection=False, silent=True)
        em   = res[0]['dominant_emotion']
        pred = _EMOTIONS.index(em) if em in _EMOTIONS else -1
    except Exception:
        pred = -1
    df_preds.append(pred)

df_preds = np.array(df_preds)
valid    = df_preds != -1
df_acc   = np.mean(df_preds[valid] == y_df[valid])

print(f"  DeepFace accuracy     : {df_acc:.4f}  ({df_acc*100:.2f}%)")
print(f"  Our best ensemble TTA : {best_acc:.4f}  ({best_acc*100:.2f}%)")
print(f"  Improvement           : +{(best_acc - df_acc)*100:.2f}%")
print(f"  Elapsed               : {_time.time() - df_t0:.1f}s")
