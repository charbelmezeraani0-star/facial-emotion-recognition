"""Benchmark DeepFace emotion recognition on the RAF-DB test set."""
import os, sys, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import cv2
from deepface import DeepFace
from sklearn.metrics import accuracy_score, classification_report

from data.dataloader import auto_load

BATCH_SIZE = 64
EMOTIONS   = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
SAMPLE     = 500   # set to None to run on full test set (slow ~15-30 min)

print("Loading data...")
_, _, test_ds, _, (X_test, y_test) = auto_load(
    rafdb_dir='data/raf', extra_dir='data/extra',
    affectnet_dir=None, batch_size=BATCH_SIZE, oversample_target=12000)

y_true = np.argmax(y_test, axis=1)

# Optionally subsample for speed
if SAMPLE and SAMPLE < len(X_test):
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_test), SAMPLE, replace=False)
    X_eval, y_eval = X_test[idx], y_true[idx]
    print(f"Sampling {SAMPLE} / {len(X_test)} test images for speed.")
else:
    X_eval, y_eval = X_test, y_true
    print(f"Full test set: {len(X_test)} images.")

def to_rgb_uint8(x):
    """Convert per-image-standardised 48x48x1 float32 to uint8 RGB."""
    img = (x[:, :, 0] * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

print(f"\nRunning DeepFace.analyze on {len(X_eval)} images...")
print("(enforce_detection=False — images are already cropped faces)\n")

y_pred  = []
failed  = 0
t_start = time.time()

for i, img in enumerate(X_eval):
    rgb = to_rgb_uint8(img)
    try:
        result   = DeepFace.analyze(rgb, actions=['emotion'],
                                    enforce_detection=False, silent=True)
        emotion  = result[0]['dominant_emotion']
        pred_idx = EMOTIONS.index(emotion) if emotion in EMOTIONS else -1
        y_pred.append(pred_idx)
    except Exception:
        y_pred.append(-1)
        failed += 1

    if (i + 1) % 50 == 0:
        elapsed = time.time() - t_start
        eta     = elapsed / (i + 1) * (len(X_eval) - i - 1)
        acc_so_far = np.mean(np.array(y_pred) == y_eval[:i+1])
        print(f"  [{i+1:4d}/{len(X_eval)}]  acc={acc_so_far:.4f}  "
              f"failed={failed}  ETA={eta:.0f}s")

y_pred = np.array(y_pred)
valid  = y_pred != -1

print("\n" + "=" * 60)
print("DeepFace  —  Benchmark Results")
print("=" * 60)
print(f"  Samples evaluated : {len(X_eval)}")
print(f"  Failed / skipped  : {failed}")
acc = accuracy_score(y_eval[valid], y_pred[valid])
print(f"  Accuracy          : {acc:.4f}  ({acc*100:.2f}%)")
print()
print(classification_report(y_eval[valid], y_pred[valid], target_names=EMOTIONS))
print("=" * 60)
print(f"  Total time : {time.time() - t_start:.1f}s")
