"""
Fine-tune EfficientNetB2 (ImageNet pretrained) on RAF-DB at 96×96.
Data stored as uint8 (~300 MB RAM) — cast to float32 only per batch.

Phases:
  1. Head only        15 epochs  lr=1e-3   (backbone frozen)
  2. Top-40% unfreeze 20 epochs  lr=2e-4
  3. Full fine-tune   10 epochs  lr=5e-5
  4. SWA               8 epochs  lr=1e-5
"""
import os, sys, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import cv2
import tensorflow as tf
import keras
keras.config.enable_unsafe_deserialization()
from sklearn.model_selection import train_test_split
from training.callbacks import StochasticWeightAveraging

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE    = 96          # 96×96 — 4× more pixels than 48×48, ~300 MB RAM
BATCH_SIZE  = 32
EPOCHS_HEAD = 15
EPOCHS_MID  = 20
EPOCHS_FULL = 10
EPOCHS_SWA  = 8
RAFDB_DIR   = 'data/raf/DATASET'

RAFDB_TO_LABEL = {
    '1': 5,  # surprise
    '2': 2,  # fear
    '3': 1,  # disgust
    '4': 3,  # happy
    '5': 4,  # sad
    '6': 0,  # angry
    '7': 6,  # neutral
}
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

tf.keras.mixed_precision.set_global_policy('mixed_float16')


# ── Load images as uint8 (small RAM footprint) ────────────────────────────────

def load_split(split):
    X, y = [], []
    split_dir = os.path.join(RAFDB_DIR, split)
    for folder_num, label in RAFDB_TO_LABEL.items():
        folder = os.path.join(split_dir, folder_num)
        if not os.path.exists(folder):
            continue
        for fname in os.listdir(folder):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE),
                             interpolation=cv2.INTER_LANCZOS4)
            X.append(img)
            y.append(label)
    return np.array(X, dtype=np.uint8), np.array(y, dtype=np.int32)


print("Loading RAF-DB into RAM (96×96 RGB, uint8)...")
t0 = time.time()
X_all,  y_all  = load_split('train')
X_test, y_test = load_split('test')
X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all, test_size=0.1, random_state=42, stratify=y_all)

mb = (X_train.nbytes + X_val.nbytes + X_test.nbytes) / 1e6
print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
print(f"  RAM used: {mb:.0f} MB  |  Loaded in {time.time()-t0:.1f}s")


# ── tf.data — cast uint8→float32 lazily per batch ─────────────────────────────

preprocess_input = tf.keras.applications.efficientnet.preprocess_input

augment_layer = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.12),
    tf.keras.layers.RandomZoom(0.12),
    tf.keras.layers.RandomTranslation(0.10, 0.10),
], name='augmentation')


def make_dataset(X, y, augment=False, shuffle=False):
    labels = tf.keras.utils.to_categorical(y, 7).astype(np.float32)
    # Store X as uint8 — cast inside map (never materialises full float32 array)
    ds = tf.data.Dataset.from_tensor_slices((X, labels))
    if shuffle:
        ds = ds.shuffle(len(X), reshuffle_each_iteration=True)
    # Cast to float32 here, per-element
    ds = ds.map(lambda x, lbl: (tf.cast(x, tf.float32), lbl),
                num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(lambda x, lbl: (augment_layer(x, training=True), lbl),
                    num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda x, lbl: (preprocess_input(x), lbl),
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


train_ds = make_dataset(X_train, y_train, augment=True,  shuffle=True)
val_ds   = make_dataset(X_val,   y_val,   augment=False, shuffle=False)
test_ds  = make_dataset(X_test,  y_test,  augment=False, shuffle=False)


# ── Model builder ─────────────────────────────────────────────────────────────

def build_model(unfreeze_from=None):
    tf.keras.backend.clear_session()
    base = tf.keras.applications.EfficientNetB2(
        include_top=False, weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling='avg')

    if unfreeze_from is None:
        base.trainable = False
    else:
        base.trainable = True
        for layer in base.layers[:unfreeze_from]:
            layer.trainable = False

    inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x   = base(inp, training=(unfreeze_from is not None))
    x   = tf.keras.layers.BatchNormalization()(x)
    x   = tf.keras.layers.Dropout(0.4)(x)
    x   = tf.keras.layers.Dense(256, activation='relu',
                                  kernel_initializer='he_normal')(x)
    x   = tf.keras.layers.Dropout(0.3)(x)
    x   = tf.keras.layers.Dense(7, dtype='float32')(x)
    out = tf.keras.layers.Activation('softmax', dtype='float32')(x)
    return tf.keras.Model(inp, out, name='EfficientNetB2_FER')


def get_callbacks(name, patience=8):
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            f'saved_models/{name}.keras',
            monitor='val_accuracy', save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=patience,
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.CSVLogger(f'logs/{name}_history.csv'),
    ]


# ── Quick timing probe ────────────────────────────────────────────────────────

print("\nEstimating epoch time...")
m_tmp = build_model(unfreeze_from=None)
m_tmp.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
for i, batch in enumerate(train_ds.take(7)):
    if i == 2: t_probe = time.time()
    m_tmp(batch[0], training=False)
step_ms  = (time.time() - t_probe) / 5 * 1000
steps_ep = len(X_train) // BATCH_SIZE
spe      = step_ms / 1000 * steps_ep
del m_tmp; tf.keras.backend.clear_session()

print(f"  {step_ms:.0f} ms/step  →  ~{spe:.0f}s per epoch")
est = spe * (EPOCHS_HEAD*0.85 + EPOCHS_MID*1.1 + EPOCHS_FULL*1.2 + EPOCHS_SWA*1.2)
print(f"  Estimated total: {est/60:.0f} min\n")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Head only
# ══════════════════════════════════════════════════════════════════════════════
print("="*60)
print(f"PHASE 1 — Head only  ({EPOCHS_HEAD} epochs, lr=1e-3)")
print("="*60)

model = build_model(unfreeze_from=None)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3, clipnorm=1.0),
              loss='categorical_crossentropy', metrics=['accuracy'])
print(f"  Trainable: "
      f"{sum(tf.size(v).numpy() for v in model.trainable_variables)/1e6:.2f}M params")

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD,
          callbacks=get_callbacks('efficientnetb2_head', patience=8), verbose=1)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Unfreeze top 40%
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"PHASE 2 — Top-40% unfreeze  ({EPOCHS_MID} epochs, lr=2e-4)")
print("="*60)

n_base        = len(tf.keras.applications.EfficientNetB2(
    include_top=False, weights=None,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)).layers)
unfreeze_from = int(n_base * 0.60)

model2 = build_model(unfreeze_from=unfreeze_from)
model2.set_weights(model.get_weights())
model2.compile(optimizer=tf.keras.optimizers.Adam(2e-4, clipnorm=1.0),
               loss='categorical_crossentropy', metrics=['accuracy'])
print(f"  Trainable: "
      f"{sum(tf.size(v).numpy() for v in model2.trainable_variables)/1e6:.2f}M params")

model2.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_MID,
           callbacks=get_callbacks('efficientnetb2_mid', patience=8), verbose=1)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Full fine-tune
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"PHASE 3 — Full fine-tune  ({EPOCHS_FULL} epochs, lr=5e-5)")
print("="*60)

model3 = build_model(unfreeze_from=0)
model3.set_weights(model2.get_weights())
model3.compile(optimizer=tf.keras.optimizers.Adam(5e-5, clipnorm=1.0),
               loss='categorical_crossentropy', metrics=['accuracy'])

model3.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FULL,
           callbacks=get_callbacks('efficientnetb2_best', patience=6), verbose=1)
model3.save('saved_models/efficientnetb2_best.keras')
print("Saved → saved_models/efficientnetb2_best.keras")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — SWA
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"PHASE 4 — SWA  ({EPOCHS_SWA} epochs, lr=1e-5)")
print("="*60)

swa_cb = StochasticWeightAveraging(swa_start_epoch=0)
model3.compile(optimizer=tf.keras.optimizers.Adam(1e-5, clipnorm=1.0),
               loss='categorical_crossentropy', metrics=['accuracy'])
model3.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_SWA,
           callbacks=[swa_cb], verbose=1)
model3.set_weights(swa_cb.get_swa_weights())
model3.save('saved_models/efficientnetb2_swa.keras')
print("Saved → saved_models/efficientnetb2_swa.keras")


# ══════════════════════════════════════════════════════════════════════════════
# Final evaluation
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("FINAL EVALUATION")
print("="*60)

best = tf.keras.models.load_model(
    'saved_models/efficientnetb2_best.keras', compile=False)
best.compile(optimizer='adam', loss='categorical_crossentropy',
             metrics=['accuracy'])

_, acc_best = best.evaluate(test_ds, verbose=0)
_, acc_swa  = model3.evaluate(test_ds, verbose=0)
print(f"  EfficientNetB2 Best (no TTA) :  {acc_best:.4f}")
print(f"  EfficientNetB2 SWA  (no TTA) :  {acc_swa:.4f}")

# TTA — augmented passes over uint8 test set (cast lazily)
def tta_predict(model, X, n=8):
    aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.12),
        tf.keras.layers.RandomZoom(0.12),
    ])
    preds = []
    for _ in range(n):
        batch_preds = []
        for i in range(0, len(X), BATCH_SIZE):
            b = tf.cast(X[i:i+BATCH_SIZE], tf.float32)
            b = aug(b, training=True)
            b = preprocess_input(b)
            batch_preds.append(model(b, training=False).numpy())
        preds.append(np.vstack(batch_preds))
    return np.mean(preds, axis=0)

print("\nRunning TTA (8 passes)...")
y_true = y_test

for name, m in [('EfficientNetB2 Best + TTA', best),
                ('EfficientNetB2 SWA  + TTA', model3)]:
    p   = tta_predict(m, X_test, n=8)
    acc = np.mean(np.argmax(p, axis=1) == y_true)
    print(f"  {name:35s}  {acc:.4f}")

from sklearn.metrics import classification_report
best_tta = tta_predict(best, X_test, n=8)
best_acc = np.mean(np.argmax(best_tta, axis=1) == y_true)

print(f"\n{'='*60}")
print("COMPARISON")
print("="*60)
print(f"  DeepFace pretrained (zero-shot)      :  0.5280")
print(f"  Our SE-CNN 4-model ensemble + TTA    :  0.7045")
print(f"  EfficientNetB2 fine-tuned + TTA      :  {best_acc:.4f}  ← NEW")
print("="*60)
print(classification_report(y_true, np.argmax(best_tta, axis=1),
      target_names=EMOTIONS))
