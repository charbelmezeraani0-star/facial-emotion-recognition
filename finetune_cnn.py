"""Continue training custom CNN from best checkpoint for 30 more epochs."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import tensorflow as tf
import keras
keras.config.enable_unsafe_deserialization()

from data.dataloader import auto_load
from training.callbacks import get_callbacks, StochasticWeightAveraging

tf.keras.mixed_precision.set_global_policy('float32')

BATCH_SIZE      = 64
FINETUNE_EPOCHS = 30
STEPS_PER_EPOCH = 84530 // BATCH_SIZE  # ≈ 1320

print("Loading data...")
train_ds, val_ds, test_ds, _, (X_test, y_test) = auto_load(
    rafdb_dir='data/raf', extra_dir='data/extra',
    affectnet_dir=None, batch_size=BATCH_SIZE, oversample_target=12000)

print("Loading best checkpoint...")
model = tf.keras.models.load_model('saved_models/custom_cnn_best.keras', compile=False)

# Fresh cosine decay from 2e-4 (10× lower than original 1e-3)
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=2e-4,
    decay_steps=STEPS_PER_EPOCH * FINETUNE_EPOCHS,
    alpha=1e-7)

model.compile(
    optimizer=tf.keras.optimizers.Adam(lr_schedule, clipnorm=1.0),
    loss='categorical_crossentropy',
    metrics=['accuracy'])

print(f"\nFine-tuning for {FINETUNE_EPOCHS} epochs at lr=2e-4 → ~0")

model.fit(
    train_ds, validation_data=val_ds,
    epochs=FINETUNE_EPOCHS,
    callbacks=get_callbacks('custom_cnn_v2', patience=15, reduce_lr=False),
    verbose=1)

# SWA pass
print("\nApplying SWA...")
swa_cb = StochasticWeightAveraging(swa_start_epoch=0)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5, clipnorm=1.0),
    loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_ds, validation_data=val_ds,
          epochs=8, callbacks=[swa_cb], verbose=1)
model.set_weights(swa_cb.get_swa_weights())
model.save('saved_models/custom_cnn_v2_swa.keras')
print("Saved → saved_models/custom_cnn_v2_swa.keras")

# Quick test eval
_, acc = model.evaluate(test_ds, verbose=0)
print(f"\nCustom CNN v2 SWA  test_acc={acc:.4f}")
