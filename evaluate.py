"""Final evaluation of all saved models."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import tensorflow as tf
import keras
keras.config.enable_unsafe_deserialization()

from data.dataloader import auto_load
from models.ensemble import evaluate_ensemble

BATCH_SIZE = 64

print("Loading data...")
train_ds, val_ds, test_ds, _, (X_test, y_test) = auto_load(
    rafdb_dir='data/raf', extra_dir='data/extra',
    affectnet_dir=None, batch_size=BATCH_SIZE, oversample_target=12000)

print("\nLoading models...")
mob_model = tf.keras.models.load_model('saved_models/mobilenetv2_swa.keras', compile=False)
mob_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

cnn_model = tf.keras.models.load_model('saved_models/custom_cnn_swa.keras', compile=False)
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("\n" + "="*60)
print("Final Test Evaluation")
print("="*60)
for name, model in [('MobileNetV2 SWA', mob_model), ('Custom CNN SWA', cnn_model)]:
    loss, acc = model.evaluate(test_ds, verbose=0)
    print(f"{name:20s}  test_acc={acc:.4f}  test_loss={loss:.4f}")

print("\nEnsemble evaluation:")
evaluate_ensemble([mob_model, cnn_model], X_test, y_test)
