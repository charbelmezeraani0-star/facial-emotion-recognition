import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from data.dataloader import auto_load, mixup_dataset
from models.custom_cnn import build_custom_cnn
from models.ensemble import soft_voting_ensemble
from training.callbacks import (get_callbacks, get_initial_epoch,
                                 load_or_build, StochasticWeightAveraging)

tf.keras.mixed_precision.set_global_policy('float32')

BATCH_SIZE    = 64
EPOCHS_CNN    = 100
STEPS_PER_EPOCH = 84530 // BATCH_SIZE   # ≈ 1320

RAFDB_DIR     = 'data/raf'
EXTRA_DIR     = 'data/extra'
AFFECTNET_DIR = None


def compile_model(model, lr, use_schedule=False, label_smoothing=0.0):
    if use_schedule:
        lr = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=lr,
            decay_steps=STEPS_PER_EPOCH * EPOCHS_CNN,
            alpha=1e-7)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=['accuracy'])


def plot_history(model_name):
    csv_path = f'logs/{model_name}_history.csv'
    if not os.path.exists(csv_path):
        return
    os.makedirs('evaluation/plots', exist_ok=True)
    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(df['accuracy'],     label='train')
    axes[0].plot(df['val_accuracy'], label='val')
    axes[0].set_title(f'{model_name} — Accuracy')
    axes[0].legend()
    axes[1].plot(df['loss'],     label='train')
    axes[1].plot(df['val_loss'], label='val')
    axes[1].set_title(f'{model_name} — Loss')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(f'evaluation/plots/{model_name}_curves.png', dpi=150)
    plt.close()
    print(f"Saved → evaluation/plots/{model_name}_curves.png")


def apply_swa(model, model_name, train_ds, val_ds, swa_epochs=8):
    swa_cb = StochasticWeightAveraging(swa_start_epoch=0)
    compile_model(model, lr=1e-5)
    model.fit(train_ds, validation_data=val_ds,
              epochs=swa_epochs, callbacks=[swa_cb], verbose=1)
    model.set_weights(swa_cb.get_swa_weights())
    model.save(f'saved_models/{model_name}_swa.keras')
    print(f"SWA weights saved → saved_models/{model_name}_swa.keras")
    return model


def train_custom_cnn(train_ds, val_ds):
    print("\n" + "="*60)
    print("Training Custom CNN")
    print("="*60)

    model = load_or_build('custom_cnn', build_custom_cnn)
    initial_epoch = get_initial_epoch('custom_cnn')

    compile_model(model, lr=1e-3, use_schedule=True, label_smoothing=0.0)
    model.fit(
        train_ds, validation_data=val_ds,
        epochs=EPOCHS_CNN, initial_epoch=initial_epoch,
        callbacks=get_callbacks('custom_cnn', patience=20, reduce_lr=False))

    model = apply_swa(model, 'custom_cnn', train_ds, val_ds)
    plot_history('custom_cnn')
    return model


def main():
    print("Loading datasets...")
    train_ds, val_ds, test_ds, class_weights, (X_test, y_test) = auto_load(
        rafdb_dir=RAFDB_DIR,
        extra_dir=EXTRA_DIR,
        affectnet_dir=AFFECTNET_DIR,
        batch_size=BATCH_SIZE,
        oversample_target=12000)

    # Load pre-trained MobileNetV2 (57% test accuracy)
    import keras
    keras.config.enable_unsafe_deserialization()
    mob_model = tf.keras.models.load_model('saved_models/mobilenetv2_swa.keras', compile=False)
    compile_model(mob_model, lr=1e-5)
    print("Loaded MobileNetV2 SWA model")

    # Train custom CNN on standard augmented data
    cnn_model = train_custom_cnn(train_ds, val_ds)

    # Ensemble
    ensemble = soft_voting_ensemble([mob_model, cnn_model])
    ensemble.save('saved_models/ensemble.keras')

    # Final evaluation
    print("\n" + "="*60)
    print("Final Test Evaluation")
    print("="*60)
    for m_name, model in [('MobileNetV2', mob_model), ('CustomCNN', cnn_model)]:
        loss, acc = model.evaluate(test_ds, verbose=0)
        print(f"{m_name:20s}  test_acc={acc:.4f}  test_loss={loss:.4f}")

    from models.ensemble import evaluate_ensemble
    evaluate_ensemble([mob_model, cnn_model], X_test, y_test)


if __name__ == '__main__':
    main()
