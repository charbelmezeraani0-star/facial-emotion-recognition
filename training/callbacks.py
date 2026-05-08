import os
import json
import numpy as np
import tensorflow as tf


class SaveEpochState(tf.keras.callbacks.Callback):
    """Saves current epoch so training can resume from exact point."""
    def __init__(self, state_path):
        super().__init__()
        self.state_path = state_path

    def on_epoch_end(self, epoch, logs=None):
        with open(self.state_path, 'w') as f:
            json.dump({'epoch': epoch + 1}, f)


class StochasticWeightAveraging(tf.keras.callbacks.Callback):
    """Averages model weights over all epochs after swa_start_epoch."""
    def __init__(self, swa_start_epoch=0):
        super().__init__()
        self.swa_start_epoch = swa_start_epoch
        self.swa_weights = None
        self.swa_count = 0

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.swa_start_epoch:
            w = self.model.get_weights()
            if self.swa_weights is None:
                self.swa_weights = [np.zeros_like(wi) for wi in w]
            for sw, wi in zip(self.swa_weights, w):
                sw += wi
            self.swa_count += 1

    def get_swa_weights(self):
        if self.swa_count == 0:
            return self.model.get_weights()
        return [sw / self.swa_count for sw in self.swa_weights]


def get_callbacks(model_name, monitor='val_accuracy', patience=10,
                  reduce_lr=True):
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    cbs = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f'saved_models/{model_name}_best.keras',
            monitor=monitor, save_best_only=True, verbose=1),

        tf.keras.callbacks.ModelCheckpoint(
            filepath=f'saved_models/{model_name}_latest.keras',
            save_best_only=False, verbose=0),

        tf.keras.callbacks.EarlyStopping(
            monitor=monitor, patience=patience,
            restore_best_weights=True, verbose=1),

        tf.keras.callbacks.CSVLogger(
            f'logs/{model_name}_history.csv', append=False),

        SaveEpochState(f'saved_models/{model_name}_state.json'),
    ]
    if reduce_lr:
        cbs.insert(3, tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5,
            min_lr=1e-7, verbose=1))
    return cbs


def get_initial_epoch(model_name):
    state_path = f'saved_models/{model_name}_state.json'
    if os.path.exists(state_path):
        with open(state_path) as f:
            epoch = json.load(f)['epoch']
        print(f"Resuming {model_name} from epoch {epoch}")
        return epoch
    return 0


def load_or_build(model_name, build_fn):
    latest = f'saved_models/{model_name}_latest.keras'
    if os.path.exists(latest):
        print(f"Loading checkpoint: {latest}")
        import keras
        keras.config.enable_unsafe_deserialization()
        return tf.keras.models.load_model(latest, compile=False)
    print(f"No checkpoint for {model_name} — building fresh.")
    return build_fn()
