import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model


def soft_voting_ensemble(models, input_shape=(48, 48, 1), num_classes=7):
    """Build a Keras model that averages softmax outputs of multiple models."""
    inp = layers.Input(shape=input_shape)
    preds = [m(inp) for m in models]
    avg   = layers.Average()(preds) if len(preds) > 1 else preds[0]
    return Model(inp, avg, name='Ensemble')


def predict_ensemble(models, X, batch_size=64):
    probs = np.mean([m.predict(X, batch_size=batch_size, verbose=0)
                     for m in models], axis=0)
    return probs


def evaluate_ensemble(models, X_test, y_test,
                      precomputed_preds=None, weights=None, batch_size=64):
    from sklearn.metrics import accuracy_score, classification_report
    if precomputed_preds is not None:
        preds = precomputed_preds
    else:
        preds = [m.predict(X_test, batch_size=batch_size, verbose=0) for m in models]

    if weights is not None:
        w = np.array(weights, dtype=np.float32) / sum(weights)
        probs = sum(p * wi for p, wi in zip(preds, w))
    else:
        probs = np.mean(preds, axis=0)

    y_pred = probs.argmax(axis=1)
    y_true = y_test.argmax(axis=1)
    acc    = accuracy_score(y_true, y_pred)
    print(f"Ensemble Accuracy: {acc:.4f}")
    print(classification_report(y_true, y_pred,
          target_names=['angry','disgust','fear','happy','sad','surprise','neutral']))
    return acc, probs
