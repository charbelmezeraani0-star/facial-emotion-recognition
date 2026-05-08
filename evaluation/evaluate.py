import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score,
                             roc_curve, auc, precision_recall_curve)
from sklearn.manifold import TSNE
from sklearn.preprocessing import label_binarize
import tensorflow as tf

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
COLORS   = ['#e74c3c','#8e44ad','#3498db','#f1c40f','#2ecc71','#e67e22','#95a5a6']


def evaluate_model(model, X_test, y_test, model_name='model'):
    y_true  = y_test.argmax(axis=1)
    y_probs = model.predict(X_test, batch_size=64, verbose=0)
    y_pred  = y_probs.argmax(axis=1)
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average='weighted')
    print(f"\n{'='*50}")
    print(f"{model_name} — Accuracy: {acc:.4f} | Weighted F1: {f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=EMOTIONS))
    return acc, f1, y_true, y_pred, y_probs


def plot_confusion_matrix(y_true, y_pred, model_name):
    os.makedirs('evaluation/plots', exist_ok=True)
    cm   = confusion_matrix(y_true, y_pred)
    norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(9, 7))
    sns.heatmap(norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=EMOTIONS, yticklabels=EMOTIONS)
    plt.title(f'{model_name} — Confusion Matrix')
    plt.ylabel('True'); plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'evaluation/plots/{model_name}_confusion.png', dpi=150)
    plt.close()
    print(f"Saved → evaluation/plots/{model_name}_confusion.png")


def plot_roc_curves(y_test, y_probs, model_name):
    """One ROC curve per emotion class (one-vs-rest)."""
    os.makedirs('evaluation/plots', exist_ok=True)
    y_bin = label_binarize(y_test.argmax(1), classes=range(7))
    plt.figure(figsize=(10, 7))
    for i, (emotion, color) in enumerate(zip(EMOTIONS, COLORS)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc     = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f'{emotion} (AUC={roc_auc:.2f})')
    plt.plot([0,1],[0,1],'k--', lw=1)
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} — ROC Curves (One vs Rest)')
    plt.legend(loc='lower right'); plt.tight_layout()
    plt.savefig(f'evaluation/plots/{model_name}_roc.png', dpi=150)
    plt.close()
    print(f"Saved → evaluation/plots/{model_name}_roc.png")


def plot_pr_curves(y_test, y_probs, model_name):
    """Precision-Recall curves — more informative than ROC for imbalanced classes."""
    os.makedirs('evaluation/plots', exist_ok=True)
    y_bin = label_binarize(y_test.argmax(1), classes=range(7))
    plt.figure(figsize=(10, 7))
    for i, (emotion, color) in enumerate(zip(EMOTIONS, COLORS)):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_probs[:, i])
        pr_auc       = auc(rec, prec)
        plt.plot(rec, prec, color=color, lw=2,
                 label=f'{emotion} (AUC={pr_auc:.2f})')
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title(f'{model_name} — Precision-Recall Curves')
    plt.legend(loc='upper right'); plt.tight_layout()
    plt.savefig(f'evaluation/plots/{model_name}_pr.png', dpi=150)
    plt.close()
    print(f"Saved → evaluation/plots/{model_name}_pr.png")


def plot_tsne(model, X_test, y_test, model_name, n_samples=2000):
    """
    t-SNE of penultimate layer features.
    Shows how well the model separates emotion classes in feature space.
    Well-separated clusters = the model has learned meaningful representations.
    """
    os.makedirs('evaluation/plots', exist_ok=True)
    # Build feature extractor (second-to-last layer)
    feature_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.layers[-2].output)

    idx  = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)
    feats = feature_model.predict(X_test[idx], batch_size=64, verbose=0)
    labels = y_test[idx].argmax(1)

    print("Running t-SNE (this takes ~1 min)...")
    tsne   = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    emb    = tsne.fit_transform(feats)

    plt.figure(figsize=(10, 8))
    for i, (emotion, color) in enumerate(zip(EMOTIONS, COLORS)):
        mask = labels == i
        plt.scatter(emb[mask, 0], emb[mask, 1], c=color,
                    label=emotion, alpha=0.6, s=15)
    plt.title(f'{model_name} — t-SNE Feature Embedding')
    plt.legend(markerscale=2); plt.tight_layout()
    plt.savefig(f'evaluation/plots/{model_name}_tsne.png', dpi=150)
    plt.close()
    print(f"Saved → evaluation/plots/{model_name}_tsne.png")


def compare_models(results: dict):
    """results = {'ModelName': (accuracy, f1)}"""
    df = pd.DataFrame(results, index=['Accuracy', 'Weighted F1']).T
    df = df.sort_values('Accuracy', ascending=False)
    print("\n=== Model Comparison ===")
    print(df.to_string())
    os.makedirs('evaluation/plots', exist_ok=True)
    df.plot(kind='bar', figsize=(10, 5), ylim=(0, 1), rot=0,
            color=['#3498db', '#e74c3c'])
    plt.title('Model Comparison'); plt.tight_layout()
    plt.savefig('evaluation/plots/model_comparison.png', dpi=150)
    plt.close()
    df.to_csv('evaluation/model_comparison.csv')
    print("Saved → evaluation/plots/model_comparison.png")
    return df


def full_evaluation(model, X_test, y_test, model_name):
    """Run all evaluation metrics + plots for one model."""
    acc, f1, y_true, y_pred, y_probs = evaluate_model(model, X_test, y_test, model_name)
    plot_confusion_matrix(y_true, y_pred, model_name)
    plot_roc_curves(y_test, y_probs, model_name)
    plot_pr_curves(y_test, y_probs, model_name)
    plot_tsne(model, X_test, y_test, model_name)
    return acc, f1
