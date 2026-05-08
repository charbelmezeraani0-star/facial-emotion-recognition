import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import tensorflow as tf


def draw_architecture_diagram(model, save_path='evaluation/plots/architecture.png'):
    """
    Draws a clean layer-by-layer architecture diagram of any Keras model.
    Uses matplotlib — no extra libraries needed.
    """
    os.makedirs('evaluation/plots', exist_ok=True)

    layer_info = []
    for layer in model.layers:
        name   = layer.name
        ltype  = type(layer).__name__
        try:
            shape = str(layer.output_shape)
        except Exception:
            shape = '?'
        params = layer.count_params()
        layer_info.append((name, ltype, shape, params))

    fig_h = max(6, len(layer_info) * 0.35)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.axis('off')

    COLOR_MAP = {
        'Conv2D':              '#3498db',
        'BatchNormalization':  '#95a5a6',
        'Activation':         '#2ecc71',
        'MaxPooling2D':        '#e67e22',
        'Dropout':             '#e74c3c',
        'Dense':               '#9b59b6',
        'GlobalAveragePooling2D': '#1abc9c',
        'Multiply':            '#f39c12',
        'Add':                 '#f39c12',
        'InputLayer':          '#34495e',
    }

    row_h = 1.0 / (len(layer_info) + 1)
    for i, (name, ltype, shape, params) in enumerate(layer_info):
        y     = 1.0 - (i + 1) * row_h
        color = COLOR_MAP.get(ltype, '#bdc3c7')
        rect  = mpatches.FancyBboxPatch(
            (0.01, y - row_h * 0.4), 0.98, row_h * 0.8,
            boxstyle='round,pad=0.01', facecolor=color, alpha=0.85,
            edgecolor='white', linewidth=0.5)
        ax.add_patch(rect)
        ax.text(0.03, y, f"{name}  [{ltype}]",
                va='center', ha='left', fontsize=7, color='white', weight='bold')
        ax.text(0.60, y, shape, va='center', ha='left', fontsize=6.5, color='white')
        ax.text(0.92, y, f"{params:,} params",
                va='center', ha='right', fontsize=6.5, color='white')

    # Legend
    handles = [mpatches.Patch(color=c, label=l) for l, c in COLOR_MAP.items()]
    ax.legend(handles=handles, loc='upper right', fontsize=6,
              bbox_to_anchor=(1.0, 1.0), framealpha=0.7)

    total = model.count_params()
    ax.set_title(f'{model.name}  —  Total parameters: {total:,}',
                 fontsize=12, weight='bold', pad=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Saved architecture diagram → {save_path}")


def plot_all_architectures():
    from models.custom_cnn import build_custom_cnn
    from models.transfer_learning import build_mobilenetv2, build_resnet50, build_efficientnetb2

    models = {
        'CustomCNN':      build_custom_cnn,
        'MobileNetV2':    lambda: build_mobilenetv2()[0],
        'ResNet50':       lambda: build_resnet50()[0],
        'EfficientNetB2': lambda: build_efficientnetb2()[0],
    }
    for name, build_fn in models.items():
        print(f"Drawing {name}...")
        model = build_fn()
        draw_architecture_diagram(model, f'evaluation/plots/{name}_architecture.png')


if __name__ == '__main__':
    plot_all_architectures()
