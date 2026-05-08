import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']


def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model.")


def compute_gradcam(model, img, class_idx, last_conv_name=None):
    if last_conv_name is None:
        last_conv_name = get_last_conv_layer(model)

    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_name).output, model.output])

    img_tensor = tf.cast(img[np.newaxis], tf.float32)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_tensor)
        loss = preds[:, class_idx]

    grads   = tape.gradient(loss, conv_out)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam     = tf.reduce_sum(tf.multiply(weights, conv_out[0]), axis=-1)
    cam     = tf.maximum(cam, 0) / (tf.math.reduce_max(cam) + 1e-8)
    cam     = cam.numpy()

    heatmap = np.uint8(255 * cam)
    heatmap = np.array(tf.image.resize(heatmap[..., np.newaxis], (48, 48)))[..., 0]
    return heatmap


def overlay_gradcam(img, heatmap, alpha=0.4):
    img_rgb = np.concatenate([img * 255] * 3, axis=-1).astype(np.uint8)
    heat_rgb = np.uint8(cm.jet(heatmap / 255.0)[..., :3] * 255)
    return np.uint8(img_rgb * (1 - alpha) + heat_rgb * alpha)


def visualize_gradcam_all_classes(model, X_test, y_test, save_dir='evaluation/gradcam'):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 7, figsize=(21, 6))

    for cls_idx, emotion in enumerate(EMOTIONS):
        idxs = np.where(y_test.argmax(1) == cls_idx)[0]
        img  = X_test[idxs[0]]

        heatmap  = compute_gradcam(model, img, cls_idx)
        overlay  = overlay_gradcam(img, heatmap)

        axes[0, cls_idx].imshow(img[..., 0], cmap='gray')
        axes[0, cls_idx].set_title(emotion, fontsize=10)
        axes[0, cls_idx].axis('off')

        axes[1, cls_idx].imshow(overlay)
        axes[1, cls_idx].axis('off')

    axes[0, 0].set_ylabel('Original', fontsize=10)
    axes[1, 0].set_ylabel('Grad-CAM', fontsize=10)
    plt.suptitle('Grad-CAM — All Emotion Classes', fontsize=14)
    plt.tight_layout()
    path = os.path.join(save_dir, 'gradcam_all_classes.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved → {path}")
