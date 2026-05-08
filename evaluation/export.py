"""
Export trained models to TFLite (mobile/edge) and ONNX (cross-platform).
TFLite with INT8 quantization runs on Android/Raspberry Pi at real-time speed.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import tensorflow as tf


def export_tflite(model_path, output_path=None, quantize=True, X_sample=None):
    """
    Converts a Keras model to TFLite.
    quantize=True applies INT8 quantization — 4× smaller, 2-3× faster on CPU/edge.
    X_sample: small numpy array used for INT8 calibration (100-500 images).
    """
    model       = tf.keras.models.load_model(model_path, compile=False)
    output_path = output_path or model_path.replace('.keras', '.tflite')

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize and X_sample is not None:
        converter.optimizations           = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset  = _representative_dataset(X_sample)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type    = tf.uint8
        converter.inference_output_type   = tf.uint8
        print("Applying INT8 quantization...")
    elif quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        print("Applying float16 quantization...")

    tflite_model = converter.convert()
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"TFLite model saved → {output_path}  ({size_mb:.2f} MB)")
    return output_path


def _representative_dataset(X_sample):
    def generator():
        for i in range(min(200, len(X_sample))):
            yield [X_sample[i:i+1].astype(np.float32)]
    return generator


def benchmark_tflite(tflite_path, X_test, n=100):
    """Measures TFLite inference speed and accuracy."""
    import time
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    inp_idx  = interpreter.get_input_details()[0]['index']
    out_idx  = interpreter.get_output_details()[0]['index']

    preds, times = [], []
    for i in range(min(n, len(X_test))):
        img = X_test[i:i+1].astype(np.float32)
        interpreter.set_tensor(inp_idx, img)
        t0 = time.perf_counter()
        interpreter.invoke()
        times.append(time.perf_counter() - t0)
        preds.append(interpreter.get_tensor(out_idx)[0])

    avg_ms = np.mean(times) * 1000
    print(f"TFLite avg inference: {avg_ms:.2f} ms/image  ({1000/avg_ms:.0f} FPS theoretical)")
    return np.array(preds)


def export_onnx(model_path, output_path=None):
    """Exports Keras model to ONNX format for cross-platform deployment."""
    try:
        import tf2onnx, onnx
    except ImportError:
        print("Install with: pip install tf2onnx onnx")
        return

    model       = tf.keras.models.load_model(model_path, compile=False)
    output_path = output_path or model_path.replace('.keras', '.onnx')
    spec        = (tf.TensorSpec(model.input_shape, tf.float32, name='input'),)
    _, _        = tf2onnx.convert.from_keras(model, input_signature=spec,
                                              output_path=output_path)
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"ONNX model saved → {output_path}  ({size_mb:.2f} MB)")
    return output_path


def export_all(X_sample=None):
    """Export all trained models to TFLite."""
    model_files = [
        'saved_models/custom_cnn_best.keras',
        'saved_models/student_best.keras',
        'saved_models/efficientnetb2_finetune_best.keras',
    ]
    os.makedirs('saved_models/exported', exist_ok=True)
    for path in model_files:
        if not os.path.exists(path):
            print(f"Skipping {path} — not found")
            continue
        name = os.path.basename(path).replace('.keras', '')
        export_tflite(path, f'saved_models/exported/{name}.tflite',
                      quantize=True, X_sample=X_sample)


if __name__ == '__main__':
    export_all()
