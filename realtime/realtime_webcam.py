import os, sys, threading
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2, time, argparse, collections
import numpy as np
import tensorflow as tf
from detection.face_detector import (detect_faces_haar, detect_faces_mtcnn,
                                     crop_face, predict_emotion, EMOTIONS)
from detection.face_tracker import MultiTracker

COLORS = {
    'angry':    (0,   0,   255),
    'disgust':  (0,   128, 255),
    'fear':     (0,   255, 255),
    'happy':    (0,   255, 0),
    'sad':      (255, 128, 0),
    'surprise': (255, 0,   0),
    'neutral':  (200, 200, 200),
}
HISTORY_LEN = 60
SIDEBAR_W   = 230
TIMELINE_H  = 130
IMG_SIZE    = 48


# ---------------------------------------------------------------------------
# Live Grad-CAM
# ---------------------------------------------------------------------------

def get_gradcam_heatmap(model, face_arr, class_idx, last_conv_name=None):
    if last_conv_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_name = layer.name
                break
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_name).output, model.output])
    inp = tf.cast(face_arr[np.newaxis], tf.float32)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(inp)
        loss = preds[:, class_idx]
    grads   = tape.gradient(loss, conv_out)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam     = tf.reduce_sum(tf.multiply(weights, conv_out[0]), axis=-1)
    cam     = tf.maximum(cam, 0) / (tf.math.reduce_max(cam) + 1e-8)
    cam     = cam.numpy()
    heatmap = np.uint8(255 * cam)
    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return colored


def overlay_gradcam_on_face(frame, x, y, w, h, heatmap, alpha=0.45):
    heatmap_resized = cv2.resize(heatmap, (w, h))
    roi = frame[y:y+h, x:x+w]
    if roi.shape[:2] == heatmap_resized.shape[:2]:
        frame[y:y+h, x:x+w] = cv2.addWeighted(roi, 1-alpha, heatmap_resized, alpha, 0)
    return frame


# ---------------------------------------------------------------------------
# Sidebar UI
# ---------------------------------------------------------------------------

def draw_sidebar(frame_h, probs, history):
    sidebar = np.zeros((frame_h, SIDEBAR_W, 3), dtype=np.uint8)
    bar_h   = (frame_h - TIMELINE_H - 30) // len(EMOTIONS)

    cv2.putText(sidebar, "Emotion Probabilities", (5, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
    for i, (em, p) in enumerate(zip(EMOTIONS, probs)):
        y0      = 22 + i * bar_h
        bar_len = int(p * (SIDEBAR_W - 75))
        color   = COLORS[em]
        cv2.rectangle(sidebar, (5, y0), (5 + bar_len, y0 + bar_h - 3), color, -1)
        cv2.putText(sidebar, f"{em[:5]:5s} {p:.0%}",
                    (5, y0 + bar_h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

    tl_y = frame_h - TIMELINE_H
    cv2.line(sidebar, (0, tl_y - 2), (SIDEBAR_W, tl_y - 2), (80, 80, 80), 1)
    cv2.putText(sidebar, "Timeline", (5, tl_y + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

    if len(history) >= 2:
        hist_arr = np.array(history)
        T        = len(hist_arr)
        for cls_idx, em in enumerate(EMOTIONS):
            color = COLORS[em]
            for t in range(1, T):
                x1 = int((t-1) / HISTORY_LEN * SIDEBAR_W)
                x2 = int(t     / HISTORY_LEN * SIDEBAR_W)
                y1 = frame_h - int(hist_arr[t-1, cls_idx] * (TIMELINE_H - 20)) - 5
                y2 = frame_h - int(hist_arr[t,   cls_idx] * (TIMELINE_H - 20)) - 5
                cv2.line(sidebar, (x1, y1), (x2, y2), color, 1)

    return sidebar


# ---------------------------------------------------------------------------
# DeepFace compare panel
# ---------------------------------------------------------------------------

class DeepFaceWorker:
    """Runs DeepFace.analyze in a background thread to keep webcam fast."""

    def __init__(self):
        self._result   = ('neutral', 0.0)   # (dominant_emotion, confidence)
        self._lock     = threading.Lock()
        self._pending  = False

    def submit(self, rgb_img):
        if self._pending:
            return
        self._pending = True
        threading.Thread(target=self._run, args=(rgb_img.copy(),), daemon=True).start()

    def _run(self, rgb_img):
        try:
            from deepface import DeepFace
            result  = DeepFace.analyze(rgb_img, actions=['emotion'],
                                       enforce_detection=False, silent=True)
            em_dict = result[0]['emotion']
            dom     = result[0]['dominant_emotion']
            conf    = em_dict[dom] / 100.0
            with self._lock:
                self._result = (dom, conf)
        except Exception:
            pass
        finally:
            self._pending = False

    def get(self):
        with self._lock:
            return self._result


def draw_compare_panel(frame_h, our_emotion, our_conf,
                       df_emotion, df_conf, df_ready):
    """Two-column panel: Our Model | DeepFace."""
    W      = 340
    panel  = np.zeros((frame_h, W, 3), dtype=np.uint8)
    col_w  = W // 2

    def draw_col(title, emotion, conf, x_off, ready=True):
        color = COLORS.get(emotion, (200, 200, 200))
        cv2.putText(panel, title, (x_off + 4, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)
        cv2.line(panel, (x_off, 25), (x_off + col_w - 2, 25), (80, 80, 80), 1)

        if ready:
            em_disp = emotion.upper()
            cv2.putText(panel, em_disp, (x_off + 4, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            conf_bar = int(conf * (col_w - 10))
            cv2.rectangle(panel, (x_off + 4, 70),
                          (x_off + 4 + conf_bar, 84), color, -1)
            cv2.putText(panel, f"{conf:.0%}", (x_off + 4, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            # Emotion bars
            for i, em in enumerate(EMOTIONS):
                yy = 115 + i * 22
                cv2.putText(panel, em[:4], (x_off + 2, yy + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.30, (150, 150, 150), 1)
        else:
            cv2.putText(panel, "...", (x_off + 4, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 100), 1)

    draw_col("Our CNN", our_emotion, our_conf, 0)
    cv2.line(panel, (col_w, 0), (col_w, frame_h), (80, 80, 80), 1)
    draw_col("DeepFace", df_emotion, df_conf, col_w, ready=df_ready)

    return panel


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(model_path='saved_models/custom_cnn_best.keras',
        detector='haar', cam_id=0, show_gradcam=True, compare=False):

    model   = tf.keras.models.load_model(model_path, compile=False)
    cap     = cv2.VideoCapture(cam_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    tracker    = MultiTracker(smoothing=8)
    history    = collections.deque(maxlen=HISTORY_LEN)
    prev_time  = time.time()
    best_probs = np.ones(7) / 7

    last_conv = next((l.name for l in reversed(model.layers)
                      if isinstance(l, tf.keras.layers.Conv2D)), None)

    df_worker   = DeepFaceWorker() if compare else None
    df_emotion  = 'neutral'
    df_conf     = 0.0
    df_ready    = False
    our_emotion = 'neutral'
    our_conf    = 0.0
    df_frame_skip = 0   # run DeepFace every N frames

    mode_str = "COMPARE mode  |  " if compare else ""
    print(f"Real-time emotion detection  |  {mode_str}Press Q to quit  |  Press G to toggle Grad-CAM")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces_haar(gray) if detector == 'haar' else detect_faces_mtcnn(frame)

        face_probs = []
        for (x, y, w, h) in faces:
            face_arr             = crop_face(gray, x, y, w, h)
            emotion, conf, probs = predict_emotion(model, face_arr)
            face_probs.append(probs)

        tracks = tracker.update(faces, face_probs) if faces else []

        for track in tracks:
            x, y, w, h = track.box
            emotion     = track.emotion
            confidence  = track.confidence
            probs       = track.smooth_probs
            color       = COLORS[emotion]
            best_probs  = probs
            our_emotion = emotion
            our_conf    = confidence

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            label = f"#{track.id} {emotion.upper()} {confidence:.0%}"
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

            if show_gradcam and last_conv:
                face_arr = crop_face(gray, x, y, w, h)
                try:
                    heatmap = get_gradcam_heatmap(model, face_arr,
                                                   probs.argmax(), last_conv)
                    frame = overlay_gradcam_on_face(frame, x, y, w, h, heatmap)
                except Exception:
                    pass

            # Submit to DeepFace every 5 frames to avoid lag
            if compare and df_worker and df_frame_skip % 5 == 0 and faces:
                fx, fy, fw, fh = x, y, w, h
                face_bgr = frame[fy:fy+fh, fx:fx+fw]
                face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB) \
                           if face_bgr.size > 0 else None
                if face_rgb is not None:
                    df_worker.submit(face_rgb)

        df_frame_skip += 1

        # Retrieve latest DeepFace result
        if compare and df_worker:
            res = df_worker.get()
            if res[1] > 0:
                df_emotion, df_conf = res
                df_ready = True

        history.append(best_probs.copy())

        fps = 1.0 / (time.time() - prev_time + 1e-6)
        prev_time = time.time()
        cv2.putText(frame, f"FPS:{fps:.0f}  Faces:{len(faces)}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cam_label = "Grad-CAM: ON" if show_gradcam else "Grad-CAM: OFF"
        cv2.putText(frame, cam_label, (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 0), 1)
        if compare:
            cv2.putText(frame, "COMPARE: ON", (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

        sidebar = draw_sidebar(frame.shape[0], best_probs, history)

        if compare:
            comp = draw_compare_panel(frame.shape[0], our_emotion, our_conf,
                                      df_emotion, df_conf, df_ready)
            display = np.hstack([frame, sidebar, comp])
        else:
            display = np.hstack([frame, sidebar])

        title = 'Emotion Detection  |  Q=quit  G=Grad-CAM'
        if compare:
            title += '  [COMPARE vs DeepFace]'
        cv2.imshow(title, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('g'):
            show_gradcam = not show_gradcam

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model',     default='saved_models/custom_cnn_best.keras')
    p.add_argument('--detector',  choices=['haar', 'mtcnn'], default='haar')
    p.add_argument('--cam',       type=int, default=0)
    p.add_argument('--no-gradcam', action='store_true')
    p.add_argument('--compare',   action='store_true',
                   help='Show DeepFace side-by-side comparison panel')
    args = p.parse_args()
    run(args.model, args.detector, args.cam, not args.no_gradcam, args.compare)
