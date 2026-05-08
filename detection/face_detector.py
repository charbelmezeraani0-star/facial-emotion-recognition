import cv2
import numpy as np

HAAR_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
EMOTIONS  = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
IMG_SIZE  = 48

_mtcnn = None


def _get_mtcnn():
    global _mtcnn
    if _mtcnn is None:
        from mtcnn import MTCNN
        _mtcnn = MTCNN()
    return _mtcnn


def detect_faces_haar(frame_gray):
    detector = cv2.CascadeClassifier(HAAR_PATH)
    faces    = detector.detectMultiScale(frame_gray, scaleFactor=1.1,
                                         minNeighbors=5, minSize=(30, 30))
    return faces   # list of (x, y, w, h)


def detect_faces_mtcnn(frame_bgr):
    rgb   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    dets  = _get_mtcnn().detect_faces(rgb)
    boxes = []
    for d in dets:
        if d['confidence'] < 0.9:
            continue
        x, y, w, h = d['box']
        x, y = max(0, x), max(0, y)
        boxes.append((x, y, w, h))
    return boxes


def crop_face(frame_gray, x, y, w, h):
    face = frame_gray[y:y+h, x:x+w]
    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    return face.astype('float32') / 255.0


def predict_emotion(model, face_arr):
    inp   = face_arr.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    probs = model.predict(inp, verbose=0)[0]
    idx   = probs.argmax()
    return EMOTIONS[idx], float(probs[idx]), probs


def draw_result(frame, x, y, w, h, emotion, confidence, probs):
    color = (0, 255, 0)
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    label = f"{emotion} {confidence:.0%}"
    cv2.putText(frame, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Mini bar chart overlay
    bar_x, bar_y = x, y + h + 5
    for i, (em, p) in enumerate(zip(EMOTIONS, probs)):
        bar_w = int(p * 80)
        cv2.rectangle(frame, (bar_x, bar_y + i*14),
                      (bar_x + bar_w, bar_y + i*14 + 10), color, -1)
        cv2.putText(frame, f"{em[:3]} {p:.0%}",
                    (bar_x + bar_w + 3, bar_y + i*14 + 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    return frame
