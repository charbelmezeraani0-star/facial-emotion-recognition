import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2, time
import tensorflow as tf
from detection.face_detector import (detect_faces_haar, detect_faces_mtcnn,
                                     crop_face, predict_emotion, draw_result)


def run(video_path, model_path='saved_models/custom_cnn_best.keras',
        detector='haar', save=False):
    model = tf.keras.models.load_model(model_path)
    cap   = cv2.VideoCapture(video_path)

    writer = None
    if save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps    = cap.get(cv2.CAP_PROP_FPS) or 25
        w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(video_path.replace('.', '_result.'), fourcc, fps, (w, h))

    prev = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces_haar(gray) if detector == 'haar' else detect_faces_mtcnn(frame)

        for (x, y, w, h) in faces:
            face_arr = crop_face(gray, x, y, w, h)
            emotion, confidence, probs = predict_emotion(model, face_arr)
            frame = draw_result(frame, x, y, w, h, emotion, confidence, probs)

        fps_val = 1 / (time.time() - prev + 1e-6)
        prev = time.time()
        cv2.putText(frame, f"FPS: {fps_val:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow('Video Emotion Detection', frame)
        if writer:
            writer.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('video')
    p.add_argument('--model', default='saved_models/custom_cnn_best.keras')
    p.add_argument('--detector', choices=['haar', 'mtcnn'], default='haar')
    p.add_argument('--save', action='store_true')
    args = p.parse_args()
    run(args.video, args.model, args.detector, args.save)
