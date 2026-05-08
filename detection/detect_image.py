import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import tensorflow as tf
from detection.face_detector import (detect_faces_haar, detect_faces_mtcnn,
                                     crop_face, predict_emotion, draw_result)


def run(image_path, model_path='saved_models/custom_cnn_best.keras',
        detector='haar', save=True):
    model  = tf.keras.models.load_model(model_path)
    frame  = cv2.imread(image_path)
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces  = detect_faces_haar(gray) if detector == 'haar' else detect_faces_mtcnn(frame)

    for (x, y, w, h) in faces:
        face_arr = crop_face(gray, x, y, w, h)
        emotion, confidence, probs = predict_emotion(model, face_arr)
        frame = draw_result(frame, x, y, w, h, emotion, confidence, probs)

    cv2.imshow('Emotion Detection', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if save:
        out_path = image_path.replace('.', '_result.')
        cv2.imwrite(out_path, frame)
        print(f"Saved → {out_path}")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('image')
    p.add_argument('--model', default='saved_models/custom_cnn_best.keras')
    p.add_argument('--detector', choices=['haar', 'mtcnn'], default='haar')
    args = p.parse_args()
    run(args.image, args.model, args.detector)
