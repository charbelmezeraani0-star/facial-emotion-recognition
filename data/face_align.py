import cv2
import numpy as np

IMG_SIZE = 48

# Desired eye positions as fraction of output image size (standard alignment target)
LEFT_EYE_TARGET  = (0.35, 0.40)
RIGHT_EYE_TARGET = (0.65, 0.40)

_mp_face_mesh = None


def _get_face_mesh():
    global _mp_face_mesh
    if _mp_face_mesh is None:
        import mediapipe as mp
        _mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1,
            refine_landmarks=True, min_detection_confidence=0.5)
    return _mp_face_mesh


def align_face_mediapipe(img_bgr, out_size=IMG_SIZE):
    """
    Aligns a face image so that both eyes are always horizontal and centered.
    Alignment makes emotion features rotation-invariant, improving accuracy
    especially for tilted or rotated faces.
    Returns aligned grayscale image (uint8), or None if no face detected.
    """
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = _get_face_mesh().process(rgb)
    if not res.multi_face_landmarks:
        return None

    lm   = res.multi_face_landmarks[0].landmark
    h, w = img_bgr.shape[:2]

    # MediaPipe landmark indices for eye centres
    # Left eye: landmarks 33, 133 | Right eye: 362, 263
    def eye_center(a, b):
        return np.array([(lm[a].x + lm[b].x) / 2 * w,
                         (lm[a].y + lm[b].y) / 2 * h])

    left_eye  = eye_center(33,  133)
    right_eye = eye_center(362, 263)

    # Compute rotation angle to make eyes horizontal
    dy    = right_eye[1] - left_eye[1]
    dx    = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    eye_dist      = np.linalg.norm(right_eye - left_eye)
    desired_dist  = (RIGHT_EYE_TARGET[0] - LEFT_EYE_TARGET[0]) * out_size
    scale         = desired_dist / (eye_dist + 1e-6)

    center = tuple(((left_eye + right_eye) / 2).astype(int))
    M      = cv2.getRotationMatrix2D(center, angle, scale)

    # Shift so left eye lands at target position
    tx = LEFT_EYE_TARGET[0] * out_size - (M[0, 0] * left_eye[0] + M[0, 1] * left_eye[1])
    ty = LEFT_EYE_TARGET[1] * out_size - (M[1, 0] * left_eye[0] + M[1, 1] * left_eye[1])
    M[0, 2] += tx
    M[1, 2] += ty

    aligned = cv2.warpAffine(img_bgr, M, (out_size, out_size), flags=cv2.INTER_CUBIC)
    gray    = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    return gray


def align_face_simple(gray_face):
    """
    Fallback alignment using Haar eye detector when MediaPipe is unavailable.
    Less accurate but always works.
    """
    import cv2
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(gray_face, 1.1, 3)
    if len(eyes) < 2:
        return gray_face   # can't align, return as-is

    eyes  = sorted(eyes, key=lambda e: e[0])[:2]
    cx    = [e[0] + e[2]//2 for e in eyes]
    cy    = [e[1] + e[3]//2 for e in eyes]
    angle = np.degrees(np.arctan2(cy[1] - cy[0], cx[1] - cx[0]))
    h, w  = gray_face.shape
    M     = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(gray_face, M, (w, h))
