import cv2
import mediapipe as mp
import numpy as np
import urllib.request
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions

# --------------> DOWNLOAD MODEL IF NOT PRESENT
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'face_landmarker.task')
if not os.path.exists(MODEL_PATH):
    print('Downloading face landmarker model...')
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
        MODEL_PATH
    )
    print('Model downloaded.')

# Set the transformation matrix (example: scaling)
transformation_matrix = np.array([[1.5, 0, 0],
                                  [0, 1.5, 0],
                                  [0, 0, 1]])

latest_result = None


def transform_3d_face(image, landmarks):
    transformed_landmarks = np.matmul(landmarks, transformation_matrix.T)
    transformed_image = image.copy()
    h, w = image.shape[:2]
    for i in range(transformed_landmarks.shape[0]):
        x, y, _ = transformed_landmarks[i]
        x = int(x * w)
        y = int(y * h)
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        cv2.circle(transformed_image, (x, y), 1, (255, 0, 0), -1)
    return transformed_image


def draw_face_connections(image, landmarks_norm):
    h, w = image.shape[:2]
    for lm in landmarks_norm:
        x = int(lm[0] * w)
        y = int(lm[1] * h)
        cv2.circle(image, (x, y), 1, (0, 255, 0), -1)


def result_callback(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result


# --------------> SETUP FACE LANDMARKER (LIVE STREAM MODE)
options = FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_faces=1,
    min_face_detection_confidence=0.5,
    result_callback=result_callback
)
landmarker = FaceLandmarker.create_from_options(options)

# --------------> INITIALIZE WEBCAM
cap = cv2.VideoCapture(0)
timestamp = 0

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Flip for mirror effect
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Send frame to landmarker
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    landmarker.detect_async(mp_image, timestamp)
    timestamp += 1

    transformed_image = image  # fallback

    if latest_result and latest_result.face_landmarks:
        for face_lms in latest_result.face_landmarks:
            landmarks = np.array([[lm.x, lm.y, lm.z]
                                 for lm in face_lms], dtype=np.float32)

            # Apply 3D transformation and draw scaled points
            transformed_image = transform_3d_face(image, landmarks)

            # Draw mesh connections
            draw_face_connections(transformed_image, landmarks)

    cv2.imshow('MediaPipe 3D Face Transform', transformed_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
landmarker.close()
cv2.destroyAllWindows()
