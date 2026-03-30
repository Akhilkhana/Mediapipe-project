# -------------->  IMPORT LIBRARY
import os
import urllib.request
import numpy as np
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -----------> 3D OBJECT DETECTION FROM IMAGES GIVEN IN URL
# Define a helper method to fetch images given a URL
def url_to_array(url):
    req = urllib.request.urlopen(url)
    arr = np.array(bytearray(req.read()), dtype=np.uint8)
    arr = cv2.imdecode(arr, -1)
    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return arr


# --------------> DOWNLOAD MODEL IF NOT PRESENT
MODEL_PATH = 'efficientdet_lite0.tflite'
if not os.path.exists(MODEL_PATH):
    print('Downloading object detection model...')
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite',
        MODEL_PATH
    )
    print('Model downloaded.')

# --------------> LOAD AN IMAGE FROM A URL
mug = 'https://t4.ftcdn.net/jpg/06/19/97/81/360_F_619978183_55XXYY6Szc8paQrDBG1UNZsRtCepVWD5.jpg'
image_array = url_to_array(mug)

# ------> CREATE OBJECT DETECTOR USING MEDIAPIPE TASKS API
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.ObjectDetectorOptions(
    base_options=base_options,
    score_threshold=0.2
)
detector = vision.ObjectDetector.create_from_options(options)

# --------------> RUN INFERENCE
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_array)
detection_result = detector.detect(mp_image)

# --------------> DISPLAY THE RESULT
if not detection_result.detections:
    print('No objects detected.')
else:
    annotated_image = image_array.copy()
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        start = (bbox.origin_x, bbox.origin_y)
        end = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)
        cv2.rectangle(annotated_image, start, end, (0, 255, 0), 3)

        label = detection.categories[0].category_name
        score = detection.categories[0].score
        cv2.putText(annotated_image, f'{label}: {score:.2f}',
                    (bbox.origin_x, max(bbox.origin_y - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # -----------> PLOT THE RESULT
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(annotated_image)
    ax.axis('off')
    plt.show()
