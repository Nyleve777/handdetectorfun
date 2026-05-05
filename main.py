import cv2
import mediapipe as mp
import numpy as np
import urllib.request
import os
from gesture_detector import classify_gesture
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

MODEL_PATH = "hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Descargando modelo de deteccion de manos...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Modelo descargado!")

options = HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.5
)
detector = HandLandmarker.create_from_options(options)

MEME_MAP = {
    "thumbs_up": cv2.imread("memes/pulgararriba.jpeg"),
    "peace":     cv2.imread("memes/amorypaz.jpeg"),
    "open_hand": cv2.imread("memes/manoabierta.jpeg"),
   
}

def overlay_image(background, overlay, x, y, size=(300, 300)):
    overlay = cv2.resize(overlay, size)
    h, w = overlay.shape[:2]
    if y + h > background.shape[0] or x + w > background.shape[1]:
        return background
    background[y:y+h, x:x+w] = overlay[:, :, :3]
    return background

cap = cv2.VideoCapture(0)
current_gesture = "unknown"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    cam_w = frame.shape[1]
    canvas = np.zeros((frame.shape[0], cam_w + 400, 3), dtype=np.uint8)
    canvas[:, :cam_w] = frame

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = detector.detect(mp_image)

    if result.hand_landmarks:
        hand_lm = result.hand_landmarks[0]

        for lm in hand_lm:
            cx = int(lm.x * cam_w)
            cy = int(lm.y * frame.shape[0])
            cv2.circle(canvas, (cx, cy), 5, (0, 255, 0), -1)

        current_gesture = classify_gesture(hand_lm)

    meme = MEME_MAP.get(current_gesture)
    if meme is not None:
        canvas = overlay_image(canvas, meme, x=cam_w + 50, y=50)

    cv2.putText(canvas, f"Gesto: {current_gesture}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Meme Detector", canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()