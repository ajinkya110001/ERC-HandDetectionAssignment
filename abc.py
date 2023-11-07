import mediapipe as mp
import cv2
from google.protobuf.json_format import MessageToDict

# Initialize the MediaPipe Hands model.
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      model_complexity=1,
                      min_detection_confidence=0.50,
                      min_tracking_confidence=0.50,
                      max_num_hands=2)

cap = cv2.VideoCapture(0)
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()  # Fixed variable name from "frame" to "img".
    img = cv2.flip(img, 1)  # Fixed variable name from "frame" to "img".
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        if len(results.multi_hand_landmarks) == 2:  # The condition to check for 2 hands.
            cv2.putText(img, 'Both Hands', (50, 50),
                        cv2.FONT_ITALIC,
                        0.9, (0, 255, 0), 2)
        else:
            for i in results.multi_handedness:
                label = MessageToDict(i)['classification'][0]['label']
                if label == 'Left':
                    cv2.putText(img, label+' Hand', (20, 50),
                                cv2.FONT_ITALIC, 0.9,
                                (255, 0, 0), 2)
                if label == 'Right':
                    cv2.putText(img, label+' Hand', (460, 50),
                                cv2.FONT_ITALIC,
                                0.9, (0, 0, 255), 2)

    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break