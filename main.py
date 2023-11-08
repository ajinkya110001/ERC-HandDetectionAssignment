import cv2
import mediapipe as mp

cap=cv2.VideoCapture(0)

mp_hands = mp.solutions.mediapipe.python.solutions.hands # idk why this is so messed up

hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.75,
                       min_tracking_confidence=0.75)

while True:
    success, frame=cap.read()
    
    cv2.imshow("Hand Detection",frame)

    success, frame = cap.read()

    frame = cv2.flip(frame, 1) # flipping the image cuz by default, it shows a mirror of the real world

    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(frameRGB)

    if result.multi_hand_landmarks:
        for hand in result.multi_handedness:
            handType = hand.classification[0].label # marks whether it is left or right
            print(handType)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
