import cv2
import mediapipe as mp

cap=cv2.VideoCapture(0)
#TODO: Add required mediapipe functions

while True:
    success, frame=cap.read()

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            for point in landmarks.landmark:
                height, width, _ = frame.shape
                cx, cy = int(point.x * width), int(point.y * height)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
    
    cv2.imshow("Hand Detection",frame)

    #TODO: Implement the functions

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
