import cv2
import mediapipe as mp
mp_drawing=mp.solutions.drawing_utils
mp_hands=mp.solutions.hands
cap=cv2.VideoCapture(0)
#TODO: Add required mediapipe functions

while True:
    success, frame=cap.read()
    
    cv2.imshow("Hand Detection",frame)

    #TODO: Implement the functions

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
