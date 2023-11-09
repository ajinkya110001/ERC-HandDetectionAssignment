import cv2
import mediapipe as mp
from cvzone import HandTrackingModule

cap=cv2.VideoCapture(0)

#TODO: Add required mediapipe functions
detector=HandTrackingModule.HandDetector()

while True:
    success, frame=cap.read()
    hands,frame=detector.findHands(frame)
    
    cv2.imshow("Hand Detection",frame)

    #TODO: Implement the functions
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cap.release
cv2.destroyAllWindows()
