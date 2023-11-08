import cv2
import mediapipe as mp
from google.protobuf.json_format import MessageToDict

cap=cv2.VideoCapture(0)
#TODO: Add required mediapipe functions

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    success, frame=cap.read()
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame,hand_landmark,mpHands.HAND_CONNECTIONS)
    
	
        if len(results.multi_handedness) == 2: 
            cv2.putText(frame, 'Both Hands', (250, 50), cv2.FONT_ITALIC, 0.9, (200, 200, 200), 2) 
  
         
        else: 
            for i in results.multi_handedness: 
                label = MessageToDict(i)['classification'][0]['label'] 
                
                if label == 'Right':  
                    cv2.putText(frame, 'Left'+' Hand', (20, 50), cv2.FONT_ITALIC, 0.9, (240, 0, 0), 2) 
  
                if label == 'Left': 
                    cv2.putText(frame, 'Right'+' Hand', (460, 50), cv2.FONT_ITALIC, 0.9, (240, 0, 0), 2)


    cv2.imshow("Hand Detection",frame)

    #TODO: Implement the functions
    
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break