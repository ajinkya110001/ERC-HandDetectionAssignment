import cv2
import mediapipe as mp
import numpy as np

from google.protobuf.json_format import MessageToDict 

cap=cv2.VideoCapture(0)
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

with mp_hands.Hands(min_detection_confidence=0.65,min_tracking_confidence=0.35) as hands:
    while True:
        success, frame=cap.read()
        
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        image = cv2.flip(image,1)
        results = hands.process(image)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
       
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS)
                
               

		 
            if len(results.multi_handedness) == 2: 
                cv2.putText(image, 'Both Hands', (250, 50), 
						cv2.FONT_HERSHEY_COMPLEX, 0.9, 
						(0, 255, 0), 2)
            else: 
                  for i in results.multi_handedness: 
                    label = MessageToDict(i)[ 
					'classification'][0]['label'] 
                    
                    if label == 'Left':
                        cv2.putText(image, label+' Hand', (20, 50), 
								cv2.FONT_HERSHEY_COMPLEX, 0.9, 
								(0, 255, 0), 2) 
                        
                    if label == 'Right': 
                        cv2.putText(image, label+' Hand', (460, 50), 
								cv2.FONT_HERSHEY_COMPLEX, 
								0.9, (0, 255, 0), 2) 

        
        cv2.imshow("Hand Detection",image)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
