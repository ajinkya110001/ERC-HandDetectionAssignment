import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)
while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_handedness:
        for idx, handedness in enumerate(results.multi_handedness):
            hand_label = handedness.classification[0].label 
            hand_score = handedness.classification[0].score 
            if hand_score > 0.5:  
                
                cv2.putText(frame, hand_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
   
    cv2.imshow('Hand Detection', frame)

    if cv2.waitKey(1) & 0xFF ==ord('q'):  
        break
