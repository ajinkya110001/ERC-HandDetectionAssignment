import mediapipe as mp
import cv2
cap = cv2.VideoCapture(0)

# Initialize the MediaPipe Hands model.
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils   #just for landmarks line view 

#landmark flip function define
def flip_landmarks_h(landmarks):
    for landmark in landmarks:
        landmark.x = 1.0 - landmark.x

while True:
    success, frame=cap.read()
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgRGB = cv2.flip(imgRGB, 1)   #flip the video frame capttured horizontally 
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        #---- This part shows the landmark lines ---- 
        for landmarks in results.multi_hand_landmarks:
                flip_landmarks_h(landmarks.landmark)
                mpDraw.draw_landmarks(frame, landmarks, mpHands.HAND_CONNECTIONS)
        #=======------------------------------========

        if len(results.multi_hand_landmarks) == 2 : #dual hand check 
            cv2.putText(frame, "Both Hands", (250 , 250), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 1)
        else:
            for x in results.multi_handedness:
                label = x.classification[0].label # just used multi_handedness to detect 
                if label == 'Left':cv2.putText(frame, label+' Hand', (20, 50),cv2.FONT_HERSHEY_DUPLEX , 0.9, (255, 0, 0), 1)
                if label == 'Right':cv2.putText(frame, label+' Hand', (460, 50),cv2.FONT_HERSHEY_DUPLEX , 0.9, (0, 0, 255), 1)

    cv2.imshow('Hand Detection', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
    

cap.release()
cv2.destroyAllWindows()
