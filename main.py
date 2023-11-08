import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands #it is an instance of mediapipe's hand module 
hands = mp_hands.Hands() #it detects/analyses hands 


while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1) #it flips the input video/image capture 
    rgbframe=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)#here we are changing the bgr format to rgb as required by mediapiper
    results = hands.process(rgbframe) # this process the rgbframe

    if results.multi_handedness:
        for handedness in (results.multi_handedness):
            hand_label = handedness.classification[0].label #used to detect orientation left or right 
                
      cv2.putText(frame, hand_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) #puts the text on the output screen
   
    cv2.imshow('Hand Detection', frame)

    if cv2.waitKey(1) & 0xFF ==ord('q'):  
        break
