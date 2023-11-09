import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False , max_num_hands=2 , min_detection_confidence=0.75 , min_tracking_confidence=0.75)

cap=cv2.VideoCapture(0)

while True:
    success, frame=cap.read()

    
    frame = cv2.flip(frame , 1)

    result = hands.process(frame)

    if result.multi_hand_landmarks:
        for hand in result.multi_handedness:
            handedness = hand.classification[0].label
            confidence = hand.classification[0].score
            cv2.putText(frame , str(handedness) + ", " + str(round(confidence , 2)) , (75,50) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 2 , cv2.LINE_AA)

    #frame = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)
    cv2.imshow("Hand Detection",frame)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
