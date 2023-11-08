import cv2
import mediapipe as mp

#cap=cv2.VideoCapture(0)
#TODO: Add required mediapipe functions

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands() 

# OpenCV Video Capture
cap = cv2.VideoCapture(1) 

while True:
    success, frame=cap.read()
    if not success:
        continue

    #TODO: Implement the functions
# Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmarks for the hand
            landmarks = hand_landmarks.landmark

            # Determine the hand position (left or right) based on the position of landmarks
            leftmost_x = min(landmark.x for landmark in landmarks)
            rightmost_x = max(landmark.x for landmark in landmarks)

            if leftmost_x < 0.4:
                hand_position = "Right Hand"
            elif rightmost_x > 0.6:
                hand_position = "Left Hand"
            else:
                hand_position = "Unknown"
            #print(hand_position)
            # Draw landmarks and hand position on the frame
            for landmark in landmarks:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            cv2.putText(frame, hand_position, (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 10)

    cv2.imshow("Hand Position Detection", frame)
    
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()