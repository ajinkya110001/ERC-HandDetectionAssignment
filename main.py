import cv2
import mediapipe as mp

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Start capturing video from the webcam.
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Flip the frame and convert it from BGR to RGB.
    frame = cv2.flip(frame, 1)
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands.
    result = hands.process(frameRGB)

    # If hand landmarks are detected, classify each hand as left or right.
    if result.multi_hand_landmarks:
        for hand in result.multi_handedness:
            handType = hand.classification[0].label
            print(handType)

    # If the 'q' key is pressed, break the loop and stop the video capture.
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and close all windows.
cap.release()
cv2.destroyAllWindows()
