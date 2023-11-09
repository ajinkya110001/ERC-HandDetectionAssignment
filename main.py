import cv2
import mediapipe as mp

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands()

# Start capturing video from the webcam.
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    # Flip the frame and convert it from BGR to RGB.
    frame = cv2.flip(frame, 1)
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands.
    result = hands.process(frameRGB)

    # If hand landmarks are detected, classify each hand as left or right.
    if result.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
            # Determine if this hand is left or right.
            handedness = result.multi_handedness[idx].classification[0].label

            # Draw the hand landmarks and connections.
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Display the label at the wrist landmark.
            cv2.putText(
                frame,
                handedness,
                tuple(
                    int(x)
                    for x in [
                        hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
                        * frame.shape[1],
                        hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
                        * frame.shape[0],
                    ]
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )

    # Display the frame with the detected hand landmarks.
    cv2.imshow("Hand Tracking", frame)

    # If the 'q' key is pressed, break the loop and stop the video capture.
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and close all windows.
video_capture.release()
cv2.destroyAllWindows()
