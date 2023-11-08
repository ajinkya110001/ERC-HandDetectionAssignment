import cv2
import mediapipe as mp

cap=cv2.VideoCapture(0)
mp_hands = mp.solutions.hands #media pipe for hands
hands = mp_hands.Hands()

while True:
    success, frame=cap.read()

    if not success:
        continue

    # Convert the frame to RGB
    frame_to_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Use MediaPipe Hands to detect hand landmarks
    results = hands.process(frame_to_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Implement logic to classify left or right hand based on landmarks
            thumb_tip_x = hand_landmarks.landmark[4].x

            # Classify the hand as left or right based on the thumb tip's x-coordinate
            if thumb_tip_x < 0.5:
                hand_label = "Left Hand"
            else:
                hand_label = "Right Hand"

            # Draw landmarks on the frame
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            # Display hand label on the frame
            cv2.putText(frame, hand_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame with hand detection and classification
    cv2.imshow("Hand Detection", frame)

    # Check for the 'q' key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    cv2.imshow("Hand Detection",frame)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
