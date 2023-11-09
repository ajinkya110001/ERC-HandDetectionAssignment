import tkinter as tk

import cv2
import mediapipe as mp

# Constants
WINDOW_NAME = "Hand Tracking"

# Initialize MediaPipe Hands and Drawing utility.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands()


def process_frame(frame):
    # Flip the frame and convert it from BGR to RGB.
    frame = cv2.flip(frame, 1)
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands.
    result = hands.process(frameRGB)

    # If hand landmarks are detected, classify each hand as left or right and draw landmarks.
    if result.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
            # Determine if this hand is left or right.
            handedness = result.multi_handedness[idx].classification[0].label

            # Choose color based on handedness.
            color = (0, 255, 0) if handedness == "Right" else (0, 0, 255)

            # Draw the hand landmarks and connections with the chosen color.
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=color),
                mp_drawing.DrawingSpec(color=color),
            )

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
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    return frame


def main():
    # Start capturing video from the webcam.
    video_capture = cv2.VideoCapture(0)

    # Get the screen size
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()

    while True:
        ret, frame = video_capture.read()
        frame = process_frame(frame)

        # Display the frame with the detected hand landmarks.
        cv2.imshow(WINDOW_NAME, frame)

        # Get the window size
        window_width = frame.shape[1]
        window_height = frame.shape[0]

        # Calculate the position to place the window in the center of the screen
        position_x = (screen_width // 2) - (window_width // 2)
        position_y = (screen_height // 2) - (window_height // 2)

        # Move the window to the center of the screen
        cv2.moveWindow(WINDOW_NAME, int(screen_width / 6), int(screen_height / 6.5))

        # If the 'q' key is pressed, break the loop and stop the video capture.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the video capture and close all windows.
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
