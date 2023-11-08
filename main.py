import mediapipe as mp
import cv2

# Initialize the MediaPipe Hands model.
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

show_lines = False #initially landmark off.

# ===== May not be required ============

def flip_landmarks_horizontally(landmarks):    # Function to flip landmarks horizontally
    for landmark in landmarks:
        landmark.x = 1.0 - landmark.x

def textinfo():        #text/info to dislay
    xcord = frame.shape[1] - 250
    ycord = frame.shape[0] - 20
    ycord2 = frame.shape[0] - 7
    cv2.putText(frame,  "Press 't' to toggle landmarks on", (xcord, ycord), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
    cv2.putText(frame,  "press 'q' to stop ", (xcord, ycord2), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
#==================-----------------------========

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgRGB = cv2.flip(imgRGB, 1)  # Flip the video frame captured horizontally
    results = hands.process(imgRGB)

    textinfo() #just to display some info

    if results.multi_hand_landmarks:
        if show_lines:
            # This part shows the landmark lines
            for x in results.multi_hand_landmarks:
                flip_landmarks_horizontally(x.landmark)
                mpDraw.draw_landmarks(frame, x, mpHands.HAND_CONNECTIONS)

        if len(results.multi_hand_landmarks) == 2:  # Dual hand check
            cv2.putText(frame, "Both Hands", (250, 250), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 1)
        else:
            for x in results.multi_handedness:
                label = x.classification[0].label  # Used multi_handedness to detect left/right hand
                if label == 'Left':
                    cv2.putText(frame, label + ' Hand', (20, 50), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 0, 0), 1)
                if label == 'Right':
                    cv2.putText(frame, label + ' Hand', (460, 50), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255), 1)


    cv2.imshow('Hand Detection', frame)

    #Ask to show the landmarker . Wait few moments after pressing q to press any key(q/t)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
    elif cv2.waitKey(1) == ord('t'):
        show_lines = not show_lines
    
cap.release()
cv2.destroyAllWindows()