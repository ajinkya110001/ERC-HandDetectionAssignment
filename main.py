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

def textinfo():        #text/info to display and configuring its postion and text format 
    xcord = frame.shape[1] - 250     
    ycord = frame.shape[0] - 20
    ycord2 = frame.shape[0] - 7
    cv2.putText(frame,  "Press 't' to toggle landmarks on", (xcord, ycord), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
    cv2.putText(frame,  "press 'q' to stop ", (xcord, ycord2), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
#==================-----------------------========

cap = cv2.VideoCapture(0) #captures the live video feed 

while True:
    success, frame = cap.read()
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #Converting to RGB for mediapipe requirement 
    frameRGB = cv2.flip(frameRGB, 1)  # Flip the video frame captured  horizontally
    results = hands.process(frameRGB)

    textinfo() #just to display some info on the frame about the key to press .

    if results.multi_hand_landmarks:
        if show_lines: # This part shows the landmark lines . Used it in the if statement to make it toogle on/off . Intially its off .
            for x in results.multi_hand_landmarks:
                flip_landmarks_horizontally(x.landmark) 
                mpDraw.draw_landmarks(frame, x, mpHands.HAND_CONNECTIONS) #draw the landmarks if the show_lines() is set True

        if len(results.multi_hand_landmarks) == 2:  # Dual hand check and display settings 
            cv2.putText(frame, "Two Hands", (250, 250), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 1) 
        else:
            for x in results.multi_handedness:  # Used multi_handedness to detect left/right hand 
                label = x.classification[0].label 
                if label == 'Left':
                    cv2.putText(frame, 'Left Hand', (20, 50), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 0, 0), 1)
                if label == 'Right':
                    cv2.putText(frame, 'Right Hand', (460, 50), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255), 1)


    cv2.imshow('Hand Detection', frame) #display the video ( frame ) with the result

    #Ask to show the landmarker . Wait few moments after pressing q to press any key(q/t)

    if cv2.waitKey(1) & 0xff == ord('q'): #Pressing the q key will end the programme 
        break
    elif cv2.waitKey(1) & 0xFF== ord('t'): #multiple press may need to be done . Below code can solve this problem 
        show_lines = not show_lines
    
    ## instead of using the above , if and elif line (at end part ), we can use the below code to make it more responsive while pressing key .
    """ 
    pressedKey = cv2.waitKey(1) & 0xFF
    if pressedKey == ord('q'):break
    elif pressedKey == ord('t'):show_lines = not show_lines 
    
    """