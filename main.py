import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# This is for getting input via webcam
hands = mp_hands.Hands(
    min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)

while cap.isOpened(): #Or can use while True:
    success, frame = cap.read()
    if not success:
        print("Empty camera frame encountered, ignoring!.")
        continue

    frame = cv2.flip(frame, 1)
    ''' Fliping image as when user will lift his left hand then hands on the
        left side of screen will rise ,(this gives a mirror like feel) ,which
        is more comfortable as conpared to normal laptop camera'''

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    '''Converting BGR image to RGB as mediapipe needs image in the 
    form of RGB'''

    results = hands.process(frame_rgb)
    '''Processing the RGB image with the hands model'''

    '''For Drawing the hand landmarks, a bounding box and showing handedness'''
    handedness_labels = []
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):

            handedness_label = handedness.classification[0].label
            handedness_labels.append(handedness_label)

            #Drawing the hand landmarks
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            #Calculate the bounding box
            min_x, min_y, max_x, max_y = float('inf'), float('inf'), -float('inf'), -float('inf')
            for lmk in hand_landmarks.landmark:
                x, y = int(lmk.x * frame.shape[1]), int(lmk.y * frame.shape[0])
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)

            # Draw the bounding box
            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 0, 0), 2)
            '''This draws a rectangular box around the detected hands.'''

            # Write the handedness label on the rectangular box
            cv2.putText(frame, handedness_label, (min_x, min_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            '''I have used this to denote the handedness on the rectangular box
            ,I used the rectangular box's coordinates to get accurate location
            of where to place the handedness result.'''

    if len(set(handedness_labels)) == 1:
        # For writing the handedness label on the image
        cv2.putText(frame, handedness_labels[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow('ERC ASSIGNMENT-3', frame)
    '''Finally showing the processed image using open cv '''
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    '''This is to end the program ,for doing so just press the escape key.'''

hands.close()
cap.release()
cv2.destroyAllWindows()
''' So this was the project to detect handedness and showing it on the screen.
 In this, live video feed was used .Also, I drew a box around the hands.
 Moreover, I drew the hand landmarks on the output.'''
