# Importing libraries
import cv2
import mediapipe as mp

# Adding mediapipe functions
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       model_complexity=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Defining OpenCV Webcam dimensions
VIDEO_X, VIDEO_Y = 640, 480

# Function to get the details of the hand if its LEFT/ RIGHT
def get_label(index, hand, results):
    # Making a list of outputs
    op = [False, 0, 0]

    for ind, some_hand in enumerate(results.multi_handedness):
        # Check if the hand is same
        if ind != index:
            continue

        # Process results
        label = some_hand.classification[0].label
        score = some_hand.classification[0].score
        text = f'{label} {round(score, 2)}'

        # Finding coordinates
        wrist_num = mp_hands.HandLandmark.WRIST     # For wrist, it is 0
        coords = tuple((int(hand.landmark[wrist_num].x * VIDEO_X), int(hand.landmark[wrist_num].y * VIDEO_Y)))

        # Updating outputs
        op[0] = True
        op[1], op[2] = text, coords

        # Breaking out of loop as we got the coordinates of the correct hand
        break

    return op


# Starting video capture
cap=cv2.VideoCapture(0)
while True:
    # Reading each frame of the video
    success, frame=cap.read()
    if not success:
        print('Ignoring empty camera frame')
        break

    # Marking the image as non-writable to improve performance
    frame.flags.writeable = False

    # Converting image from BGR to RBG
    image = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 1)
    frame.flags.writeable = True

    # Finding hand details if any
    results = hands.process(image)

    # We draw the landmarks and process if we find a hand
    if results.multi_hand_landmarks:
        
        for index, hand in enumerate(results.multi_hand_landmarks):
            # Highlighting the joints of the hand in the image
            mp_draw.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS)

            # Getting the Left/Right label of the index'th hand in the frame
            hand_label = get_label(index, hand, results)
            
            # If we found the coordinates (i.e. if value is True)
            if hand_label[0]:
                text, coords = hand_label[1], hand_label[2]
                
                # Overlapping the text 'Left' or 'Right' with the score
                cv2.putText(image, text, coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Showing the results on the screen
    cv2.imshow("Hand Detection", image)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

# Release the hardware/ software resources or any pointers
cap.release()