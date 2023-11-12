import cv2
import mediapipe as mp
import numpy as np

# mediapipe components

mp_drawing = mp.solutions.drawing_utils 
# makes it easier for us to render all the diff landmarks in our hand 
# landmarks means all those points marked

mp_hands = mp.solutions.hands 
# makes it easier to work with hands

# returns the text that we want to render for the particular hand
# and coordinates of where you are going to render it
def get_label(index, hand, results):
    # index = index no. of the detection(hand) we are working with
    # hand represents hand landmarks
    # results = results from multi handedness and their landmarks
    # (all detections from the model)
    
    output = None # final variable that we push out as a result of this func
    for idx, classification in enumerate(results.multi_handedness):
        if classification.classification[0].index == index:
            
            # Process results
            label = classification.classification[0].label
            score = classification.classification[0].score
            text = '{} {}'.format(label, round(score, 2))
            
            # Extract Coordinates
            coords = tuple(np.multiply(
                np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x, 
                          hand.landmark[mp_hands.HandLandmark.WRIST].y)),
                [640,480]).astype(int))
            
            output = text, coords
            
    return output

cap=cv2.VideoCapture(0)
#TODO: Add required mediapipe functions

# max num hands
with mp_hands.Hands(min_detection_confidence = 0.8, min_tracking_confidence = 0.5) as hands:
    # detection confidence: threshold for initial detection to be successful (80% left atleast)
    # detection tracking: threshold for tracking after initial detection
    
    # reading through every frame in the video capture
    while cap.isOpened():# while video capture is on

        # frame represents the image from our webcam
        ret, frame = cap.read()
        
        #Detections
        
        # recolouring frame
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # flip on horizontal
        image = cv2.flip(image, 1)
        
        # set flag
        image.flags.writeable = False
        
        # Actual detections
        results = hands.process(image)
        
        # lets us render/write on the image
        image.flags.writeable = True
        # recolouring
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # printing detections
        print(results)
        
        # allows us to draw our landmarks to our image
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                         )
                
                # render detection (left or right)
                if get_label(num, hand, results):
                    text, coord = get_label(num, hand, results)
                    cv2.putText(image, text, coord, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        
        # save image
        # cv2.imwrite(os.path.join("output Images", "{}.jpg".format(uuid.uuid1())), image)

        # then we 'render' that image to screen
        cv2.imshow('Hand Tracking', image) 
        # Hand Tracking is the name of our frame

        # everything below is closing down your window
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
cap.release()
cv2.destroyAllWindows()
