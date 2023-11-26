import cv2
import mediapipe as mp
import numpy as np
import time
# Initialize webcam
from google.protobuf.json_format import MessageToDict 
cap=cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,640)

# Initialize hand tracking
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
#initialising puck and velocity
puckx = 640
pucky = 320
dx = 10
dy = 10
# Initialize timer variables
start_time = time.time()
game_duration = 30  # 1/2 minute in seconds

#overlay function
def overlay_image(background, foreground, position):
    y1, x1 = position
    h1, w1, _ = foreground.shape

    alpha = foreground[:, :, 3] / 255.0
    for c in range(0, 3):
        background[y1:y1 + h1, x1:x1 + w1, c] = (1 - alpha) * background[y1:y1 + h1, x1:x1 + w1, c] + alpha * foreground[:, :, c] * alpha

    return background
#distance between center of ball and donut
def calculate_distance(coord1, coord2):
    return np.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

    
# Load target image and resize it to 30,30
#4
target = cv2.imread("target.png",cv2.IMREAD_UNCHANGED)
target = cv2.resize(target,(30,30))

# Initialize 5 target positions randomly(remember assignment 2!!)
#5

ret, frame = cap.read()
height, width, _ = frame.shape
num_coordinates = 5
random_coordinates = [(np.random.randint(0, height - target.shape[0]), np.random.randint(0, width - target.shape[1])) for _ in range(num_coordinates)]
proximity_threshold = 31.5 # distance from centre of donut to puck, so 30 + 5% of 30

with mp_hands.Hands(min_detection_confidence=0.65,min_tracking_confidence=0.35, max_num_hands =1) as hands:
    while True:
    # Calculate remaining time and elapsed time in minutes and seconds   
    #9
        time_remaining= game_duration-int((time.time()-start_time))
    # Read a frame from the webcam
    #10
        success, frame=cap.read()
    # Convert the BGR image to RGB
    #11
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

    # Flip the frame horizontally for a later selfie-view display
    #12
        image = cv2.flip(image,1)
    # Process the frame with mediapipe hands
    #13
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        ret, frame = cap.read()

        for coord in random_coordinates:
            image = overlay_image(image,target, coord) 


# Initialize paddle and puck positions
        if results.multi_hand_landmarks:
            #for num, hand in enumerate(results.multi_hand_landmarks):
                #mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS)
            for handLMS in results.multi_hand_landmarks:
                for id , lm in enumerate(handLMS.landmark):
                    h , w , c = image.shape 
                    if id == 8 : # this is the finger tip (index)
                        cx, cy = int(lm.x * w), int(lm.y * h)# Update paddle position based on index finger tip
                        cv2.rectangle(image , (cx-50,cy+15),(cx+50,cy-15) , (110,100,100), -1 )
                        # Check for collisions with the paddle
                        #17   
                        if puckx + 10 in range (cx-50,cx+50) and pucky + 10 in range (cy-15,cy):
                            dy = -dy
                        if puckx - 10 in range (cx-50,cx+50) and pucky - 10 in range (cy,cy+15):
                            dy = -dy
                        if pucky + 10 in range (cy-14,cy+14) and puckx + 10 in range (cx-50,cx):
                            dx = -dx
                        if pucky - 10 in range (cy-14,cy+14) and puckx - 10 in range (cx,cx+50):
                            dx = -dx
                    for coord in random_coordinates:
                        frame = overlay_image(frame, target, coord)                         
                        distance = calculate_distance((coord[1] + target.shape[1] // 2, coord[0] + target.shape[0] // 2), (puckx,pucky))
                        if time_remaining>=0:
                            if distance < proximity_threshold:
                                random_coordinates.remove(coord)
                                print(random_coordinates)
                                dx= int(1.2*dx)
                                dy= int(1.2*dy)               
        for coord in random_coordinates:
                                frame = overlay_image(frame, target, coord)                         
                                distance = calculate_distance((coord[1] + target.shape[1] // 2, coord[0] + target.shape[0] // 2), (puckx,pucky))
                                if time_remaining>=0:
                                    if distance < proximity_threshold:
                                        random_coordinates.remove(coord)#basically removing the coords of the donut hit, so that it doesnt appear in the next frame
                                        print(random_coordinates)
                                        dx= int(1.2*dx)
                                        dy= int(1.2*dy)                                
# Update puck position based on its velocity
        puckx += dx
        pucky += dy
        cv2.circle(image,(puckx,pucky),10,(255,0,0),-1)
# Check for collisions with the walls
        if puckx + 10 >= 1280:
            dx = -dx
        if puckx - 10 <= 0:
            dx = -dx
        if pucky - 10 >= 640:
            dy = -dy
        if pucky + 10 <= 0:
            dy = -dy
    # Display the player's score on the frame
    #20

    # Display the remaining time on the frame
    #21
        if time_remaining >=0:
            cv2.putText(image, 'Time left: ', (250, 50), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255, 255, 0), 2)
            cv2.putText(image, str(time_remaining), (407, 50), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255, 255, 0), 2)
            score=5-len(random_coordinates)
            cv2.putText(image,"Score: ",(550,50), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(image,str(score),(650,50), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2)
    # Check if all targets are hit or time is up    
    #22
        if time_remaining<=0 and score!=5:
            cv2.putText(image,"GAME OVER",(550, 200), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 0, 255), 2,cv2.LINE_AA)
            cv2.putText(image,"Score: ",(550,280), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 0, 255), 2,cv2.LINE_AA)
            cv2.putText(image,str(score),(650, 280), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 0,255), 2,cv2.LINE_AA)
            cv2.putText(image,"Press q to exit",(550, 360), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 0, 255), 2,cv2.LINE_AA)
        if score ==5:
            cv2.putText(image,"YOU WIN!!!!!!!",(550, 200), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2,cv2.LINE_AA)
            cv2.putText(image,"Press q to exit",(550, 300), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2,cv2.LINE_AA)

    

    # Display the resulting frame
        
        cv2.imshow("Hand Detection",image)

    # Exit the game when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
