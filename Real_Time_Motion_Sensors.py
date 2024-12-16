import cv2
import numpy as np
from playsound import playsound # type: ignore

# Initialize video capture (use the webcame or video file)
cam = cv2.VideoCapture(0)

# Initialize the first frame for motion detection 
ret,first_frame = cam.read()
first_frame = cv2.cvtColor(first_frame,cv2.COLOR_BGR2GRAY)
first_frame = cv2.GaussianBlur(first_frame,(21,21),0)

while True:
    ret,frame = cam.read()
    if not ret:
        break

    # Convert the frame to grayscale and apply Gaussian blur
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame,(21,21),0)

    #Compute the absolute difference between current frame and first frame
    delta_frame = cv2.absdiff(first_frame,gray_frame)

    #Threshold the delta image to identify regions with significant change
    thresh_frame = cv2.threshold(delta_frame,30,255,cv2.THRESH_BINARY)[1]

    #Dilate the thresholded image to fill in gaps
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    #Calculate the number of white pixels in the threshold frame
    white_pixel_count = np.sum(thresh_frame == 255)

    #Trigger an alert if significant motion is detected 
    if white_pixel_count > 1000:
        playsound('alert_sound.mp3')

    #Display the original frame
    cv2.imshow('Motion Detection',frame)

    #Press 'ESC' to quit the program
    if cv2.waitKey(1) == 27: #27 is ASCII key for 'ESC' 
        break

cam.release()
cv2.destroyAllWindows()
