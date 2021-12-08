import numpy as np
import cv2 as cv

# Device index of 0 is used.
video_cap_obj = cv.VideoCapture(0)

if not video_cap_obj.isOpened():
    print("Not able to open the camera")
    exit()

running = True

while running:

    # Get the next frame
    frameGood, frame = video_cap_obj.read()

    if not frameGood:
        print("Steam end.")
        break

    # Now we do operations on the frame.
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow('frame', gray_frame)

    # Make it such that if q is pressed the program will quit.
    if cv.waitKey(1) == ord('q'):
        break

# program done, release the capture
video_cap_obj.release()
cv.destroyAllWindows()
