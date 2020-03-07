import cv2
import numpy as np

'''
Python Program to find the correct hsv filter value for object detection.
Modified code used in: ProgrammingKnowledge YT - https://www.youtube.com/watch?v=3D7O_kZi8-o.
'''

def nothing(x):
    pass

# Default variable to get out image
pic = '/home/blank/Data/FaceDirection/0/309.jpg'

# Use webcam
cap = cv2.VideoCapture(0)

# Create a trackbar to easily find value for object we want to track
cv2.namedWindow("Tracking")
cv2.createTrackbar("LH", "Tracking", 9, 255, nothing)
cv2.createTrackbar("LS", "Tracking", 18, 255, nothing)
cv2.createTrackbar("LV", "Tracking", 116, 255, nothing)
cv2.createTrackbar("UH", "Tracking", 36, 255, nothing)
cv2.createTrackbar("US", "Tracking", 255, 255, nothing)
cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)

# Set threshold trackbar
cv2.createTrackbar("LT", "Tracking", 0, 255, nothing)
cv2.createTrackbar("UT", "Tracking", 255, 255, nothing)

while True:
    # For images
    image = cv2.imread(pic)

    # Capture from webcam
    #_, image = cap.read()

    blur = cv2.blur(image, (7, 7))

    # Convert to HSV
    #hsv = cv2.cvtColor( blur, cv2.COLOR_BGR2HSV )
    hsv = cv2.cvtColor( image, cv2.COLOR_BGR2HSV )


    lower_bound_hue = cv2.getTrackbarPos('LH', 'Tracking')
    lower_bound_sat = cv2.getTrackbarPos('LS', 'Tracking')
    lower_bound_val = cv2.getTrackbarPos('LV', 'Tracking')

    upper_bound_hue = cv2.getTrackbarPos('UH', 'Tracking')
    upper_bound_sat = cv2.getTrackbarPos('US', 'Tracking')
    upper_bound_val = cv2.getTrackbarPos('UV', 'Tracking')

    lower_bound = np.array([lower_bound_hue, lower_bound_sat, lower_bound_val])
    upper_bound = np.array([upper_bound_hue, upper_bound_sat, upper_bound_val])


    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # To reduce noise
    #kernel = np.ones((5, 5), np.uint8)
    #mask = cv2.dilate(mask, kernel, iterations=5)
    #mask = cv2.erode(mask, kernel, iterations=5)
    
    cv2.imshow("Mask", mask)

    #result = cv2.bitwise_and(image, image, mask=mask)

    #__, __, gray_result = cv2.split(result)

    #cv2.imshow("Gray result", gray_result)
    
    #lower_bound_thresh = cv2.getTrackbarPos('LT', 'Tracking')
    #upper_bound_thresh = cv2.getTrackbarPos('UT', 'Tracking')

    #ret, thresh = cv2.threshold(gray_result, lower_bound_thresh, upper_bound_thresh, cv2.THRESH_TOZERO)

    #thresh = cv2.blur(thresh, (7, 7))

    #cv2.imshow("Threshold", thresh)

    key = cv2.waitKey(1)    
    if key == 27:
        break

cv2.destroyAllWindows()
cap.release()
