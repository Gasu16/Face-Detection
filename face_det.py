# Now using python 3.7
# Last update: 22 September 2018 - 00:50
import numpy as np
import cv2
# import aprifile
print(cv2.__version__)
istruzioni = "Premi Esc per uscire"

cap = cv2.VideoCapture(0) # Activate webcam

# Read and catch frames;
# ret -> a boolean variable where true/false value is stored, if it's true than the frame has get read correctly
ret,frame = cap.read()

# setup initial location of window
r,h,c,w = 250,90,400,125
track_window = (c,r,w,h)

# set up the ROI for tracking
roi = frame[r:r+h, c:c+w]
hsv_roi =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Convert color from RGB to HSV
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.))) # Do a specific operation on all values inside a specific range
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180]) # Calculate the histogram of a set of array the parameters are:
""" 
frame -> video source
channel -> list of channels used in order to compute the histogram
mask -> a 8 bit array
histsize -> size of the histogram
range -> array with 2 variables, one lower (0) and one higher (180) they rapresent the range of color
"""
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

nomefile = "haarcascade_frontalface_default.xml" # Name of the xml file
percorso = "/home/matteo/opencv-3.4.3/data/haarcascades"+nomefile # Path

# aprifile.Inserisci_nome_file(nomefile) # Checks if file exists


print(istruzioni)
while(1):
    """
    Calculate the histogram projection
    For every position (x,y) store the values on selected channels 
    and find the corresponding binary histogram
    """
    ret ,frame = cap.read()
    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        fc = cv2.CascadeClassifier(percorso) # Load the XML file
        fc.load('/home/matteo/opencv-3.4.3/data/haarcascades/haarcascade_frontalface_default.xml')
        # f = fc.detectMultiScale(frame, 1.3, 5, (30,30), cv2.cv.CV_HAAR_SCALE_IMAGE)
        f = fc.detectMultiScale(frame, 1.3, 5) # Do the detection operations

        x,y,w,h = track_window
        cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2) # Draw a rectangle


        cv2.imshow('Rilevamento', frame) # Show output

        k = cv2.waitKey(60) & 0xff # Wait key pression
        if k == 27: # If you press ESC then close and exit the program
            break
        """
        else:
            cv2.imwrite(chr(k)+".jpg",frame)
        """

    else:
        break

cv2.destroyAllWindows() # Close all the windows
cap.release() # Exit
raise SystemExit