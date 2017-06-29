import numpy as np
import cv2
# import aprifile

istruzioni = "Premi Esc per uscire"

cap = cv2.VideoCapture(0)

ret,frame = cap.read()


r,h,c,w = 250,90,400,125
track_window = (c,r,w,h)

roi = frame[r:r+h, c:c+w]
hsv_roi =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

nomefile = "haarcascade_frontalface_default.xml" # Name of the xml file
percorso = "/home/matteo/opencv/opencv-2.4.9/data/haarcascades"+nomefile # Path

# aprifile.Inserisci_nome_file(nomefile) # Verifica l'esistenza del file


print(istruzioni)
while(1):
    ret ,frame = cap.read()
    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        fc = cv2.CascadeClassifier(percorso)
        
        # f = fc.detectMultiScale(frame, 1.3, 5, (30,30), cv2.cv.CV_HAAR_SCALE_IMAGE)
        f = fc.detectMultiScale(frame, 1.3, 5)

        x,y,w,h = track_window
        cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)


        cv2.imshow('Rilevamento', frame)

        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        """
        else:
            cv2.imwrite(chr(k)+".jpg",frame)
        """

    else:
        break

cv2.destroyAllWindows()
cap.release()
