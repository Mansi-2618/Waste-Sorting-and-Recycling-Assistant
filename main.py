import os
import cvzone
from cvzone.ClassificationModule import Classifier
import cv2

cap = cv2.VideoCapture(0)  # Initialize video capture
classifier = Classifier('Resources/keras_model.h5', 'Resources/labels.txt')


while True:
        _, img = cap.read()  # Capture frame-by-frame
       # Break the loop if no frame is captured

        #imgResize = cv2.resize(img, (454,340 ))
        #imgBg = cv2.imread(r"Resources\Image.jpg")
        prediction = classifier.getPrediction(img)
        print(prediction)  # Print prediction result

        #if imgBg is None:
                #break

        #imgBg[148:148+340,159:159+454] = imgResize


        # Displays

        cv2.imshow("Image", img)
        #cv2.imshow("Output", imgBg)
        cv2.waitKey(1)

