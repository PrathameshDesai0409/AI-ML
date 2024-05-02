import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
import os

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize hand detector and classifier
detector = HandDetector(maxHands=1)
classifier = Classifier(r"C:\Users\prathamesh\OneDrive\Desktop\Model\keras_model.h5" , r"C:\Users\prathamesh\OneDrive\Desktop\Model\labels.txt")

# Other parameters
offset = 20
imgSize = 300
counter = 0

# Labels for gestures
labels = ["Hello","I Love You","No","Thank you","Yes"]

# Initialize counters for true positives and true negatives
true_positives = 0
true_negatives = 0

# Ground truth label for comparison (replace with actual labels from your dataset)
ground_truth_label = "Hello"

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Ensure that the bounding box dimensions are valid
        if w > 0 and h > 0:
            imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]
            imgCropShape = imgCrop.shape

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize-wCal)/2)
                imgWhite[:, wGap: wCal + wGap] = imgResize
                prediction , index = classifier.getPrediction(imgWhite, draw= False)
                print(prediction, index)

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap: hCal + hGap, :] = imgResize
                prediction , index = classifier.getPrediction(imgWhite, draw= False)

            # Compare predicted label with ground truth label
            if ground_truth_label == labels[index]:
                if ground_truth_label == "Positive":  # Adjust based on your classification
                    true_positives += 1
                else:
                    true_negatives += 1

            cv2.rectangle(imgOutput,(x-offset,y-offset-70),(x -offset+400, y - offset+60-50),(0,255,0),cv2.FILLED)  

            cv2.putText(imgOutput,labels[index],(x,y-30),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,0),2) 
            cv2.rectangle(imgOutput,(x-offset,y-offset),(x + w + offset, y+h + offset),(0,255,0),2)   

            cv2.imshow('ImageCrop', imgCrop)
            cv2.imshow('ImageWhite', imgWhite)

    else:
        print("Invalid hand bounding box dimensions")

    # Display image
    cv2.imshow('Image', imgOutput)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Print true positives and true negatives
print("True Positives:", true_positives)
print("True Negatives:", true_negatives)

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
