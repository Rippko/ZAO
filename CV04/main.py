import sys
import cv2
import numpy as np
import math
import struct
from datetime import datetime
import glob

WHITE = (255, 255, 255)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)

def detect_eye_status(frame, eye_classifier):
    eyes = eye_classifier.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=17, minSize=(6, 6))
    
    for (ex, ey, ew, eh) in eyes:
        eye_region = frame[ey+45:ey+eh, ex+10:ex+ew]
        gray_eye_region = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        eye_center = (ex + ew // 2, ey + eh // 2)

        gray_eye_region = cv2.GaussianBlur(gray_eye_region, (3, 3), 0)
        
        circles = cv2.HoughCircles(gray_eye_region, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
                                   param1=35, param2=18, minRadius=10, maxRadius=ew//2)
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                center = (circle[0], circle[1])
                radius = circle[2]
                cv2.circle(frame, eye_center, ew // 3, GREEN, 3)
        else:
            cv2.circle(frame, eye_center, ew // 3, RED, 3)

def detect_mouth(frame, mouth_classifier):
    mouths = mouth_classifier.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=45, minSize=(20, 20))
    for mouth in mouths:
        cv2.rectangle(frame, mouth, YELLOW, 10)
        cv2.rectangle(frame, mouth, WHITE, 4)
    

def detect_faces(frame, front_classifier, profile_classifier, eye_classifier, mouth_classifier):
    front_faces, ints = front_classifier.detectMultiScale2(frame, scaleFactor=1.2, minNeighbors=6, minSize=(2, 2))
    profile_faces, p_ints = profile_classifier.detectMultiScale2(frame, scaleFactor=1.1, minNeighbors=12, minSize=(2, 2))

    for face in front_faces:
        detect_eye_status(frame, eye_classifier)
        detect_mouth(frame, mouth_classifier)
        cv2.rectangle(frame, face, GREEN, 10)
        cv2.rectangle(frame, face, WHITE, 4)
    for profile in profile_faces:
        cv2.rectangle(frame, profile, RED, 10)
        cv2.rectangle(frame, profile, WHITE, 4)

    return frame

def main():
    camera = cv2.VideoCapture(0)
    cv2.namedWindow('win_name', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('win_name', 640, 480)

    video = cv2.VideoCapture('fusek_face_car_01.avi')

    front_classifier = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
    profile_classifier = cv2.CascadeClassifier('./haarcascades/haarcascade_profileface.xml')
    eyes_classifier = cv2.CascadeClassifier('./haarcascades/eye_cascade_fusek.xml')
    mouth_classifier = cv2.CascadeClassifier('./haarcascades/haarcascade_smile.xml')

    while True:
        # ret, frame = camera.read()
        # if not ret:
        #     break
        ret, frame = video.read()
        if not ret:
            break

        frame = detect_faces(frame, front_classifier, profile_classifier, eyes_classifier, mouth_classifier)

        cv2.imshow('win_name', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()