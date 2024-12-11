'''
기계학습과 응용 프로젝트
2020131013 정준혁
목적 : 얼굴과 눈을 탐지하는 모델
'''
import numpy as np
import cv2

#load the xml files for face, eye and mouth detection into the program
#trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

image = cv2.imread('/Users/joonmac/Downloads/기학응/test3.jpg')
cv2.putText(image, 'ML_project', (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
cv2.imshow('Original image', image)
cv2.waitKey(1000)

# converting하는 과정
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 얼굴
faces = face_cascade.detectMultiScale(image, 1.3, 5)

# 얼굴을 인식해서 직사각형을 만든다
for(x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 0), 2)
    roi_gray = gray_image[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]

# 눈
eyes = eye_cascade.detectMultiScale(gray_image, 1.3, 5)

# 눈 주변으로 직사각형을 만듦
for(ex, ey, ew, eh) in eyes:
    cv2.rectangle(image,(ex, ey), (ex+ew, ey+eh), (255, 255, 0), 2)

#show the final image after detection
cv2.imshow('face, eyes and mouth detected image', image)
cv2.waitKey()

#show a successful message to the user
print("Face, eye and mouth detection is successful")
