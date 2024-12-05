# MachineLearning Project
기계학습과응용 MAT3123 프로젝트입니다. 

첫 번째 프로젝트 : 눈 주변 감지 모델 -> 시선 tracking 모델
Motivation : 불과 몇 년 전만 해도 핸드폰 잠금해제 기능은 패턴 입력, 숫자 입력 등으로 간단한 논리였습니다. 그러나 요즘 핸드폰에는, 얼굴 인식 기능이 탑재되어 얼굴을 인식해서 눈을 마주치고 핸드폰 주인의 얼굴일 경우 Unlock이 되는 잠금 해제 기법을 많이 쓰고 있습니다. 이는 특히 코로나19 이후로 발달이 많이 되었는데, 코로나 초기에는 마스크를 쓰고 있으면 얼굴 인식이 제대로 되지 않아 잠금 해제가 어려웠습니다. 그러나 기술이 발전하여 마스크를 쓰고 있어도 시선을 track하여 잠금 해제하는 방식이 쓰이게 되었습니다. 저는 코로나19 때 이러한 기술의 발전이 삶을 매우 편리하게 해준다는 것을 상기하여 시선을 track하는 모델을 만들어보고 싶어졌습니다. 그러나 시선을 track하는 모델을 바로 만들기 어려워서, 눈 주변을 먼저 감지하는 모델을 만들어보고, 시선을 track하는 모델을 구현했습니다.

_눈 주변 감지 모델_
우선 이미징 프로세싱의 핵심인 OpenCV, numpy를 import한다
```
import numpy as np
import cv2
```
다음, 얼굴 감지를 위해 학습된 데이터를 OpenCV의 ```CascadeClassifier``` 메서드를 통해 받아온다.
```
image = cv2.imread('trainingset/Face1.jpg')
cv2.putText(image, 'ML_project', (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
cv2.imshow('Original image', image)
cv2.waitKey(1000)
```
Trainingset 파일 중 하나의 이미지 파일을 불러와서 이미지 원본을 1초 동안 화면에 띄워준다.
```
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```
이 줄은 컬러였던 이미지를 그레이스케일로 변환한다. 이는 이미지 처리의 핵심이다. 원본 컬러 이미지는 RGB 채널로 구성되며, 각 픽셀이 세 개의 값을 가지는 데에 반해, 그레이스케일로 변환하면 각 픽셀이 단일 밝기 값을 가지게 된다. 이는 데이터의 차원이 줄어들게 하여, computation이 매우 빨라지게 된다. 이 모델에서 쓰이는 객체 검출 알고리즘에도 그레이스케일 변환은 필수이다. 또한, 원본 이미지의 밝기 조건, 조명 조건, 등의 영향으로부터 자유로워 져, 패턴 인식이 더 안정적이다.
```
faces = face_cascade.detectMultiScale(image, 1.3, 5)

#iteration through the faces array and draw a rectangle
for(x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
    roi_gray = gray_image[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]
```
이는 얼굴을 인식하여 얼굴 주변을 네모난 박스로 감싸는, 즉 모델이 얼굴을 인식했는지 알 수 있는 코드이다.
이를 실행하면 다음과 같은 결과를 얻는다.
사진 넣기      !@^&*!@^$*&!@^$*&!^@$*&!@^$*&@^!$&*!@$
```
#identify the eyes and mouth using haar-based classifiers
eyes = eye_cascade.detectMultiScale(gray_image, 1.3, 5)

#iteration through the eyes and mouth array and draw a rectangl
for(ex, ey, ew, eh) in eyes:
    cv2.rectangle(image,(ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
```
마찬가지로, 눈을 인식하여 눈 주변에 박스로 감싼다. 그러나 학습된 모델이더라도 입 쪽을 눈으로 착각하는 경우가 있다. 다음의 사진이 그렇다.
사진 넣기

이를 해결하는 방법으로는 더 많은 training set으로 해결하는 방법이 있지만, 시간을 줄일 수 있는 좋은 방법이 있다. 바로 인식된 얼굴의 height의 절반 이하에 눈이 (보통 사람이라면) 있지 않음을 이용하여(즉 height로 제한을 둬, 입 쪽을 눈으로 인식할 수 없게끔 만듦), computation을 매우 아끼면서 해결할 수 있다. 이는 시선 track 모델에서 구현하였다.
```
cv2.imshow('face, eyes and mouth detected image', image)
cv2.waitKey()
```
박스로 감싸진 결과를 확인할 수 있다. waitKey()는 어떤 키를 누르기 전까지 결과가 나온다는 것이다.
training set은 pinterest를 통해 직접 다운 받은 이미지들의 set이다.
결과 첨부하기 *!(&@(&%(!@%&!@%&(

비록 간단해 보일 수 있지만, OpenCV를 이해하는데 매우 큰 도움이 된 구현이었다. 이를 기반으로 최종적인 목적인 시선 track 모델을 구현하였다.

_시선 tracking 모델_
```
detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
detector = cv2.SimpleBlobDetector_create(detector_params)
```
detector는 Blob 감지기이며, Blob의 최대 크기는 1500픽셀이다. 블롭이란, Binary Large Object의 약칭이며, 블롭 감지란 형태, 물체, 점 등 특정 영역을 찾는다는 것이다.












