# MachineLearning Project
기계학습과응용 MAT3123 프로젝트입니다. 
# 수정점 : 코드 전반적으로 수정, my_eye는 84line part 수정하기, 결과 첨부하기
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
```
cap = cv2.VideoCapture(0)
cv2.namedWindow('image')
cv2.createTrackbar('threshold', 'image', 0, 255, lambda x: None)
```
컴퓨터의 내장된 카메라로 영상을 캡처하기 시작한다. 이 ```threshold```는 이미지를 이진화하는데에 쓰이는 값이며, 픽셀 값이 ```threshold```보다 크면 흰색(255), 작으면 검정색(0)으로 설정된다. 영상이 찍히는 곳 주변이 밝을 경우와 어두울 경우에 threshold를 바꿔가며 track이 잘 되는 최적점을 찾아낸다.
```
    # 얼굴 검출
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    coords = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    if len(coords) == 1:  # 얼굴이 하나만 감지된 경우
        x, y, w, h = coords[0]  # 좌표 추출
        face_frame = frame[y:y + h, x:x + w]  # 얼굴 영역 잘라냄
    else:  # 얼굴이 감지되지 않거나 여러 개일 경우
        face_frame = None
        cv2.imshow('image', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
```
이전의 눈 주변 감지 모델에서처럼, 그레이스케일로 변환하여, ```detectMultiScale``` 함수를 통해 여러 크기의 객체를 탐지한다. 이때 반환되는 값은 얼굴, 눈의 사각형 좌표이다. (x,y,w,h)를 반환 받으며, (x,y)는 객체의 왼쪽 위 모서리 좌표, w은 너비, h는 높이이다. 이 모델은 사람 한 명의 눈을 track하기 때문에, 얼굴이 하나만 감지된 경우를 다룬다. 물론 인식되는 얼굴이 여러개면 전부 다 처리해주면 되나, 핸드폰의 잠금해제와 같은 경우에는 두 명 이상의 얼굴을 detect하는 일이 드물 것이다. 이로써 모델이 영상 속에서 얼굴을 인식할 수 있게 되었다.
```
    if face_frame is not None:
        # 눈 검출
        gray_face = cv2.cvtColor(face_frame, cv2.COLOR_BGR2GRAY)
        eyes_coords = eye_cascade.detectMultiScale(gray_face, 1.3, 5)
        width = np.size(face_frame, 1)
        height = np.size(face_frame, 0)
        left_eye = None
        right_eye = None
        for (ex, ey, ew, eh) in eyes_coords:
            if ey > height / 2:
                continue
            eyecenter = ex + ew / 2
            if eyecenter < width * 0.5:
                left_eye = face_frame[ey:ey + eh, ex:ex + ew]
            else:
                right_eye = face_frame[ey:ey + eh, ex:ex + ew]
        eyes = [left_eye, right_eye]
```
드디어 눈을 검출하는 부분이다. 마찬가지로 그레이스케일로 변환 이후, 눈의 좌표를 받아서 왼쪽 오른쪽 눈을 구분한다.
```
        for eye in eyes:
            if eye is not None:
                threshold = cv2.getTrackbarPos('threshold', 'image')
                # 블롭 처리
                gray_eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
                _, img = cv2.threshold(gray_eye, threshold, 255, cv2.THRESH_BINARY)
                img = cv2.erode(img, None, iterations=5)
                img = cv2.dilate(img, None, iterations=5)
                img = cv2.medianBlur(img, 5)
                keypoints = detector.detect(img)
                print(keypoints)

                eye = cv2.drawKeypoints(eye, keypoints, eye, (0, 0, 255),
                                        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

```
위에서 받아온 왼쪽, 오른쪽 눈의 위치 정보를 바탕으로 시선을 track하는 빨간색 원을 영상에 입력(print)한다. 이때 erode 5번(불필요한 픽셀 제거), dilate 5번(이미지의 밝은 영역 확장) medianBlur(커널 크기 5를 사용하여 작은 점 노이즈를 제거)을 통해 모델이 눈을 더 잘 인식하게 한다. (여기 조금 수정해야할듯).
이를 통해 눈동자(블롭)을 감지할 수 있다.
마지막으로, 아무 키를 눌러서 quit하게 만드는 코드를 추가하여 시선 tracking을 끝마칠 수 있다.
```
    if cv2.waitKey(1) != -1:  # 아무 키나 눌렀을 때
        break
```





