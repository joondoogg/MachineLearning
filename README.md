# MachineLearning Project
기계학습과응용 MAT3123 프로젝트입니다. 
2020131013 수학과 정준혁
# 첫 번째 프로젝트 : 눈 주변 감지 모델 -> 시선 tracking 모델
Motivation : 불과 몇 년 전만 해도 핸드폰 잠금해제 기능은 패턴 입력, 숫자 입력 등으로 간단한 논리였습니다. 그러나 요즘 핸드폰에는, 얼굴 인식 기능이 탑재되어 얼굴을 인식해서 눈을 마주치고 핸드폰 주인의 얼굴일 경우 Unlock이 되는 잠금 해제 기법을 많이 쓰고 있습니다. 이는 특히 코로나19 이후로 발달이 많이 되었는데, 코로나 초기에는 마스크를 쓰고 있으면 얼굴 인식이 제대로 되지 않아 잠금 해제가 어려웠습니다. 그러나 기술이 발전하여 마스크를 쓰고 있어도 시선을 track하여 잠금 해제하는 방식이 쓰이게 되었습니다. 저는 코로나19 때 이러한 기술의 발전이 삶을 매우 편리하게 해준다는 것을 상기하여 시선을 track하는 모델을 만들어보고 싶어졌습니다. 그러나 시선을 track하는 모델을 바로 만들기 어려워서, 눈 주변을 먼저 감지하는 모델을 만들어보고, 시선을 track하는 모델을 구현했습니다.

_눈 주변 감지 모델_\\
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
이 줄은 컬러였던 이미지를 그레이스케일로 변환한다. 이는 이미지 처리의 핵심이다. 원본 컬러 이미지는 RGB 채널로 구성되며, 각 픽셀이 세 개의 값을 가지는 데에 반해, 그레이스케일로 변환하면 각 픽셀이 단일 밝기 값을 가지게 된다. 이는 데이터의 차원이 줄어들게 하여, computation이 매우 빨라지게 된다. 이 모델에서 쓰이는 객체 검출 알고리즘에도 그레이스케일 변환은 필수이다. 또한, 원본 이미지의 밝기 조건, 조명 조건, 등의 영향으로부터 자유로워져서 패턴 인식이 더 안정적이다.
```
# 얼굴
faces = face_cascade.detectMultiScale(image, 1.3, 5)

# 얼굴을 인식해서 직사각형을 만든다
for(x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 0), 2)
    roi_gray = gray_image[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]
```
이는 얼굴을 인식하여 얼굴 주변을 네모난 박스로 감싸는, 즉 모델이 얼굴을 인식했는지 알 수 있는 코드이다.
이를 실행하면 다음과 같은 결과를 얻는다.

<img width="958" alt="스크린샷 2024-12-11 오후 8 13 17" src="https://github.com/user-attachments/assets/6c19721e-2323-4c46-b70b-4cf68aeb59af">

```
# 눈
eyes = eye_cascade.detectMultiScale(gray_image, 1.3, 5)

#iteration through the eyes and mouth array and draw a rectangl
for(ex, ey, ew, eh) in eyes:
    cv2.rectangle(image,(ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
```
마찬가지로, 눈을 인식하여 눈 주변에 박스로 감싼다. 

<img width="812" alt="image" src="https://github.com/user-attachments/assets/0ff00f2d-2814-47c9-a5ba-84dc9d9ff1e1">

그러나 위 모델은 눈이 아닌 곳을 눈으로 착각하는 경우가 있다.

<img width="494" alt="image" src="https://github.com/user-attachments/assets/4f67cf99-57c7-4845-8c15-f318b1bfe17c">

<img width="579" alt="image" src="https://github.com/user-attachments/assets/1825688d-057d-4b36-b6c7-6ee8f1ba2d38">

이를 해결하는 방법으로는 더 많은 training set으로 해결하는 방법이 있지만, 시간을 줄일 수 있는 좋은 방법이 있다.
우선 눈은 face로 인식한 직사각형 안에 있어야하며, 인식된 얼굴의 height의 절반 이하에 눈이 (보통 사람이라면) 있지 않음을 이용하여 computation을 아끼고 문제를 해결할 수 있다. 이는 시선 track 모델에서 구현하였다.
```
cv2.imshow('face, eyes and mouth detected image', image)
cv2.waitKey()
```
박스로 감싸진 결과를 확인할 수 있다. waitKey()는 어떤 키를 누르기 전까지 결과가 나온다는 것이다.
training set은 pinterest를 통해 직접 다운 받은 이미지들의 set이다.
```https://kr.pinterest.com/search/pins/?q=face%20eye&rs=typed```
<img width="531" alt="image" src="https://github.com/user-attachments/assets/2d894cbc-5c56-45c0-a675-4471bb9587f3">

이는 OpenCV 라이브러리를 이해하는데 큰 도움이 된 구현이었다. 이를 기반으로 다음의 시선 track 모델을 구현하였다.

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
cv2.createTrackbar('threshold', 'image', 64, 255, lambda x: None)
```
컴퓨터의 내장된 카메라로 영상을 캡처하기 시작한다. 이 ```threshold```는 이미지를 이진화하는데에 쓰이는 값이며, 픽셀 값이 ```threshold```보다 크면 흰색(255), 작으면 검정색(0)으로 설정된다. 영상이 찍히는 곳 주변이 밝을 경우와 어두울 경우에 threshold를 바꿔가며 track이 잘 되는 최적점을 찾아낸다. 기본값은 64로 설정하였다.
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

https://github.com/user-attachments/assets/5e866e56-041d-402a-ad5d-8e93a53385e5

# 두 번째 프로젝트 : 주가 지수 예측 모델
Motivation : 퀀트 투자는 고급 수학 능력과 컴퓨터 알고리즘을 활용하여 주가 지수를 예측하고, 최선의 hedging전략을 펼치는 투자 전략입니다. 이는 투자 결정에서 인간의 감정, 직관 등을 배제하고 데이터를 기반으로 분석한다는 점에서 기존의 분석을 대체하였습니다. 머신러닝 알고리즘은 대규모 금융 데이터에서 패턴을 탐지하는데 강력한 도구가 될 수 있습니다. 저는 파이썬에서 어떤 라이브러리가 이러한 금융 데이터를 분석하는 머신 러닝을 구현할까? 라는 궁금증에서 출발해서 이 모델을 구현하게 되었습니다.

_주가예측 모델_
Yahoo finance를 사용하여, 총 네 개의 회사의 ticker를 googling하여 다음과 같이 딕셔너리를 만든다.
```
# 종목 딕셔너리
tickers = {
    'Samsung': '005930.KS',
    'Kakao': '035720.KS',
    'Naver': '035420.KS',
    'SKHynix': '000660.KS'
}
```
총 1년, 2년간의 1시간 마다의 주식 데이터를 불러온 값을 비교하여 각각을 시각화할 것입니다.
```
period = '2y'
interval = '1h'

models = {}  # model per ticker를 저장하기 위함
test_indices = None  # for alignment
```
```yf.download``` 를 통해 원하는 기간동안의 데이터를 다운 받을 수 있다.
```stock_datas = yf.download(ticker, interval=interval, period=period)```
또한 주식시장, 파생상품시장에서 분석을 할 때 기본적 도구로서 쓰이는 이동평균을 계산할 것이다.
분석의 비교를 용이하게 하기 위해, 5일, 10일, 50일 동안 종가의 이동평균을 구하였다.
```
    # 이동평균 계산
    stock_datas['moving_avgs_5'] = stock_datas['Close'].rolling(window=5).mean()
    stock_datas['moving_avgs_10'] = stock_datas['Close'].rolling(window=10).mean()
    stock_datas['moving_avgs_50'] = stock_datas['Close'].rolling(window=50).mean()
```
데이터의 전처리를 위해, NaN을 없애주고 ```stock_datas = stock_datas.dropna()```
feature와 target을 정의해준다. 
```
feat_set = stock_datas[['Close', 'moving_avgs_5', 'moving_avgs_10', 'moving_avgs_50']]
close_price = stock_datas['Close'].shift(-1).dropna()
```
```train_test_split```을 사용하여 train data, test data를 나눠준다.
```
    # train_test_split 사용
    # 70% : train set
    # 30% : test set
    X_train, X_test, y_train, y_test = train_test_split(
        feat_set, close_price, test_size=0.3, random_state=10, shuffle=False
    )
```
위는 전처리를 거친 data들중 70%는 train set, 30%는 test set으로 쓰겠다는 것이다.
Lasso 회귀를 사용하여 모델을 훈련한다.
```
    # 모델 초기화 및 훈련
    prediction_model = Lasso().fit(X_train, y_train)
    models[name] = prediction_model  # Store the trained model
    # 예측
    predictions = prediction_model.predict(X_test)
```
각각의 모델은 mae, mse, r2, rmse를 사용하여 평가하였다.
결국 내일의 주가를 예측해보기 위해서 이러한 모델이 있는 것이므로, 다음의 코드는 바로 다음날의 주가를 예측하여 전날 대비 주가가 상승했는지, 감소했는지, 변동이 없는지 예측한다.
```
    # 가장 최신의 특성을 사용하여 다음 날의 주가 예측
    latest_feat = stock_datas[['Close', 'moving_avgs_5', 'moving_avgs_10', 'moving_avgs_50']].iloc[-1].values.reshape(1, -1)
    predicted_next_close = prediction_model.predict(latest_feat)

    # 오늘의 실제 종가
    latest_close = stock_datas['Close'].iloc[-1]
    
    # 예측값과 실제값의 차이 계산
    difference = predicted_next_close[0] - latest_close

    # difference가 Series인지 확인하고 스칼라로 변환
    if isinstance(difference, (pd.Series, np.ndarray)):
        difference = difference.item()

    # 상승/감소/변동 없음 여부 판단
    if difference > 0:
        change_status = "상승"
    elif difference < 0:
        change_status = "감소"
    else:
        change_status = "변동 없음"

    # 결과 출력
    print(f'Predicted Next Closing Price for {name}: {predicted_next_close[0]:.2f}')
    print(f'Difference (Prediction - Today\'s Close): {difference:.2f} ({change_status})')
```
latest feature을 가져와서, 한 번의 prediction을 거친다. 오늘의 주가 지수로 상승, 감소, 변동 없음을 출력한다.
다음과 같이 plot하였다.
```
    # Plot: 실제 vs 예측
    plt.figure(figsize=(14, 7))
    plt.plot(y_test.index, y_test.values, label='Actual Price')
    plt.plot(y_test.index, predictions, label='Predicted Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'{name} - Actual vs. Predicted Stock Prices')
    plt.legend()
    plt.show()

    # Plot: 이동평균 + 종가
    plt.figure(figsize=(14, 7))
    plt.plot(stock_datas.index, stock_datas['Close'], label='Close Price')
    plt.plot(stock_datas.index, stock_datas['moving_avgs_5'], label='5-Period MA')
    plt.plot(stock_datas.index, stock_datas['moving_avgs_10'], label='10-Period MA')
    plt.plot(stock_datas.index, stock_datas['moving_avgs_50'], label='50-Period MA')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'{name} - Close Price Over Time')
    plt.legend()
    plt.show()
```
2 year analysis :

<img width="497" alt="image" src="https://github.com/user-attachments/assets/b292c4ea-2754-404f-940e-72831d9617e3">

![image](https://github.com/user-attachments/assets/239bbed1-6ef0-4c39-b5fb-90c1a190bd13)
![image](https://github.com/user-attachments/assets/72ef1e47-e338-4959-b97a-9a0a32a35c58)

<img width="494" alt="image" src="https://github.com/user-attachments/assets/1120e270-dbc0-45e8-876c-113dde6ed7ca">

![image](https://github.com/user-attachments/assets/18490fb1-009a-4bf5-aa4b-bfaec02c9091)
![image](https://github.com/user-attachments/assets/5282f0ed-1e91-43ad-8aa7-c2f09568d658)

<img width="520" alt="image" src="https://github.com/user-attachments/assets/230d5288-35a7-49cb-8d57-ffb07dbd09e7">

![image](https://github.com/user-attachments/assets/30eddf0c-0de2-4c83-9264-7810d473ad6b)
![image](https://github.com/user-attachments/assets/d8e9615e-2975-47c6-9581-59ee70ab7c4e)

<img width="496" alt="image" src="https://github.com/user-attachments/assets/af2d6a92-1bbe-4038-a44b-0126877653bf">

![image](https://github.com/user-attachments/assets/efb1bb9d-0ddc-4776-b8fd-9f1c32416797)
![image](https://github.com/user-attachments/assets/bab03775-a529-430b-8404-0b111eee62e8)

1 year analysis :

<img width="496" alt="image" src="https://github.com/user-attachments/assets/ff3d0a3f-9f92-4d3a-b2b1-46ee64cda497">

![image](https://github.com/user-attachments/assets/604ceed2-e64b-444c-a20f-34dcae13a639)

![image](https://github.com/user-attachments/assets/cb3f9a92-a35c-4fb9-894b-a9fcfb8e2c69)

<img width="501" alt="image" src="https://github.com/user-attachments/assets/75607cce-7430-4583-96f4-827f349a1bd3">

![image](https://github.com/user-attachments/assets/8e846b76-4f9e-44a6-bcde-b99b8527a54e)

![image](https://github.com/user-attachments/assets/29c128e9-5397-40d3-84fb-472ac6b0755c)

<img width="501" alt="image" src="https://github.com/user-attachments/assets/6831220c-52f5-48ba-85f9-26523adb68ed">

![image](https://github.com/user-attachments/assets/6306576a-6719-48fe-a2b2-ef40e62e6649)

![image](https://github.com/user-attachments/assets/2bb9817e-6b68-4270-8d3d-7242be94a791)

<img width="507" alt="image" src="https://github.com/user-attachments/assets/c70fe182-150f-4dbd-a23f-e310c25ac59c">

![image](https://github.com/user-attachments/assets/777ee5aa-cfcd-4727-a4b5-4157cfba5961)

![image](https://github.com/user-attachments/assets/823b2dd2-ac1d-4599-9311-126832328588)






