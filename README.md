# MachineLearning Project
기계학습과응용 MAT3123 프로젝트입니다. 
2020131013 수학과 정준혁
# 첫 번째 프로젝트 : 눈 주변 감지 모델 -> 시선 tracking 모델
Motivation : 불과 몇 년 전만 해도 핸드폰 잠금해제 기능은 패턴 입력, 숫자 입력 등으로 간단한 논리였습니다. 그러나 요즘 핸드폰에는, 얼굴 인식 기능이 탑재되어 얼굴을 인식해서 눈을 마주치고 핸드폰 주인의 얼굴일 경우 Unlock이 되는 잠금 해제 기법을 많이 쓰고 있습니다. 이는 특히 코로나19 이후로 발달이 많이 되었는데, 코로나 초기에는 마스크를 쓰고 있으면 얼굴 인식이 제대로 되지 않아 잠금 해제가 어려웠습니다. 그러나 기술이 발전하여 마스크를 쓰고 있어도 시선을 track하여 잠금 해제하는 방식이 쓰이게 되었습니다. 저는 코로나19 때 이러한 기술의 발전이 삶을 매우 편리하게 해준다는 것을 상기하여 시선을 track하는 모델을 만들어보고 싶어졌습니다. 그러나 시선을 track하는 모델을 바로 만들기는 어려워서, 눈 주변을 감지하는 모델을 먼저 만들어보고, 시선을 track하는 모델을 구현했습니다.

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

# 두 번째 프로젝트 : 얼굴 인식 모델
Motivation : 첫 번째 프로그램과 같은 맥락에서, 결국 얼굴 인식으로 잠금해제를 하는 것에서 영감을 받아, 마지막으로는 얼굴을 인식하는 모델을 만들어보았습니다. 저를 인식하는 모델을 만들어보고 싶었지만, 아무래도 train data, test data를 충분히 확보하지 못 할 것 같아서, 연예인 아이유를 인식하는 모델을 만들어보았습니다. 연예인에 대해서 잘 몰랐지만, 연예인을 검색할 때 아이유 사진이 많이 나온다는 것을 확인하였고, 데이터를 많이 수집할 수 있을 것 같아서 아이유 얼굴 인식 모델을 만들어 보았습니다. 첫 번째 프로그램과 달리, 제가 직접 데이터들을 선별하였고(허수 데이터들 분별) CNN 기법으로 모델 훈련을 마쳤습니다. 이미지를 다운 받을 수 있는 bing-image-downloader를 사용하였습니다.

_(아이유)얼굴 인식 모델_

```
from bing_image_downloader import downloader

# '아이유' 이미지를 다운로드
downloader.download('아이유', limit=100, output_dir='dataset', force_replace=False, verbose=True)

# 원본 이미지가 저장된 디렉토리
original_dataset_dir = 'dataset/IU'
```
bing image downloader를 통해 bing에서 아이유 사진 100장을 다운받고, 저장될 디렉토리를 명시적으로 만든다. (이때 일일이 확인하여 삭제할 사진들은 전부 삭제하였다)

```
from sklearn.model_selection import train_test_split

# 원본 이미지 디렉토리
all_images_dir = 'dataset/IU'

# 분할 후 이미지가 저장될 디렉토리
train_dir = 'dataset/train/iu'
test_dir = 'dataset/test/iu'

# 디렉토리 생성
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 모든 이미지 파일 리스트 가져오기
images = [f for f in os.listdir(all_images_dir) if os.path.isfile(os.path.join(all_images_dir, f))]

# 학습용과 테스트용으로 분할 (80% 학습, 20% 테스트)
train_images, test_images = train_test_split(images, test_size=0.2, random_state=42)

# 학습용 이미지 복사
for img in train_images:
    src = os.path.join(all_images_dir, img)
    dst = os.path.join(train_dir, img)
    shutil.copy(src, dst)

# 테스트용 이미지 복사
for img in test_images:
    src = os.path.join(all_images_dir, img)
    dst = os.path.join(test_dir, img)
    shutil.copy(src, dst)

print(f"학습용 이미지 수: {len(train_images)}")
print(f"테스트용 이미지 수: {len(test_images)}")
```
데이터 중 80%는 학습용 데이터, 20%는 test용 데이터로 구분하였으며, 각각의 디렉토리를 만들어서 이미지를 복사하는 과정이다.

결과(도중 아이유가 아닌 사진들은 수작업으로 삭제하여 총 88개의 데이터를 사용하였다) : 

<img width="204" alt="image" src="https://github.com/user-attachments/assets/548cdfd7-9f0b-41c5-bcdb-0b9847481df9">

```
# 'not_iu' 이미지 다운로드 예시
downloader.download('인물', limit=100, output_dir='dataset', 
                   adult_filter_off=True, force_replace=False, 
                   timeout=60, verbose=True)

# 'not_iu' 이미지 통합
original_not_iu_dir = 'dataset/인물'

# 'not_iu' 데이터 분할
train_not_iu_dir = 'dataset/train/not_iu'
test_not_iu_dir = 'dataset/test/not_iu'

os.makedirs(train_not_iu_dir, exist_ok=True)
os.makedirs(test_not_iu_dir, exist_ok=True)

not_iu_images = [f for f in os.listdir(original_not_iu_dir) if os.path.isfile(os.path.join(original_not_iu_dir, f))]

train_not_iu, test_not_iu = train_test_split(not_iu_images, test_size=0.2, random_state=42)

for img in train_not_iu:
    src = os.path.join(original_not_iu_dir, img)
    dst = os.path.join(train_not_iu_dir, img)
    shutil.copy(src, dst)

for img in test_not_iu:
    src = os.path.join(original_not_iu_dir, img)
    dst = os.path.join(test_not_iu_dir, img)
    shutil.copy(src, dst)

print(f"'not_iu' 학습용 이미지 수: {len(train_not_iu)}")
print(f"'not_iu' 테스트용 이미지 수: {len(test_not_iu)}")
```
-> 아이유가 아닌 사진을 다운 받고, 마찬가지로 디렉토리를 정리해준다.
```
# 데이터셋 디렉토리 경로 설정
base_dir = 'dataset/'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'test')  # 여기서는 test 데이터를 validation으로 사용

# 이미지 크기 및 배치 사이즈 설정
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32

# 데이터 증강 및 전처리
train_datagen = ImageDataGenerator(
    rescale=1./255,         # 픽셀 값을 0-1 범위로 스케일링
    rotation_range=40,      # 회전 범위
    width_shift_range=0.2,  # 수평 이동 범위
    height_shift_range=0.2, # 수직 이동 범위
    shear_range=0.2,        # 전단 변환
    zoom_range=0.2,         # 확대/축소 범위
    horizontal_flip=True,   # 수평 뒤집기
    fill_mode='nearest'     # 빈 공간 채우기 방법
)

validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
```
-> 데이터 셋을 정리 한 후, 데이터의 전처리 과정이다. 이미지 크기, 배치 사이즈를 정하여 ```ImageDataGenerator```로 전처리를 하였다.(훈련 데이터에는 데이터 증강, 검증 데이터에는 데이터 증강을 하지 않고 스케일링만 한다). 전처리의 세세한 설명(원본 코드에는 없는)을 챗지피티를 통해 주석으로 작성하였다.
```validation_datagen```, ```test_datagen```으로 나눈 것을 확인할 수 있다.

```
# 데이터 생성기
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary'  # 이진 분류
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)
```
-> ```flow_from_directory```를 사용하여 훈련, 검증 제너레이터를 생성한다. 
<img width="431" alt="image" src="https://github.com/user-attachments/assets/65174729-6d43-4da2-8aea-2de8c9ee7d43">

```
# 전이 학습 모델 구축
base_model = MobileNetV2(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                         include_top=False,
                         weights='imagenet')
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # 이진 분류
])

model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])
```
-> Transfer Learning을 활용하여, MobileNetV2를 기반으로 한 이진 분류모델을 구축한다(아이유인지 아닌지 확인).
* 베이스 모델이 MobileNetV2인 이유는 빠른 학습과 추론이 가능하여 colab의 GPU 메모리 환경에 적합하여 이를 선택하였다.
* ImageNet은 다양한 이미지 인식 작업에 효과적이어서 사용하였다.
* ```trainable = False``` 로 두어, 사전 학습된 베이스 모델의 가중치를 고정시켜 학습 과정에서 업데이트되지 않게 하였다.(overfitting을 방지)
* 여기서 _일반적으로_ 더 높은 성능을 원한다면, 일부 층을 학습 가능하게 만들고 Fine tuning을 거치면 된다. (False를 True로 한 뒤, 특정 층까지의 layer.trainable=True로 하여, 재컴파일하고 epoch를 재설정하여 재학습하면 된다(이는 해보았지만, 큰 변화가 없고 Colab의 환경에서 GPU사용이 제한적이어서 최종 코드에는 없습니다))
* 모델의 컴파일에서, loss를 ```binary_corssentropy``` 로 설정하였는데, 이는 이진 분류 모델에 적합한 loss function이라서 선택하였다.

```
# 모델 학습
EPOCHS = 10
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE
)
```
-> Colab의 환경에서 Epochs를 10을 설정하여 모델을 학습하면 충분히 reasonable한 시간 하에 모델의 훈련이 끝났다. (Epoch 100은 8분 걸린다)
<img width="229" alt="image" src="https://github.com/user-attachments/assets/b793704f-cdde-435e-97bc-b01af6bdc4b8">

Epoch를 높여가면 반복 학습을 통해 데이터의 패턴을 점진적으로 정확하게 측정하는 것으로 알고 있다. 이를 비교해보기 위해, Epochs 10 과 100을 사용해보았다. 물론 과적합 문제와, 학습 시간이 길어진다. 이는 Early Stopping을 하면 되지만, 한 것과 안 한 것의 차이가 없어, 반영을 안 한 코드 상태로 유지했다.

```
# 테스트 데이터 평가
test_loss, test_acc = model.evaluate(validation_generator, steps=validation_generator.samples // BATCH_SIZE)
print(f'\n테스트 정확도: {test_acc:.4f}')

# 예측 수행
predictions = model.predict(validation_generator, steps=validation_generator.samples // BATCH_SIZE)
predicted_classes = (predictions > 0.5).astype(int).reshape(-1)
true_classes = validation_generator.classes[:len(predicted_classes)]
class_labels = list(validation_generator.class_indices.keys())
```
-> Epochs = 10
<img width="818" alt="image" src="https://github.com/user-attachments/assets/6684c975-21be-44e9-9a9e-45a38ffabbd9">
-> Epoch = 100
<img width="848" alt="image" src="https://github.com/user-attachments/assets/fbf4263a-f340-4591-aa33-24b03d97832f">

확실히 Epoch를 높였더니, 모델의 정확도가 올라갔다!

```
# 모델을 'iu_model.keras' 파일로 저장
model.save('iu_model.keras')
```
모델을 저장한 후, 이미지를 업로드 하여 모델을 사용해보았다.
```
# 업로드된 파일 중 첫 번째 파일을 선택
uploaded_filename = list(uploaded.keys())[0]

# 이미지 로드 및 전처리
img = image.load_img(uploaded_filename, target_size=(224, 224))  # MobileNetV2의 입력 크기
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가
img_array /= 255.0  # 스케일링 (학습 시 적용한 전처리와 동일하게)

# 이미지 시각화 (꼭 필요한 것은 아님)
plt.imshow(img)
plt.axis('off')
plt.title('uploaded image')
plt.show()
```
-> 업로드 할 이미지의 전처리 과정이다
업로드 한 이미지의 전처리 과정 이후 :
<img width="581" alt="스크린샷 2024-12-11 오후 10 36 15" src="https://github.com/user-attachments/assets/07cffc19-dc27-4e8f-921c-8d4b973b38cf">

Epochs = 10 모델의 결과 :
<img width="414" alt="스크린샷 2024-12-11 오후 10 36 02" src="https://github.com/user-attachments/assets/ed87ef41-44cb-4ce0-a3ce-cbdd0432d8e7">

Epochs = 100 모델의 결과 :
<img width="384" alt="image" src="https://github.com/user-attachments/assets/d86e939b-b1f4-4a4a-ab36-1850ec9aa1a8">

정확도가 늘어난 것을 확인할 수 있다.

두 번째 프로젝트를 마무리하며,
첫 번째 두 번째 프로젝트를 통해 이미지 프로세싱과 CNN의 기법을 통해 전처리 과정, 모델 학습 등을 경험해보며 이론적으로만 알았던 개념들이 눈으로 직접 비교하고 실험해보니, 정리가 많이 되었습니다. 다음에는 GNN기법을 공부해봐서, 스스로 프로젝트를 만들어 볼 것입니다.

# 세 번째 프로젝트 : 주가 지수 예측 모델
Motivation : 퀀트 투자는 고급 수학 능력과 컴퓨터 알고리즘을 활용하여 주가 지수를 예측하고, 최선의 hedging전략을 펼치는 투자 전략입니다. 이는 투자 결정에서 인간의 감정, 직관 등을 배제하고 데이터를 기반으로 분석한다는 점에서 기존의 분석을 대체하였습니다. 머신러닝 알고리즘은 대규모 금융 데이터에서 패턴을 탐지하는데 강력한 도구가 될 수 있습니다. 저는 파이썬을 활용하여 어떻게 금융 데이터를 분석하는 모델을 구현할까? 라는 궁금증에서 출발해서 이 모델을 구현하게 되었습니다.

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






