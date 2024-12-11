'''
기계학습과 응용 프로젝트
2020131013 정준혁
목적 : 특정 인물의 얼굴을 인식하는 모델
'''
!pip install bing-image-downloader
from bing_image_downloader import downloader
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np
# '아이유' 이미지를 다운로드
downloader.download('아이유', limit=100, output_dir='dataset', force_replace=False, verbose=True)
import os
import shutil

# 원본 이미지가 저장된 디렉토리
original_dataset_dir = 'dataset/IU'
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
# 데이터셋 디렉토리 경로 설정
base_dir = 'dataset/'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'test')  # 여기서는 test 데이터를 validation으로 사용

# 이미지 크기 및 배치 사이즈 설정
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32

# 데이터 증강 및 전처리
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

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

model.summary()
# 모델 학습
EPOCHS = 100
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE
)
# 테스트 데이터 평가
test_loss, test_acc = model.evaluate(validation_generator, steps=validation_generator.samples // BATCH_SIZE)
print(f'\n테스트 정확도: {test_acc:.4f}')

# 예측 수행
predictions = model.predict(validation_generator, steps=validation_generator.samples // BATCH_SIZE)
predicted_classes = (predictions > 0.5).astype(int).reshape(-1)
true_classes = validation_generator.classes[:len(predicted_classes)]
class_labels = list(validation_generator.class_indices.keys())
# 모델을 'iu_model.keras' 파일로 저장
model.save('iu_model.keras')
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
import io
# 이미지 업로드
uploaded = files.upload()
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

# 예측 수행
prediction = model.predict(img_array)

predicted_class = '아이유' if prediction[0][0] > 0.8 else 'Not IU'

# 예측 확률
probability = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]

print(f'예측 결과: {predicted_class} ({probability * 100:.2f}% 확신)')
