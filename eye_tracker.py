'''
기계학습과 응용 프로젝트
2020131013 정준혁
목적 : 눈을 track하는 모델
'''
import cv2
import numpy as np

# 초기화 부분
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_eye.xml')
detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
detector = cv2.SimpleBlobDetector_create(detector_params)

cap = cv2.VideoCapture(0)
cv2.namedWindow('image')
cv2.createTrackbar('threshold', 'image', 64, 255, lambda x: None)

while True:
    _, frame = cap.read()

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

    text = 'ML_project'
    frame = cv2.putText(frame, text, (40,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    cv2.imshow('image', frame)
    if cv2.waitKey(1) != -1:  # 아무 키나 눌렀을 때
        break


cap.release()
cv2.destroyAllWindows()
