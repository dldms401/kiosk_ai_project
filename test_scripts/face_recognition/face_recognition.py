import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)

# 아래의 출력을 무한 반복. 실시간같게 보임
while True:
    # ret. 반복해서, 읽기 작업 성공했는지 bool값
    # frame. 반복해서, 읽어낸 이미지MatLike값 frame에
    ret, frame = cap.read()

    if not ret:
        break

    # 얼굴 감지 및 나이 분석. opencv를 기본 감지 백엔드로 삼아 얼굴 감지
    # result = DeepFace.analyze(frame, actions=['age'], detector_backend='opencv', enforce_detection=False)

    # opencv보다 더욱 훈련된 모델 yolov8을 사용하여 얼굴 감지
    result = DeepFace.analyze(frame, actions=['age'], detector_backend='yolov8', enforce_detection=False)

    face_region = result[0]['region']
    
    # 감지된 얼굴에 대한 처리
    for face in result:
        x, y, w, h = face_region['x'], face_region['y'], face_region['w'], face_region['h']
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    age = result[0]['age']
    print("사용자의 예상 나이:", age)
    
    key = cv2.waitKey(10)

    # 사용자가 ESC 키를 누르면 창 닫기
    if key == 27:
        cv2.destroyAllWindows()
        break

    cv2.imshow('Face Detection', frame)

cap.release()