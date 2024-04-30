#import cv2
from deepface import DeepFace

def recog(frame):

    # opencv보다 더욱 훈련된 모델 yolov8을 사용하여 얼굴 감지
    # enforce_detection=False. 얼굴 인식 캡처.
    result = DeepFace.analyze(frame, actions=['age'], detector_backend='yolov8', enforce_detection=False)

    age = result[0]['age']
    return age
    