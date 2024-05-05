#deepface, tf-keras
from deepface import DeepFace

def recog(image_np):

    # opencv보다 더욱 훈련된 모델 yolov8를 백엔드 디텍터로 사용하여 얼굴 감지
    # enforce_detection=False. 얼굴 캡처 확인.
    result = DeepFace.analyze(image_np, actions=['age'], detector_backend='yolov8', enforce_detection=False)

    age = result[0]['age']
    return age