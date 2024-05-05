from fastapi import FastAPI, UploadFile, File

from test_ai_face_recog import recog
from PIL import Image
import io
import numpy as np

app = FastAPI()

@app.post("/analyze-image/")
async def analyze_image(file: UploadFile = File(...)):

    # post요청으로 file(이미지 파일) 받아옴
    # file을 byte형태로 읽어서 contents에 저장.
    contents = await file.read()

    # byte형태의 contents를 읽어 이미지파일 형태로 image에 저장
    image = Image.open(io.BytesIO(contents))

    # Pillow image를 numpy array로 변환
    image_np = np.array(image)

    age = recog(image_np)
    return age