from fastapi import FastAPI, UploadFile, File

from test_ai_face_recog import recog
from test_ai_order import order
from test_ai_stt import stt


app = FastAPI()

@app.post("/face-capture")
async def face_recognition(img_file: UploadFile = File):


    age = recog(img_file)

    return age




@app.post("/speech-to-text")
async def speech_to_text(file: UploadFile = File):

    # wav파일 읽고 텍스트로 추출, 텍스트를 읽고 답변 생성. json 포맷 만들기.
    order(stt(file))

    # json
    return 