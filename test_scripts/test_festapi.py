from fastapi import FastAPI, UploadFile, File

from test_ai_face_recog import recog
from test_ai_order import order


app = FastAPI()

@app.post("/face-capture")
async def face_recognition(img_file: UploadFile = File):


    age = recog(img_file)

    return age




@app.post("/order-ai")
async def order_ai(string_data: str):
    # 프론트 엔드에서 stt 하고 str 출력.
    # 텍스트를 읽고 답변 또는 json 포맷 주문리스트 생성.
    result = order(string_data)

    # json
    return result