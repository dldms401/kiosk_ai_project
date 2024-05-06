from fastapi import FastAPI, UploadFile, File

# Pydantic 모델들 정의
from data_models import Menu, UserScript

# 다른 scr 에서 호출될 함수들
from test_ai_face_recog import recog
from test_ai_order import add_history
from test_ai_order import order

# 이미지 로드 기능 사용
from PIL import Image

# json 변환
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


# 메뉴 추가
@app.post("/add-menu")
async def add_menu(menu:Menu):
    # 원래 메뉴객체 불러들여올것
    # 관리자 페이지에서 메뉴 등록느낌의, 메뉴 객체 생성
    # menu = Menu("americano", 5500, "쓰지 않은 커피입니다.", "커피")

    # 등록된 메뉴 가져와서 프롬프트로 입력
    menu_prompt = f'새로운 메뉴가 등록되었습니다. 메뉴명: {menu.name}, 메뉴 가격: {menu.price}, 메뉴 설명: {menu.description}, 메뉴 카테고리: {menu.categoryName}'
    # 메뉴 프롬프트 order의 매개변수로.
    # openai api사용한 langchain의 buffermemory내에 입력시킴.
    add_history(menu_prompt)

    return {"message" : "add success"}


# chatgpt 응답
@app.post("/order-ai")
async def order_ai(userscr_question:UserScript):

    user_question = userscr_question.userScript

    # return값은 json 포맷으로.
    result = order(user_question)

    
    return result