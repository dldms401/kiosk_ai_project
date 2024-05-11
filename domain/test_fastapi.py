from fastapi import FastAPI, UploadFile, File

from data_models import Menu, UserScript, SearchKeywords

from face_recognition.test_ai_face_recog import recog
from order.test_ai_order import add_history, order

from PIL import Image

import io
import numpy as np

app = FastAPI()


# 얼굴 인식
@app.post("/analyze-image/")
async def analyze_image(file: UploadFile = File(...)):

    # 파일 변환
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    image_np = np.array(image)

    # 얼굴 인식, 나이 추출
    age = recog(image_np)

    return age


# 메뉴 추가
@app.post("/add-menu")
async def add_menu(menu:Menu):

    menu_prompt = f'(name: {menu.name}, price: {menu.price}, description: {menu.description}, categoryName: {menu.categoryName}, imgPath: {menu.imagePath})'

    # langchain의 buffermemory에 저장
    add_history(menu_prompt)

    return {"message" : "add success"}


# 키워드 선택후 AI search
@app.post("/fast/api/search")
async def search_menu(search_keywords:SearchKeywords):

    # 리스트를 콤마로 구분, 대괄호로 묶어 문자열로 변환 후 ai에 전달
    ingredients_str = ','.join(search_keywords.ingredients)
    result = order('[' + ingredients_str + ']')

    return result


# chatgpt 응답
@app.post("/order-ai")
async def order_ai(userscr_question:UserScript):

    user_question = userscr_question.userScript

    # ai에 전달
    result = order(user_question)

    return result