from pydantic import BaseModel
from typing import List
# Pydantic 모델들 정의


# 사용자 질문
class UserScript(BaseModel):
    userScript: str

# 메뉴 저장
class Menu(BaseModel):
    name: str
    price: float
    description: str
    categoryName: str
    imagePath: str

# 키워드 리스트
class SearchKeywords(BaseModel):
    ingredients: List[str]


# 학습 테스트용
class AddMenuTest(BaseModel):
    id: int
    name: str
    price: float
    description: str
    categoryName: str