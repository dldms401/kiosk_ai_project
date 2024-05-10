from pydantic import BaseModel
from typing import List
# Pydantic 모델들 정의
# 데이터 유효성 검사 및 직렬화 기능

class UserScript(BaseModel):
    userScript: str


class Menu(BaseModel):
    name: str
    price: float
    description: str
    categoryName: str


class SearchKeywords(BaseModel):
    ingredients: List[str]