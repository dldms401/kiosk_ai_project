from data_models import SearchKeywords
from typing import List

def ingredients_to_string(search_keywords: SearchKeywords) -> str:
    ingredients_str = ','.join(search_keywords.ingredients)
    return '[' + ingredients_str + ']'

if __name__ == "__main__":

# 예시
    search_keywords = SearchKeywords(ingredients=["apple", "banana", "orange"])
    ingredients_str = ingredients_to_string(search_keywords)
    print(ingredients_str)  # 출력: "[apple,banana,orange]"