from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# langchain은 프롬프트 템플릿을 만들고 작업할 수 있는 도구를 제공함. =>Template
# 빠른 참조. 프롬프트를 생성하기 위한 사전 정의 된 레시피. for 언어모델

import os
from dotenv import load_dotenv, find_dotenv

# 현 폴더에 없으면 상위폴더로 찾아가면서 .env 파일찾으면 로드
_ = load_dotenv(find_dotenv())

# import os 노출 안되게끔 윈도우 환경변수 OPENAI_API_KEY로 키값 넣기 가능
llm = ChatOpenAI(
    api_key=os.getenv("SERV_KEY"),
    # model_name="gpt-3.5-turbo", default값
    temperature=0,
    max_tokens=50
    )


template = ChatPromptTemplate.from_messages([
    ("system", "너는 까페 알바생. 다음 순으로 주문을 받아. 1. 주문메뉴, 2. 옵션, 3. 수량. 확인해줘. 1. 주문메뉴 먼저 받아줘."),
    ("human", "{menu}")
])


chat = template | llm

def order(str):
    return chat.invoke({"menu" : str})


if __name__=="__main__":
    while True:
        print(order(input('human : ')))