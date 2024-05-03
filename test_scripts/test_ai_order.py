from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from langchain.chains import LLMChain
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory

# langchain은 프롬프트 템플릿을 만들고 작업할 수 있는 도구를 제공함. =>Template
# 빠른 참조. 프롬프트를 생성하기 위한 사전 정의 된 레시피. for 언어모델

import json
import os
from dotenv import load_dotenv, find_dotenv

# 현 폴더에 없으면 상위폴더로 찾아가면서 .env 파일찾으면 로드
_ = load_dotenv(find_dotenv())

# import os 노출 안되게끔 윈도우 환경변수 OPENAI_API_KEY로 키값 넣기 가능
llm = ChatOpenAI(
    api_key=os.getenv("SERV_KEY"),
    # model_name="gpt-3.5-turbo", default값
    temperature=0,
    max_tokens=200
    )


memory = ConversationSummaryBufferMemory(
    llm= llm,
    max_token_limit=400,
    memory_key="history",        
    return_messages=True,
)



prompt = ChatPromptTemplate.from_messages(
    [
        (f"system", """You're a cafe barista. Your mission is to assist customers with their orders in a kind and polite manner. 
         Ensure to collect menus, options, and cup preferences from the user. Options are limited to hot or cold only. Remember the user's order information and allow them to cancel or modify their order at any time.
         Actively check for all necessary information and prompt for any missing details one at a time. Before finalizing the order, confirm the details with the user. 
         If confirmed, generate a JSON response containing the order details without any additional natural language. Ensure consistent handling of corrections or cancellations by the user.
         Additionally, calculate the total price accurately before generating the JSON response.
         Example response in JSON format without any line breaks or spaces:
         {{"takeout": "takeout","totalPrice": 10000,"orderDetailRequestDtoList": [{{"menuName": "americano","amount": 1,"price": 3000,"temperature": "ice"}},{{"menuName": "latte","amount": 2,"price": 7000,"temperature": "ice"}}]}}.
         {{"takeout": "takeout","totalPrice": 6000,"orderDetailRequestDtoList": [{{"menuName": "americano","amount": 2,"price": 6000,"temperature": "hot"}}]}}.
         {{"takeout": "takeout","totalPrice": 7000,"orderDetailRequestDtoList": [{{"menuName": "apple juice","amount": 1,"price": 3500,"temperature": "ice"}},{{"menuName": "apple juice","amount": 1,"price": 3500,"temperature": "hot"}}]}}.
         """
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{order}")
    ]
)

chain = LLMChain(
    llm=llm,
    memory=memory,
    prompt=prompt,
    verbose=True
)


def order(str):
    result = chain.predict(order = str)

    # JSON 포맷으로 변환 가능한지 시도
    try:
        json_data = json.loads(result)
        return json_data
    except ValueError:
        return result