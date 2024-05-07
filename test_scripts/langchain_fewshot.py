from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.few_shot import FewShotChatMessagePromptTemplate
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
    max_tokens=75
    )

# 데이터 셋
examples = [
    {"order_message": "메뉴 알려줘요", 
     "answer": "죄송합니다, 현재 메뉴는 아직 준비중에 있어요."},
    {"answer": "(새로운 메뉴 등록. 메뉴명: americano, 메뉴 가격: 6000, 메뉴 설명: 에스프레소에 물을 희석 시켜 만든 음료, 메뉴 카테고리: 커피)", 
     "order_message": "메뉴 뭐 있어요", 
     "answer": "현재 아메리카노가 있습니다. 주문하시겠어요?", 
     "order_message": "아메리카노 뜨겁게 세 잔", 
     "answer": "뜨거운 아메리카노 세 잔 추가하였어요. 더 필요하신게 있으신가요?", 
     "order_message": "아니 이대로 주문할께", 
     "answer": '{"takeout": "takeout", "totalPrice": 18000, "orderDetailRequestDtoList": [{"menuName": "americano","amount": 3,"price": 18000,"temperature": "hot"}]}'},
    {"answer": "(새로운 메뉴 등록. 메뉴명: latte, 메뉴 가격: 5000, 메뉴 설명: 카페 라떼라 부르기도 하며 우유를 곁들인 커피 음료, 메뉴 카테고리: 커피)", 
     "order_message": "라떼 설명", 
     "answer": "라떼는 우유를 곁들인 커피 음료입니다. 가격은 5000원 이고요. 주문하시겠어요?"},
    {"order_message": "아메리카노", 
     "answer": "아메리카노는 따뜻하게 드릴까요? 시원하게 드릴까요?", 
     "order_message": "차갑게", 
     "answer": "차가운 아메리카노 몇 잔 필요하시나요?", 
     "order_message": "두 잔", 
     "answer": "네, 차가운 아메리카노 두 잔 추가하겠습니다. 다른 주문 있으시나요?", 
     "order_message": "아니요", 
     "answer": "차가운 아메리카노 두 잔, 주문 확정하시겠습니까?", 
     "order_message": "네", 
     "answer": '{"takeout": "takeout", "totalPrice": 8000, "orderDetailRequestDtoList": [{"menuName": "americano","amount": 2,"price": 8000,"temperature": "ice"}]}'},
]


example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{order_message}"),
        ("ai", "{answer}")
    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You're a coffee shop attendant. Respond to customer orders."""
        ),
        few_shot_prompt,
        ("human", "{order_message}"),
    ]
)


chat = final_prompt | llm | StrOutputParser()


def order(order_message):
    return chat.invoke(order_message)


if __name__=="__main__":
    while True:
        print(order(input('human : ')))