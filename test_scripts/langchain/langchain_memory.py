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
    # fine tuning된 모델을 사용. 
    # venv\Lib\site-packages\langchain_openai\chat_models\base.py
    # get_num_tokens_from_messages에 새로운 모델 이름 추가해야할것.
    # 패키지를 수정하는 일은, 배포에 알맞지 않은 방식
    model_name="ft:gpt-3.5-turbo-0125:personal:cafebot:9Ly4475o",
    temperature=0.2,
    # max_tokens=40
    )


memory = ConversationSummaryBufferMemory(
    llm= llm,
    # max_token_limit=40,
    memory_key="history",        
    return_messages=True,
)



prompt = ChatPromptTemplate.from_messages(
    [
        (f"system", """You're a coffee shop attendant. Respond to customer orders."""),
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


# def num_tokens_from_messages(messages, model=ft_cafebot):
#   """Returns the number of tokens used by a list of messages."""
#   try:
#       encoding = tiktoken.encoding_for_model(model)
#   except KeyError:
#       encoding = tiktoken.get_encoding("cl100k_base")
#   if model == ft_cafebot:  # note: future models may deviate from this
#       num_tokens = 0
#       for message in messages:
#           num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
#           for key, value in message.items():
#               num_tokens += len(encoding.encode(value))
#               if key == "name":  # if there's a name, the role is omitted
#                   num_tokens += -1  # role is always required and always 1 token
#       num_tokens += 2  # every reply is primed with <im_start>assistant
#       return num_tokens
#   else:
#       raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.""")


def order(str):
    result = chain.predict(order = str)

    try:
        json_data = json.loads(result)
        return json_data
    except ValueError:
        return result
    

# 임시 클래스
class Menu:
    # 인스턴스 초기화
    def __init__(self, name, price, description, categoryName):
        self.name = name
        self.price = price
        self.description = description
        self.categoryName = categoryName

# 원래 메뉴객체 불러들여올것
# def generate_menu_prompt(menu):
def generate_menu_prompt():
    # 관리자 페이지에서 메뉴 등록느낌의, 메뉴 객체 생성
    menu = Menu("americano", 6500, "쓰지 않은 물탄 커피입니다", "커피")
    # 등록된 메뉴 가져와서 프롬프트로 입력예정
    menu_prompt = f'(새로운 메뉴 등록. 메뉴명: {menu.name}, 메뉴 가격: {menu.price}, 메뉴 설명: {menu.description}, 메뉴 카테고리: {menu.categoryName})'


    return menu_prompt

def add_history(menu_prompt):
    memory.chat_memory.add_ai_message(menu_prompt)

if __name__=="__main__":

    # 메뉴 객체 생성후 프롬프트 생성.
    # 생성된 메뉴 프롬프트는 곧바로 인공지능에게 질문을 건네는 방식.
    menu_prompt = generate_menu_prompt()
    add_history(menu_prompt)
    
    while True:
        user_input = input("human : ")
        output_data = order(user_input)

        print(output_data)
        print(type(output_data))