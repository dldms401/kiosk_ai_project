from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
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


examples = [
    {"question": "아메리카노를 줘",
        "answer": "아메리카노를 주문하시겠습니까? 한잔당 4000원 입니다",
    },
    {"question": "네 주세요",
        "answer": "따뜻한걸 원하시나요? 차가운걸 원하시나요?",
    },
    {"question": "차가운거요",
        "answer": "네, 차가운거 추가하였습니다. 더 필요한게 있으시나요?",
    },
]


example_prompt = PromptTemplate.from_template(
    "Human: {question}\nAI: {answer}"
)

# fewshot 프롬 템플릿 객체 생성.
prompt = FewShotPromptTemplate(
    example_prompt=example_prompt,
    examples=examples,              # input type dic으로. fewshot lerning할 예시들 집합.
    suffix="Human: {order}",
    input_variables=["order"],
)


# 연결. 모델과 위의 프롬프트 템플릿까지. 나온 결과물을 스트링으로 
chain = prompt | llm | StrOutputParser()

def order(str):
    # test_communication에서 텍스트 추출한것을 받아옴
    print(chain.invoke({"order" : str}))