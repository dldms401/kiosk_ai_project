from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate


from langchain.prompts import FewShotPromptTemplate
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
   {
       "menu" : {
        "menu_item" : "Just Veggie Omelet",
        "menu_description" : "Sauteed spinach and mushrooms, goat cheese."    
       },
        "attrib_structure" : {
            "primary_protein" : "veggie",
            "secondary_protein" :  "egg",
            "tertiary_protein" : "",
            "Cheese Types": "Cheese",
            "Lettuce Types": "Spinach"
        }
    },
   {
        "menu" : {
            "menu_item" : "Buddha bowl",
            "menu_description" : "Assorted vegetables with egg, bacon, chicken, Spinach, Cheddar and black red bean sausage",
        },
        "attrib_structure" : {
            "primary_protein": "chicken",
            "secondary_protein": "bacon",
            "tertiary_protein": "egg",
            "Cheese Types": "Cheddar",
            "Lettuce Types": "Spinach"
        }
    },
   {
        "menu" : {
            "menu_item" : "Bacon topped grilled Organic Chicken Breast",
            "menu_description" : "Tomato, Spring Mix with American cheese, suasage and avacado",
        },
        "attrib_structure" : {
            "primary_protein": "Chicken",
            "secondary_protein": "Bacon",
            "tertiary_protein": "sausage",
            "Cheese Types": "American Cheese",
            "Lettuce Types": "Spring Mix"
        }
    }
]

# This is a prompt template used to format each individual example.
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Enter Menu Description:"),
        ("human", "{menu}"),
        ("ai", "{attrib_structure}"),
    ]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

print(few_shot_prompt.format())