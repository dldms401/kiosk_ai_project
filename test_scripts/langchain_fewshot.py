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
    {
        "order_message": "Americano",
        "answer": "Sure, I'll add an Americano. Would you like it cold or hot?"
    },
    {
        "order_message": "Apple Juice",
        "answer": "Sure, I'll add apple juice. Would you like it cold or hot?"
    },
    {
        "order_message": "Chocolate Latte",
        "answer": "Sure, I'll add a chocolate latte. Would you like it cold or hot?"
    },
    {
        "order_message": "Matcha Latte",
        "answer": "Sure, I'll add a matcha latte. Would you like it cold or hot?"
    },
    {
        "order_message": "Milk tea",
        "answer": "Sure, I'll add an milk tea. Would you like it cold or hot?"
    },
    {
        "order_message": "Green tea",
        "answer": "Sure, I'll add green tea. Would you like it cold or hot?"
    },
    {
        "order_message": "Latte",
        "answer": "Sure, I'll add a latte. Would you like it cold or hot?"
    },
    {
        "order_message": "Vanilla latte",
        "answer": "Sure, I'll add a vanilla latte. Would you like it cold or hot?"
    },
    {
        "order_message": "Make it cold",
        "answer": "Sure, I'll add a cold americano. How many cups would you like?"
    },
    {
        "order_message": "Make it Cold",
        "answer": "Sure, I'll add cold apple juice. How many cups would you like?"
    },
    {
        "order_message": "Make it Cold",
        "answer": "Sure, I'll add a cold chocolate Latte. How many cups would you like?"
    },
    {
        "order_message": "Make it Cold",
        "answer": "Sure, I'll add cold matcha Latte. How many cups would you like?"
    },
    {
        "order_message": "Make it cold",
        "answer": "Sure, I'll add a cold milk tea. How many cups would you like?"
    },
    {
        "order_message": "Make it Cold",
        "answer": "Sure, I'll add cold green tea. How many cups would you like?"
    },
    {
        "order_message": "Make it Cold",
        "answer": "Sure, I'll add a cold latte. How many cups would you like?"
    },
    {
        "order_message": "Make it Cold",
        "answer": "Sure, I'll add cold vanilla latte. How many cups would you like?"
    },
    {
        "order_message": "Make it Hot",
        "answer": "Sure, I'll add a hot Americano. How many cups would you like?"
    },
    {
        "order_message": "Make it Hot",
        "answer": "Sure, I'll add a hot apple juice. How many cups would you like?"
    },
    {
        "order_message": "Make it Hot",
        "answer": "Sure, I'll add a hot Chocolate Latte. How many cups would you like?"
    },
    {
        "order_message": "Make it Hot",
        "answer": "Sure, I'll add a hot Matcha Latte. How many cups would you like?"
    },
    {
        "order_message": "Make it Hot",
        "answer": "Sure, I'll add a hot milk tea. How many cups would you like?"
    },
    {
        "order_message": "Make it Hot",
        "answer": "Sure, I'll add a hot green tea. How many cups would you like?"
    },
    {
        "order_message": "Make it Hot",
        "answer": "Sure, I'll add a hot latte. How many cups would you like?"
    },
    {
        "order_message": "Make it Hot",
        "answer": "Sure, I'll add a hot vanilla latte. How many cups would you like?"
    },
    {
        "order_message": "One cup",
        "answer": "Sure, I'll add a cold Americano, one cup. Would you like to confirm your order?"
    },
    {
        "order_message": "Two cups",
        "answer": "Sure, I'll add cold apple juice, two cups. Would you like to confirm your order?"
    },
    {
        "order_message": "Three cups",
        "answer": "Sure, I'll add a hot matcha latte, three cups. Would you like to confirm your order?"
    },
    {
        "order_message": "Four cups",
        "answer": "Sure, I'll add hot strawberry juice, four cups. Would you like to confirm your order?"
    },
    {
        "order_message": "Five cups",
        "answer": "Sure, I'll add a cold milk tea, five cups. Would you like to confirm your order?"
    },
    {
        "order_message": "Six cups",
        "answer": "Sure, I'll add cold green tea, six cups. Would you like to confirm your order?"
    },
    {
        "order_message": "Seven cups",
        "answer": "Sure, I'll add a hot latte, Seven cups. Would you like to confirm your order?"
    },
    {
        "order_message": "Eight cups",
        "answer": "Sure, I'll add hot vanilla latte, eight cups. Would you like to confirm your order?"
    }
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
        ("system", """You're a cafe barista. Be kind and polite, assist with orders.
        X = menu, Y = option (cold or hot), Z = num
        When taking an order, offer additions. ex)user saying : X. answer is : Sure, I'll add an X. Would you like it cold or hot?
        Then, Ask if user like user order cold or hot. ex)user saying : Make it Y. answer is : Sure, I'll add a Y X. How many cups would you like?
        Then, Finally, confirm the quantity and ask if user like to finalize the order, while remembering the details. ex)user saying : Z cups. answer is : Sure, I'll add a Y X, Z cups. Would you like to confirm your order?
        When the user confirms an order, I'll repeat back the previous menu, options, and quantities. 
        Once a user places an order, it should never be altered to something else.
        Only use Korean."""
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