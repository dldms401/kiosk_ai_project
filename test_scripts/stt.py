from openai import OpenAI

# langchain의 퓨샷 템플릿 활용한 chatgpt모델 사용
from langchain_fewshot import order

import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

client = OpenAI(api_key=os.getenv("SERV_KEY"))

# pyaudio, keyboard
import pyaudio
import wave
import keyboard

# 녹음 설정
FORMAT = pyaudio.paInt16
CHANNELS = 1  # 1채널: 모노, 2채널: 스테레오
RATE = 44100  # 샘플링 레이트
CHUNK = 1024  # 버퍼 사이즈ㄱ
WAVE_OUTPUT_FILENAME = "output.wav"

def record_audio():
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    print("녹음 시작...")
    frames = []

    while not keyboard.is_pressed('esc'):  # 'esc' 키를 누를 때까지 계속 녹음
        data = stream.read(CHUNK)
        frames.append(data)

    print("녹음 종료...")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    print("파일 저장 완료:", WAVE_OUTPUT_FILENAME)


# def gpt_turbo_instruct(question):
#     response = client.completions.create(
#         model="gpt-3.5-turbo-instruct",  # 일반 다빈치 모델은 종료. 002사용. 그러나 훈련이 안되어 있는 모델인듯함.
#         prompt=question,
#         max_tokens=20,
#         stream=False    # 스트림. True일때는, 연속적인 질문에 대한 답변을 생성.
#     )
#     return response.choices[0]


# def gpt_turbo(question):
#     return client.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             {"role": "assistant", "content": "안녕하세요 저는 도서관리 시스템입니다. 책 내용을 알려주시면 해당 책에서 답변을 느리겠습니다."},
#             {"role": "user", "content": question}
#         ],
#         max_tokens=20,
#         temperature=0
#     )


def speech_to_text():
    audio_file= open("output.wav", "rb")
    transcription = client.audio.transcriptions.create(
        model="whisper-1", 
        file=audio_file
    )
    return transcription.text


if __name__ == "__main__":
    while True:
        print("녹음을 시작하려면 'r' 키를 누르고, 녹음 종료하려면 'esc' 키를 누르세요.")
        keyboard.wait('r')  # 'r' 키를 누를 때까지 대기

        record_audio()  # 녹음과 저장

        transcr = speech_to_text()
        print(transcr)
        # result = gpt_turbo(transcr)
        # result = gpt_turbo_instruct(transcr)

        result = order(transcr)
        print(result)

        # 파일 삭제
        # file_path = "output.wav"
        # os.remove(file_path)