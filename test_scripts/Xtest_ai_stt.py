from openai import OpenAI

import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

client = OpenAI(api_key=os.getenv("SERV_KEY"))


def stt(file):

    # wav file 경로 지정 필요
    audio_file= open("output.wav", "rb")

    transcription = client.audio.transcriptions.create(
        model="whisper-1", 
        file=audio_file
    )
    return transcription.text
