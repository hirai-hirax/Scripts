import os
import base64
from openai import AzureOpenAI

AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
API_VERSION = "2025-01-01-preview"  # ご利用の API バージョンに合わせてください
DEPLOYMENT = "gpt-4o-mini-audio-preview"
model = "gpt-4o-mini-audio-preview"

prompt = """
あなたはプロの翻訳者です。
与えられた日本語の文字列を忠実に英訳し、音声合成してください。
"""

text_to_talk = """
今日はABC株式会社のリスク管理について説明します。  
"""

# クライアント初期化
client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=API_VERSION,
)

# テキスト＋音声の同時リクエスト
response = client.chat.completions.create(
    model=model,
    modalities=["text", "audio"],
    audio={
        "voice": "alloy",     # 選択可能な voice: alloy, echo, shimmer
        "format": "wav"       # フォーマット: wav, mp3 など
    },
    messages=[
        {"role": "system", "content": prompt},
        {"role": "user",   "content": text_to_talk}
    ],
)

# --- テキスト部分を取り出してファイルに書き出し ---
print(response)
text_response = response.choices[0].message.audio.transcript
with open("output.txt", "w", encoding="utf-8") as txt_file:
    txt_file.write(text_response)
print("「output.txt」を生成しました。")

# --- 音声部分を取り出して WAV ファイルに書き出し ---
audio_base64 = response.choices[0].message.audio.data
audio_bytes = base64.b64decode(audio_base64)
with open("output.wav", "wb") as wav_file:
    wav_file.write(audio_bytes)
print("「output.wav」を生成しました。")
