import base64
import os
from pydub import AudioSegment
from openai import AzureOpenAI

# 環境変数からAzure OpenAIのエンドポイントとAPIキーを取得
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
API_VERSION = "2023-09-01-preview"

client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=API_VERSION
)

# 変換したい音声ファイルのパスを指定（MP3以外にもWAV、OGG、FLACなどに対応）
input_path = r"C:\Users\hir31\Downloads\interview_aps-smp.mp3"

# ファイルの拡張子から形式を自動判別（例：".mp3" → "mp3"）
_, ext = os.path.splitext(input_path)
audio_format = ext.lower().strip(".")

# 音声ファイルを読み込む
audio = AudioSegment.from_file(input_path, format=audio_format)

# モノラル（1チャンネル）に変換
mono_audio = audio.set_channels(1)

# 読み込んだ音声データ（モノラル）をWAV形式に変換して出力
output_wav = "sample.wav"
mono_audio.export(output_wav, format="wav")

# 変換したWAVファイルを読み込み、文字起こしAPIに送信する
with open(output_wav, "rb") as audio_file:
    prompt = """
以下の音声ファイルの内容を、できるだけ忠実に文字起こししてください。
"""
    completion = client.audio.transcriptions.create(
        model="gpt-4o-transcribe", # Azure OpenAIのデプロイ名に合わせてください
        file=audio_file,
        prompt=prompt,
        response_format="json", # または "json", "srt", "vtt", "text"
        timestamp_granularities=["segment"]
    )

print(completion.model_dump_json(indent=2))
