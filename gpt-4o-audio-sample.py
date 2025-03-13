import base64
import os
from pydub import AudioSegment
from openai import AzureOpenAI

# 環境変数からAzure OpenAIのエンドポイントとAPIキーを取得
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
API_VERSION = "2023-09-01-preview"

client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
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

# 変換したWAVファイルを読み込み、Base64エンコードする
with open(output_wav, "rb") as wav_file:
    encoded_string = base64.b64encode(wav_file.read()).decode("utf-8")

prompt = """
以下の音声ファイルの内容を、できるだけ忠実に文字起こししてください。

複数の話者が含まれている場合、各話者の発言をセットで出力してください。
話者の名前が明確でない場合は、自動的に( 話者A』『話者B』などの名前を割り当て、それぞれの発言内容を記録してください。
最終的な文字起こしは、誰が何を発言したかが明確にわかる形式で、CSV形式で出力してください。
"""

# 変換後の音声データ（Base64エンコード済み）をAzure OpenAIに送信する例
completion = client.chat.completions.create(
    model="gpt-4o-mini-realtime-preview",
    modalities=["text"],
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": encoded_string,
                        "format": "wav"
                    }
                }
            ]
        },
    ]
)

print(completion)
