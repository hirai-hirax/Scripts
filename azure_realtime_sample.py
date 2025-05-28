import os
import asyncio
import websockets
import pyaudio
import numpy as np
import base64
import json
from openai import AzureOpenAI

# 環境変数からエンドポイントとAPIキーを取得
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
DEPLOYMENT_NAME = "gpt-4o-mini-realtime-preview"  # デプロイしたモデルの名前

# WebSocketのURLとヘッダーの設定
WS_URL = f"wss://{AZURE_OPENAI_ENDPOINT}/openai/deployments/{DEPLOYMENT_NAME}/realtime?api-version=2024-12-17"
HEADERS = {
    "Authorization": f"Bearer {AZURE_OPENAI_API_KEY}",
    "OpenAI-Beta": "realtime=v1"
}
# Azure OpenAI Service client initialization
client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2024-12-17"
)

# 音声入力の設定
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# PCM16形式に変換する関数
def pcm16_to_base64(audio_data):
    return base64.b64encode(audio_data).decode("utf-8")

# 音声を送信する非同期関数
async def send_audio(websocket):
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

    print("マイクからの入力を開始します。話しかけてください。")

    try:
        while True:
            data = stream.read(CHUNK)
            base64_audio = pcm16_to_base64(data)
            audio_event = {
                "type": "input_audio_buffer.append",
                "audio": base64_audio
            }
            await websocket.send(json.dumps(audio_event))
            await asyncio.sleep(0.1)
    except KeyboardInterrupt:
        print("音声入力を終了します。")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

# 応答を受信する非同期関数
async def receive_response(websocket):
    async for message in websocket:
        response = json.loads(message)
        if response.get("type") == "text":
            japanese = response["text"]
            print(f"文字起こし（日本語）: {japanese}")
            # 翻訳
            resp_tr = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "日本語を英語に翻訳してください。"},
                    {"role": "user", "content": japanese}
                ]
            )
            translation = resp_tr.choices[0].message.content
            print(f"翻訳（英語）: {translation}")
            # 合成音声
            resp_audio = client.chat.completions.create(
                model="gpt-4o-mini-audio-preview",
                modalities=["audio"],
                audio={"voice": "alloy", "format": "wav"},
                messages=[{"role": "user", "content": translation}]
            )
            audio_b64 = resp_audio.choices[0].message.audio.data
            audio_bytes = base64.b64decode(audio_b64)
            # 再生
            p_play = pyaudio.PyAudio()
            stream_out = p_play.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True)
            stream_out.write(audio_bytes)
            stream_out.stop_stream()
            stream_out.close()
            p_play.terminate()

# メイン関数
async def main():
    async with websockets.connect(WS_URL, additional_headers=HEADERS) as websocket:
        await asyncio.gather(
            send_audio(websocket),
            receive_response(websocket)
        )

if __name__ == "__main__":
    asyncio.run(main())
