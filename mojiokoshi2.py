import os
import streamlit as st
from openai import AzureOpenAI
import tempfile
import ffmpeg
from io import BytesIO

# 環境変数から設定を取得（Azure OpenAI のエンドポイント・API キーを設定してください）
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
API_VERSION = "2023-09-01-preview"  # ご利用の API バージョンに合わせてください



def convert_to_mp3_from_mp4(file: BytesIO)

    # ffmpegのinputはファイル形式しか受け付けないためtempfileで一時ファイルを作成します
    with tempfile.NamedTemporaryFile() as temp_file:
        temp_file.write(file.getvalue())
        # mp3に変換します
        (
            ffmpeg
            .input(temp_file.name)
            .output("output.mp3")
            .run()
        )


# Azure:OpenAI クライアントのインスタンスを生成
client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=API_VERSION
)

st.title("オーディオファイルの文字起こし＆テキストファイルダウンロード")
st.write("以下のファイルアップローダーからオーディオファイルをアップロードし、【文字起こし開始】ボタンを押してください。")

# Streamlit のファイルアップローダー（audio/* のみ許可）
uploaded_file = st.file_uploader("オーディオファイルを選択", type=["mp3", "wav", "m4a", "mp4", "webm", "ogg"])

if uploaded_file is not None:
    st.write(f"アップロードされたファイル名: {uploaded_file.name}")

    if st.button("文字起こし開始"):
        try:
            # FileStorage の情報をタプル形式に変換 (filename, stream, content_type)
            file_tuple = (uploaded_file.name, uploaded_file, uploaded_file.type)
            # AzureOpenAI クライアント経由で文字起こし API を呼び出し
            transcript = client.audio.transcriptions.create(
                model="whisper",  # Azure ポータル上のデプロイ名に合わせてください
                file=file_tuple
            )
            # transcript は pydantic モデルなので dict に変換して "text" フィールドを取得
            transcript_text = transcript.model_dump().get("text", "")
            
            st.subheader("文字起こし結果")
            st.text_area("結果", transcript_text, height=300)

            # Shift_JIS でエンコードしてダウンロード用バイト列に変換
            transcript_bytes = transcript_text.encode("shift_jis")
            st.download_button(
                label="文字起こし結果をダウンロード",
                data=transcript_bytes,
                file_name="transcript.txt",
                mime="text/plain; charset=shift_jis"
            )
        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
else:
    st.info("オーディオファイルをアップロードしてください。")
