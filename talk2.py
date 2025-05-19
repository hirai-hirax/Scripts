import os
import base64
import streamlit as st
from openai import AzureOpenAI
import fitz  # PyMuPDF

# — Azure OpenAI の設定 —
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
API_VERSION = "2025-01-01-preview"
DEPLOYMENT = "gpt-4o-mini-audio-preview"

client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=API_VERSION,
)

SYSTEM_PROMPT = """
あなたはプロの翻訳者です。
与えられた日本語（テキストまたは音声）を、以下の英語参考資料のスタイルや用語を反映して忠実に英訳し、音声合成してください。
"""

def extract_text_from_pdf(file) -> str:
    """
    fitz (PyMuPDF) を使って PDF からテキストを抽出
    """
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = []
    for page in doc:
        txt = page.get_text()
        if txt:
            text.append(txt)
    return "\n".join(text)

def translate_and_speak(text=None, audio_bytes=None, audio_ext="wav", reference=None):
    # messages の組立
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if reference:
        messages.append({
            "role": "system", 
            "content": f"## 参考資料（英語）\n{reference}"})

    if text:
        messages.append({"role": "user", "content": text})
    else:
        b64 = base64.b64encode(audio_bytes).decode("utf-8")
        messages.append({
            "role": "user",
            "content": [
                {"type": "input_audio",
                 "input_audio": {"data": b64, "format": audio_ext}}
            ]
        })

    # Azure OpenAI 呼び出し
    resp = client.chat.completions.create(
        model=DEPLOYMENT,
        modalities=["text", "audio"],
        audio={"voice": "alloy", "format": "wav"},
        messages=messages,
    )
    transcript = resp.choices[0].message.audio.transcript
    audio_b64 = resp.choices[0].message.audio.data
    audio_out = base64.b64decode(audio_b64)
    return transcript, audio_out

# --- Streamlit UI ---
st.title("🈂️ 日本語→英語 翻訳＋音声合成（参考資料対応版）")

# 参考資料ファイルのアップロード
uploaded_ref = st.file_uploader(
    "英語参考資料ファイルをアップロード（PDFまたはTXT）", type=["pdf", "txt"]
)
reference_text = ""
if uploaded_ref is not None:
    ext = os.path.splitext(uploaded_ref.name)[1].lower()
    if ext == ".pdf":
        reference_text = extract_text_from_pdf(uploaded_ref)
    else:
        reference_text = uploaded_ref.read().decode("utf-8")

input_mode = st.radio("入力モードを選択", ["テキスト", "音声ファイル"])

user_text = None
user_audio = None
audio_ext = "wav"

if input_mode == "テキスト":
    user_text = st.text_area("日本語テキストを入力", height=150)
else:
    uploaded = st.file_uploader("日本語音声ファイルをアップロード", type=["wav", "mp3", "m4a"])
    if uploaded is not None:
        user_audio = uploaded.read()
        audio_ext = os.path.splitext(uploaded.name)[1].lstrip(".")

if st.button("翻訳＆合成実行"):
    if (input_mode == "テキスト" and not user_text) or (input_mode == "音声ファイル" and not user_audio):
        st.error("入力を指定してください。")
    else:
        with st.spinner("処理中…"):
            text_out, audio_out = translate_and_speak(
                text=user_text,
                audio_bytes=user_audio,
                audio_ext=audio_ext,
                reference=reference_text
            )
        st.success("完了！")
        st.subheader("▶️ 翻訳結果（英語）")
        st.write(text_out)
        st.subheader("🔊 音声プレビュー")
        st.audio(audio_out, format="audio/wav")
        st.download_button("🔽 音声ファイルをダウンロード", audio_out, file_name="output.wav", mime="audio/wav")
        st.download_button("🔽 翻訳テキストをダウンロード", text_out, file_name="output.txt", mime="text/plain")
