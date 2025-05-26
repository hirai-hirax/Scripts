import os
import base64
import streamlit as st
from openai import AzureOpenAI
import fitz  # PyMuPDF
from dotenv import load_dotenv

load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = "2025-01-01-preview"

# — Azure OpenAI の設定 —
DEPLOYMENT = "gpt-4o-mini-audio-preview"

client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
)

SYSTEM_PROMPT1 = """
あなたはプロの翻訳者です。
与えられた日本語（テキストまたは音声）を、英語参考資料のスタイルや用語を反映して忠実に英訳し、音声合成してください。
"""
SYSTEM_PROMPT2 = """
与えられたテキストを、一字一句変えずに忠実にそのまま出力してください。与えられたテキストを出力した後は直ちに出力を打ち切ってください。
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

def translate_and_speak(system_prompt,
                        text=None,
                        audio_bytes=None,
                        audio_ext="wav",
                        reference=None):
    # messages の組立
    messages = [{"role": "system", "content": system_prompt}]
    if reference:
        messages.append({
            "role": "system",
            "content": f"## 参考資料\n{reference}"
        })

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
    "参考資料ファイルをアップロード（PDFまたはTXT）", type=["pdf", "txt"]
)
reference_text = ""
if uploaded_ref is not None:
    ext = os.path.splitext(uploaded_ref.name)[1].lower()
    if ext == ".pdf":
        reference_text = extract_text_from_pdf(uploaded_ref)
    else:
        reference_text = uploaded_ref.read().decode("utf-8")

input_mode = st.radio("入力モードを選択", ["テキスト", "音声ファイル"])
system_prompt = st.radio("出力モードを選択", ["翻訳", "読み上げ"])

user_text = None
user_audio = None
audio_ext = "wav"

if input_mode == "テキスト":
    user_text = st.text_area("テキストを入力", height=150)
else:
    uploaded = st.file_uploader("音声ファイルをアップロード", type=["wav", "mp3", "m4a"])
    if uploaded is not None:
        user_audio = uploaded.read()
        audio_ext = os.path.splitext(uploaded.name)[1].lstrip(".")

# 実行ボタン
if st.button("翻訳＆合成実行"):
    if (input_mode == "テキスト" and not user_text) or (input_mode == "音声ファイル" and not user_audio):
        st.error("入力を指定してください。")
    else:
        if system_prompt == "翻訳":
            prompt = SYSTEM_PROMPT1
        else:
            prompt = SYSTEM_PROMPT2

        with st.spinner("処理中…"):
            text_out, audio_out = translate_and_speak(
                system_prompt=prompt,
                text=user_text,
                audio_bytes=user_audio,
                audio_ext=audio_ext,
                reference=reference_text
            )
            # 一時ファイルに保存
            with open("output.txt", "w", encoding="utf-8") as f_txt:
                f_txt.write(text_out)
            with open("output.wav", "wb") as f_wav:
                f_wav.write(audio_out)
            st.session_state["generated"] = True

# 結果表示とダウンロードボタン
if st.session_state.get("generated"):
    st.success("完了！")
    st.subheader("▶️ テキスト出力")
    with open("output.txt", "r", encoding="utf-8") as f_txt:
        st.write(f_txt.read())
    st.subheader("🔊 音声プレビュー")
    with open("output.wav", "rb") as f_wav:
        st.audio(f_wav.read(), format="audio/wav")
    with open("output.wav", "rb") as f_wav:
        st.download_button(
            "🔽 音声ファイルをダウンロード",
            f_wav,
            file_name="output.wav",
            mime="audio/wav",
            key="audio_dl"
        )
    with open("output.txt", "rb") as f_txt:
        st.download_button(
            "🔽 翻訳テキストをダウンロード",
            f_txt,
            file_name="output.txt",
            mime="text/plain",
            key="text_dl"
        )
