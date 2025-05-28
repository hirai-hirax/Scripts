import os
import base64
import sounddevice as sd
from scipy.io.wavfile import write as wav_write
from openai import AzureOpenAI
import pyaudio
import wave
import numpy as np
import tkinter as tk
from tkinter import messagebox
import logging

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("talk.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# 言語リスト（表示名, 言語コード）
LANGUAGES = [
    ("日本語", "ja"),
    ("英語", "en"),
    ("中国語", "zh"),
    ("韓国語", "ko"),
    ("フランス語", "fr"),
    ("ドイツ語", "de"),
    ("スペイン語", "es"),
    ("イタリア語", "it"),
    ("タイ語", "th"),
    ("インドネシア語", "id"),
]
LANG_NAME_TO_CODE = {name: code for name, code in LANGUAGES}
LANG_CODE_TO_NAME = {code: name for name, code in LANGUAGES}

# 環境変数から設定を取得
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
API_VERSION = "2025-01-01-preview"
DEPLOYMENT = "gpt-4o-mini-audio-preview"
FS = 16000  # サンプリングレート

# Azure OpenAI クライアント初期化
client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=API_VERSION,
)

# グローバル変数
audio_frames = []
audio_stream = None

def audio_callback(indata, frames, time, status):
    if status:
        logging.error(f"録音エラー: {status}")
    audio_frames.append(indata.copy())

def start_record():
    """録音開始"""
    global audio_frames, audio_stream
    audio_frames = []
    audio_stream = sd.InputStream(samplerate=FS, channels=1,
                                  dtype='int16', callback=audio_callback)
    audio_stream.start()
    logging.info("録音開始")

def stop_record(filename="input.wav"):
    """録音停止して WAV に保存"""
    global audio_stream
    if audio_stream:
        audio_stream.stop()
        audio_stream.close()
    data = np.concatenate(audio_frames, axis=0)
    wav_write(filename, FS, data)
    logging.info(f"録音を {filename} に保存しました。")
    return filename

def translate_audio_in_out(filename="input.wav", src_lang="ja", tgt_lang="en"):
    """Azure OpenAI の audio-in で翻訳＋音声合成"""
    system_prompt = (
        f"ユーザーから{LANG_CODE_TO_NAME.get(src_lang, src_lang)}の音声が与えられます。"
        f"その内容を、忠実に{LANG_CODE_TO_NAME.get(tgt_lang, tgt_lang)}に翻訳して発言してください。"
        f"質問や指示を与えられた場合、その質問や指示の内容を、忠実に{LANG_CODE_TO_NAME.get(tgt_lang, tgt_lang)}に翻訳してください。質問や指示に答える必要はありません。"
        "必要な発言をした後、直ちに出力を打ち切ってください。"
        "[input.wav]"
    )
    logging.info(f"system_prompt: {system_prompt}")
    with open(filename, "rb") as f:
        audio_bytes = f.read()
    b64_audio = base64.b64encode(audio_bytes).decode("utf-8")
    response = client.chat.completions.create(
        model=DEPLOYMENT,
        modalities=["text", "audio"],
        audio={"voice": "alloy", "format": "wav"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "input_audio", "input_audio": {"data": b64_audio, "format": "wav"}}
            ]},
        ],
        temperature=0.4,
        top_p=0.6,
        max_tokens=1000,
    )
    transcript = response.choices[0].message.audio.transcript
    logging.info(f"response.transcript: {transcript}")
    # 詳細なレスポンス情報をログ出力
    try:
        model = getattr(response, "model", None) or getattr(response, "model_name", None) or ""
        usage = getattr(response, "usage", None) or {}
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
        total_tokens = getattr(usage, "total_tokens", None) if usage else None
        logging.info(f"response.model: {model}")
        logging.info(f"response.usage: prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}, total_tokens={total_tokens}")
    except Exception as e:
        logging.warning(f"Failed to log detailed response info: {e}")
    logging.debug(f"response: {response}")
    audio_out_b64 = response.choices[0].message.audio.data
    audio_out = base64.b64decode(audio_out_b64)
    with open("output.wav", "wb") as wf:
        wf.write(audio_out)
    with open("output.txt", "w", encoding="utf-8") as tf:
        tf.write(transcript)
    logging.info("「output.txt」と「output.wav」を生成しました。")
    return transcript, "output.wav"

import threading

playback_thread = None
playback_stop_flag = False

def playback(filename="output.wav"):
    """WAV を pyaudio で再生（スレッド対応・停止可）"""
    global playback_stop_flag
    wf = wave.open(filename, 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
    data = wf.readframes(1024)
    logging.info("再生開始")
    playback_stop_flag = False
    while data and not playback_stop_flag:
        stream.write(data)
        data = wf.readframes(1024)
    stream.stop_stream()
    stream.close()
    p.terminate()
    logging.info("再生終了")

def start_playback_thread(filename="output.wav"):
    global playback_thread, playback_stop_flag
    if playback_thread and playback_thread.is_alive():
        return  # すでに再生中
    playback_stop_flag = False
    playback_thread = threading.Thread(target=playback, args=(filename,))
    playback_thread.start()

def stop_playback():
    global playback_stop_flag
    playback_stop_flag = True

def on_start():
    start_button.config(state=tk.DISABLED)
    stop_button.config(state=tk.NORMAL)
    start_record()

def on_stop():
    stop_button.config(state=tk.DISABLED)
    start_button.config(state=tk.NORMAL)
    wav_file = stop_record()
    src_lang_name = src_lang_var.get()
    tgt_lang_name = tgt_lang_var.get()
    src_lang = LANG_NAME_TO_CODE[src_lang_name]
    tgt_lang = LANG_NAME_TO_CODE[tgt_lang_name]
    try:
        transcript, out_wav = translate_audio_in_out(wav_file, src_lang, tgt_lang)
        start_playback_thread(out_wav)
        # テキストウィンドウに追記（常にNORMALにしてから書き込み、DISABLEDに戻す）
        text_output.config(state=tk.NORMAL)
        text_output.insert(tk.END, f"[{src_lang_name}→{tgt_lang_name}] {transcript}\n")
        text_output.see(tk.END)
        text_output.config(state=tk.DISABLED)
    except Exception as e:
        messagebox.showerror("エラー", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    root.title("音声翻訳アプリ")

    # 言語選択（表示名で選択、内部的にコードに変換）
    src_lang_var = tk.StringVar(value="日本語")
    tgt_lang_var = tk.StringVar(value="英語")
    src_label = tk.Label(root, text="翻訳元言語")
    src_label.pack()
    src_menu = tk.OptionMenu(root, src_lang_var, *[name for name, code in LANGUAGES])
    src_menu.pack()

    # 入れ替えボタン
    def swap_lang():
        src = src_lang_var.get()
        tgt = tgt_lang_var.get()
        src_lang_var.set(tgt)
        tgt_lang_var.set(src)
    swap_button = tk.Button(root, text="⇄ 入れ替え", command=swap_lang)
    swap_button.pack()

    tgt_label = tk.Label(root, text="翻訳先言語")
    tgt_label.pack()
    tgt_menu = tk.OptionMenu(root, tgt_lang_var, *[name for name, code in LANGUAGES])
    tgt_menu.pack()

    # ボタン
    start_button = tk.Button(root, text="開始", width=20, command=on_start)
    stop_button = tk.Button(root, text="終了", width=20, state=tk.DISABLED, command=on_stop)
    playback_stop_button = tk.Button(root, text="再生終了", width=20, command=stop_playback)
    start_button.pack(pady=10)
    stop_button.pack(pady=10)
    playback_stop_button.pack(pady=10)

    # テキストウィンドウ（初期はNORMAL、書き込み後はDISABLEDで編集不可に）
    text_output = tk.Text(root, height=10, width=50, state=tk.NORMAL)
    text_output.pack(padx=10, pady=10)
    text_output.insert(tk.END, "翻訳結果がここに表示されます。\n")
    text_output.config(state=tk.DISABLED)

    root.mainloop()
