import pyaudio
import wave
import time
import os
import io
import threading
import queue
from datetime import datetime
import json

from pydub import AudioSegment
from openai import AzureOpenAI
from dotenv import load_dotenv

# --- 環境変数の設定 ---
# .env ファイルから環境変数を読み込む（ローカル実行時）
load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = "2025-01-01-preview"

# APIキーなどが設定されていない場合はエラーを出す
if not all([AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION]):
    print("エラー: Azure OpenAIの設定情報が不足しています。.envファイルを確認してください。")
    exit(1)

client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION
)

# Whisperモデル名
WHISPER_MODEL = "whisper" # デプロイしたWhisperモデル名に合わせてください
# 要約に使用するGPTモデル名
GPT_MODEL = "gpt-4o-mini" # デプロイしたgpt-4o-miniモデル名に合わせてください

# --- 録音設定 ---
FORMAT = pyaudio.paInt16 # 16-bit resolution
CHANNELS = 1 # mono
RATE = 44100 # 44.1kHz sampling rate (Whisper推奨は16kHzだが、システム音声は高レートで拾う)
CHUNK = 1024 # 1024 samples for each buffer
RECORD_SECONDS_PER_CHUNK = 30 # Whisperに投げる音声の長さ（秒）
AUDIO_DEVICE_INDEX = None # 使用するオーディオデバイスのインデックス (Noneの場合、自動選択)

# --- スレッド間通信用のキュー ---
audio_queue = queue.Queue() # 録音された音声チャンクを格納
transcription_queue = queue.Queue() # 文字起こし結果を格納
stop_event = threading.Event() # 録音スレッドを停止させるためのイベント

# --- PyAudioデバイスの選択 ---
def find_loopback_device_index():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    num_devices = info.get('deviceCount')
    
    print("\n利用可能なオーディオデバイス:")
    for i in range(0, num_devices):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            dev_info = p.get_device_info_by_host_api_device_index(0, i)
            print(f"  ID: {i}, Name: {dev_info.get('name')}, Input Channels: {dev_info.get('maxInputChannels')}, Sample Rate: {dev_info.get('defaultSampleRate')}")
            # WindowsのWASAPIループバックデバイスを探すヒント
            if "Stereo Mix" in dev_info.get('name') or "ステレオ ミキサー" in dev_info.get('name') or "Loopback" in dev_info.get('name'):
                 print(f"  -> おそらくシステム音声を拾えるデバイスです: {dev_info.get('name')}")
                 # 自動選択したい場合はここで return i 
                 # return i 
    
    print("\nシステム音声を拾うには、通常、'ステレオ ミキサー' や 'Stereo Mix'、'Loopback' などと名付けられた入力デバイスを選択します。")
    print("macOSの場合、SoundflowerやBlackHoleなどの仮想オーディオデバイスをインストールして選択する必要があります。")
    
    while True:
        try:
            selected_index = input("使用する入力デバイスのIDを入力してください (デフォルト: Noneでシステムが選択): ")
            if selected_index == "":
                return None
            else:
                return int(selected_index)
        except ValueError:
            print("無効な入力です。数字を入力してください。")

# --- 録音スレッド ---
def record_audio(audio_queue, stop_event, device_index):
    p = pyaudio.PyAudio()
    
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    input_device_index=device_index
                    )
    
    print("\n--- 録音を開始しました。Ctrl+C で停止します。 ---")
    
    frames = []
    
    try:
        while not stop_event.is_set():
            data = stream.read(CHUNK)
            frames.append(data)
            
            if len(frames) * CHUNK >= RATE * RECORD_SECONDS_PER_CHUNK:
                audio_queue.put(b''.join(frames))
                frames = [] # バッファをクリア
    except KeyboardInterrupt:
        print("\n録音を停止します...")
    except Exception as e:
        print(f"録音中にエラーが発生しました: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        # --- 録音終了時に残りの音声データをキューに追加 ---
        if frames: # 残りのデータがあれば
            audio_queue.put(b''.join(frames))
            print(f"残りの音声データ ({len(frames) * CHUNK / RATE:.2f}秒) を文字起こしキューに追加しました。")
        # --------------------------------------------------
        audio_queue.put(None) # 録音終了を知らせるマーカー

# --- 文字起こしスレッド ---
def transcribe_audio(audio_queue, transcription_queue):
    # PyAudioインスタンスをスレッド内で初期化 (get_sample_sizeのため)
    p_local = pyaudio.PyAudio() 
    try:
        while True:
            audio_data_raw = audio_queue.get()
            if audio_data_raw is None:
                transcription_queue.put(None)
                break
            
            # 空のデータチャンクはスキップ
            if not audio_data_raw:
                continue

            audio_segment = AudioSegment(
                data=audio_data_raw,
                sample_width=p_local.get_sample_size(FORMAT), # p_localを使用
                frame_rate=RATE,
                channels=CHANNELS
            )
            
            with io.BytesIO() as audio_file_in_memory:
                audio_segment.export(audio_file_in_memory, format="wav")
                audio_file_in_memory.seek(0)
                
                try:
                    print(f"文字起こしリクエストを送信中... ({len(audio_data_raw) / RATE / CHANNELS / p_local.get_sample_size(FORMAT):.2f}秒の音声)") # p_localを使用
                    response = client.audio.transcriptions.create(
                        model=WHISPER_MODEL,
                        file=("recorded_audio_chunk.wav", audio_file_in_memory.read(), "audio/wav"),
                        language="ja",
                        prompt="こんにちは。\n\nはい、こんにちは。\n\nお元気ですか？\n\nはい、元気です。\n\nそれは何よりです。\n\nはい、よろしくお願いいたします。"
                    )
                    transcribed_text = response.text
                    print(f"文字起こし結果: {transcribed_text[:50]}...") # 結果の一部を表示
                    transcription_queue.put(transcribed_text)
                except Exception as e:
                    print(f"Whisper API呼び出し中にエラーが発生しました: {e}")
                    transcription_queue.put(f"[エラー: {e}]")
    finally:
        p_local.terminate() # スレッド終了時にPyAudioインスタンスをクリーンアップ


# --- 要約関数 ---
def summarize_text_with_gpt4o_mini(text_to_summarize, gpt_client, gpt_model_name):
    print(f"\n--- GPT-4o-miniで要約を生成中 ({gpt_model_name})... ---")
    
    system_prompt = """
    以下のテキストをYAML形式で要約してください。
    出力は以下の構造に従ってください。
    
    summary:
      title: "要約のタイトル"
      overview: "全体の概要を簡潔に記述 (50-100字程度)"
      key_points:
        - "重要なポイント1"
        - "重要なポイント2"
        - "重要なポイント3"
      action_items:
        - "今後のアクションがあれば記述"
    
    注記:
    - 重要なポイントは3つ程度に絞ってください。
    - アクションアイテムがない場合は 'None' と記述してください。
    - 全て日本語で記述してください。
    """
    
    try:
        response = gpt_client.chat.completions.create(
            model=gpt_model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text_to_summarize}
            ],
            temperature=0.7, # 創造性の度合いを調整 (0.0-1.0)
            max_tokens=500 # 要約の最大長を調整
        )
        
        summary_content = response.choices[0].message.content
        print("\n--- 要約結果 (YAML形式) ---")
        print(summary_content)
        return summary_content

    except Exception as e:
        print(f"GPT-4o-miniでの要約中にエラーが発生しました: {e}")
        return f"要約生成エラー: {str(e)}"

# --- メイン処理 ---
if __name__ == "__main__":
    p = pyaudio.PyAudio() # デバイス選択のために一時的にインスタンス作成
    AUDIO_DEVICE_INDEX = find_loopback_device_index()
    p.terminate() # 一時インスタンスを終了

    if AUDIO_DEVICE_INDEX is None:
        print("警告: デバイスが指定されなかったため、システムデフォルトの入力デバイスが使用されます。システム音声を拾えない可能性があります。")
        print("      Windowsの場合、'ステレオ ミキサー'などのループバックデバイスが有効になっているか確認してください。")
        print("      macOS/Linuxの場合、仮想オーディオデバイスが適切に設定されているか確認してください。")

    print(f"\n使用する入力デバイスID: {AUDIO_DEVICE_INDEX if AUDIO_DEVICE_INDEX is not None else 'システムデフォルト'}")
    print(f"録音チャンクの長さ: {RECORD_SECONDS_PER_CHUNK}秒")

    transcribed_output = []

    # 録音スレッドを開始
    record_thread = threading.Thread(target=record_audio, args=(audio_queue, stop_event, AUDIO_DEVICE_INDEX))
    record_thread.start()

    # 文字起こしスレッドを開始
    transcribe_thread = threading.Thread(target=transcribe_audio, args=(audio_queue, transcription_queue))
    transcribe_thread.start()

    try:
        while True:
            text_chunk = transcription_queue.get()
            if text_chunk is None:
                break
            transcribed_output.append(text_chunk)
            print(transcribed_output)
            os.system('cls' if os.name == 'nt' else 'clear')
            print("--- 現在の文字起こし結果 ---")
            print("".join(transcribed_output))
            print("\n--- 録音中... Ctrl+C で停止 ---")

    except KeyboardInterrupt:
        print("\nユーザーによって停止されました。")
    finally:
        stop_event.set()
        record_thread.join()
        transcribe_thread.join()

        final_transcript = "".join(transcribed_output).strip() # 最終的な文字起こし結果
        
        # 最終的な文字起こし結果の保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_transcript_filename = f"system_audio_transcript_{timestamp}.txt"
        
        try:
            with open(output_transcript_filename, "w", encoding="shift_jis", errors="ignore") as f:
                f.write(final_transcript)
            print(f"\n文字起こし結果を '{output_transcript_filename}' に保存しました。")
        except Exception as e:
            print(f"文字起こし結果の保存中にエラーが発生しました: {e}")
            try:
                with open(output_transcript_filename, "w", encoding="utf-8") as f:
                    f.write(final_transcript)
                print(f"\n文字起こし結果を '{output_transcript_filename}' にUTF-8で保存しました。")
            except Exception as e_utf8:
                print(f"UTF-8での保存も失敗しました: {e_utf8}")

        # --- ここから要約処理の追加 ---
        if final_transcript:
            print("\n--- 文字起こし結果の要約を開始します ---")
            summary_yaml = summarize_text_with_gpt4o_mini(final_transcript, client, GPT_MODEL)
            
            summary_filename = f"summary_{timestamp}.yaml"
            try:
                # YAML形式で保存
                with open(summary_filename, "w", encoding="utf-8") as f:
                    f.write(summary_yaml)
                print(f"要約結果を '{summary_filename}' に保存しました。")
            except Exception as e:
                print(f"要約結果の保存中にエラーが発生しました: {e}")
        else:
            print("\n文字起こし結果がないため、要約はスキップされました。")
        # --- 要約処理の追加ここまで ---