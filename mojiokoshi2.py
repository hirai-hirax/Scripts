import os
import streamlit as st
from openai import AzureOpenAI
import tempfile
import ffmpeg
from io import BytesIO
from pydub import AudioSegment

# 環境変数から設定を取得（Azure OpenAI のエンドポイント・API キーを設定してください）
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
API_VERSION = "2023-09-01-preview"  # ご利用の API バージョンに合わせてください

def convert_to_mp3_from_mp4(file: BytesIO):

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
    return temp_file

def mojiokoshi():
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

def mp3maker():
    st.title("M4A から指定範囲をモノラルMP3に変換（複数ファイル対応）")

    uploaded_file = st.file_uploader("M4A ファイルをアップロード", type=["m4a"])

    # 時間フォーマット変換関数
    def hms_to_seconds(hms):
        """H:MM:SS または MM:SS を秒数に変換"""
        parts = hms.split(":")
        if len(parts) == 3:  # H:MM:SS
            h, m, s = map(int, parts)
            return h * 3600 + m * 60 + s
        elif len(parts) == 2:  # MM:SS
            m, s = map(int, parts)
            return m * 60 + s
        else:
            return int(parts[0])  # 秒のみ

    def seconds_to_hms(seconds):
        """秒数を H:MM:SS 形式に変換"""
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h}:{m:02}:{s:02}"

    if uploaded_file is not None:
        # 一時ファイルとして保存
        with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as temp_m4a:
            temp_m4a.write(uploaded_file.read())
            temp_m4a_path = temp_m4a.name

        # 音声情報を取得
        audio = AudioSegment.from_file(temp_m4a_path, format="m4a")
        duration = len(audio) // 1000  # ミリ秒を秒に変換
        st.write(f"音声の長さ: {seconds_to_hms(duration)}")

        # ユーザーが範囲を入力（複数行対応）
        st.write("範囲を複数指定できます（1行ずつ H:MM:SS - H:MM:SS 形式で入力）")
        time_ranges = st.text_area(
            "例:\n0:00:10 - 0:01:00\n0:02:00 - 0:03:30", 
            height=150
        ).splitlines()

        # 変換処理
        if st.button("変換開始"):
            output_files = []
            
            for i, line in enumerate(time_ranges):
                try:
                    # `H:MM:SS - H:MM:SS` の形式を解析
                    parts = line.split("-")
                    if len(parts) != 2:
                        st.error(f"時間指定のフォーマットが間違っています: {line}")
                        continue

                    start_time = hms_to_seconds(parts[0].strip())
                    end_time = hms_to_seconds(parts[1].strip())

                    if not (0 <= start_time < end_time <= duration):
                        st.error(f"無効な時間範囲: {line}")
                        continue

                    # 一時ファイル名を作成
                    temp_mp3_path = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{i+1}.mp3").name
                    output_files.append(temp_mp3_path)

                    # ffmpeg を使用して変換
                    ffmpeg.input(temp_m4a_path, ss=start_time, to=end_time).output(
                        temp_mp3_path, ac=1, ab="128k", format="mp3"
                    ).run(overwrite_output=True, capture_stdout=True, capture_stderr=True)

                    st.success(f"変換成功: {seconds_to_hms(start_time)} - {seconds_to_hms(end_time)}")

                except Exception as e:
                    st.error(f"変換エラー: {line} → {e}")

            # ダウンロードボタンを表示
            for i, mp3_file in enumerate(output_files):
                with open(mp3_file, "rb") as f:
                    st.download_button(
                        label=f"MP3をダウンロード ({i+1})",
                        data=f,
                        file_name=f"output_{i+1}.mp3",
                        mime="audio/mp3"
                    )
            
            # 一時ファイル削除
            os.remove(temp_m4a_path)
            for mp3_file in output_files:
                os.remove(mp3_file)


def main():
    app_selection = st.sidebar.selectbox("アプリを選択", ["文字起こし", "動画->MP3切り出し"])


    if app_selection == "文字起こし":
        mojiokoshi()
    elif app_selection == "動画->MP3切り出し":
        mp3maker()

if __name__ == "__main__":
    main()