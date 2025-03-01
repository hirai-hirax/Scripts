import os
import streamlit as st
from openai import AzureOpenAI
import tempfile
import ffmpeg
from io import BytesIO
from pydub import AudioSegment
import zipfile
import shutil

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

    st.title("M4A から指定範囲をモノラルMP3に変換（まとめてダウンロード対応）")

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

    # セッションステートで範囲リストを管理
    if "time_ranges" not in st.session_state:
        st.session_state.time_ranges = []

    if uploaded_file is not None:
        # 一時ファイルとして保存
        with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as temp_m4a:
            temp_m4a.write(uploaded_file.read())
            temp_m4a_path = temp_m4a.name

        # 音声情報を取得
        audio = AudioSegment.from_file(temp_m4a_path, format="m4a")
        duration = len(audio) // 1000  # ミリ秒を秒に変換
        st.write(f"音声の長さ: {seconds_to_hms(duration)}")

        # 範囲の管理
        st.subheader("変換する範囲を設定")

        # 範囲を追加するボタン
        if st.button("範囲を追加"):
            st.session_state.time_ranges.append({"start": "0:00:00", "end": seconds_to_hms(duration)})

        # 各範囲の入力フォーム
        for idx, time_range in enumerate(st.session_state.time_ranges):
            col1, col2, col3 = st.columns([4, 4, 1])

            with col1:
                start_time = st.text_input(f"開始時間 {idx+1}", value=time_range["start"], key=f"start_{idx}")
            with col2:
                end_time = st.text_input(f"終了時間 {idx+1}", value=time_range["end"], key=f"end_{idx}")
            with col3:
                if st.button("❌", key=f"remove_{idx}"):
                    st.session_state.time_ranges.pop(idx)
                    st.experimental_rerun()

            # 更新された値を保存
            st.session_state.time_ranges[idx] = {"start": start_time, "end": end_time}

        # 変換処理
        if st.button("変換開始"):
            output_files = []
            zip_file_path = tempfile.NamedTemporaryFile(delete=False, suffix=".zip").name
            temp_dir = tempfile.mkdtemp()

            for i, time_range in enumerate(st.session_state.time_ranges):
                try:
                    start_time = hms_to_seconds(time_range["start"])
                    end_time = hms_to_seconds(time_range["end"])

                    if not (0 <= start_time < end_time <= duration):
                        st.error(f"無効な時間範囲: {time_range['start']} - {time_range['end']}")
                        continue

                    # ファイル名に範囲情報を追加
                    mp3_filename = f"clip_{seconds_to_hms(start_time).replace(':', '-')}_to_{seconds_to_hms(end_time).replace(':', '-')}.mp3"
                    temp_mp3_path = os.path.join(temp_dir, mp3_filename)
                    output_files.append(temp_mp3_path)

                    # ffmpeg を使用して変換
                    ffmpeg.input(temp_m4a_path, ss=start_time, to=end_time).output(
                        temp_mp3_path, ac=1, ab="128k", format="mp3"
                    ).run(overwrite_output=True, capture_stdout=True, capture_stderr=True)

                    st.success(f"変換成功: {seconds_to_hms(start_time)} - {seconds_to_hms(end_time)}")

                except Exception as e:
                    st.error(f"変換エラー: {time_range['start']} - {time_range['end']} → {e}")

            # ZIP ファイルに圧縮
            with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file in output_files:
                    zipf.write(file, os.path.basename(file))

            # ダウンロードボタンを表示
            with open(zip_file_path, "rb") as f:
                st.download_button(
                    label="すべてのMP3をZIPでダウンロード",
                    data=f,
                    file_name="converted_mp3s.zip",
                    mime="application/zip"
                )

            # 一時ファイル削除
            os.remove(temp_m4a_path)
            os.remove(zip_file_path)
            shutil.rmtree(temp_dir)


def main():
    app_selection = st.sidebar.selectbox("アプリを選択", ["文字起こし", "動画->MP3切り出し"])


    if app_selection == "文字起こし":
        mojiokoshi()
    elif app_selection == "動画->MP3切り出し":
        mp3maker()

if __name__ == "__main__":
    main()