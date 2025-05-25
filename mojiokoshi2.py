import os
import streamlit as st
from openai import AzureOpenAI
import tempfile
import ffmpeg
from io import BytesIO
from pydub import AudioSegment
import zipfile
import shutil
import base64
import fitz
import pandas as pd


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

def get_text_from_pdf(file: BytesIO):
    # ファイル全体をバイト列として読み込む
    file_bytes = file.read()
    # ファイルポインタをリセット（必要に応じて）
    file.seek(0)
    # バイト列とファイルタイプを指定してPDFを開く
    pdf_document = fitz.open(stream=file_bytes, filetype='pdf')
    text = ""
    for page in pdf_document:
        text += page.get_text()
    pdf_document.close()
    return text

def get_text_from_txt(file: BytesIO):
    # テキストファイルを読み込む
    text = file.read().decode("utf-8")
    return text

def mojiokoshi(duration, offset):
    model = "whisper"
    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=API_VERSION
    )

    st.title("オーディオファイルの文字起こし＆テキストファイルダウンロード")
    st.write("以下のファイルアップローダーからオーディオファイルをアップロードし、【文字起こし開始】ボタンを押してください。")

    uploaded_file = st.file_uploader("オーディオファイルを選択", type=["mp3", "wav", "m4a", "mp4", "webm", "ogg"])

    if uploaded_file is not None:
        st.write(f"アップロードされたファイル名: {uploaded_file.name}")

        if st.button("文字起こし開始"):
            try:
                audio_format = uploaded_file.name.split(".")[-1]
                audio = AudioSegment.from_file(uploaded_file, format=audio_format)
                chunk_duration_ms = duration * 1000  # 15分 = 900000ミリ秒
                total_duration_ms = len(audio)
                num_chunks = (total_duration_ms + chunk_duration_ms - 1) // chunk_duration_ms

                full_transcript = ""
                for i in range(num_chunks):
                    start_ms = i * chunk_duration_ms
                    end_ms = min((i + 1) * chunk_duration_ms, total_duration_ms)
                    chunk = audio[start_ms:end_ms]

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        chunk.export(tmp.name, format="wav")
                        file_tuple = (f"chunk_{i+1}.wav", open(tmp.name, "rb"), "audio/wav")

                        transcript = client.audio.transcriptions.create(
                            model=model,
                            file=file_tuple,
                            language="ja",
                            prompt="こんにちは。\n\nはい、こんにちは。\n\nお元気ですか？\n\nはい、元気です。\n\nそれは何よりです。では早速始めましょう。\n\nはい、よろしくお願いいたします。"
                        )
                        chunk_text = transcript.model_dump().get("text", "")
                        full_transcript += f"\n{chunk_text}"

                st.subheader("文字起こし結果（全文）")
                st.text_area("結果", full_transcript.strip(), height=400)

                transcript_bytes = full_transcript.encode("shift_jis")
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


def mojiokoshi_gpt4o_audio_api(model, duration, offset):
    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=API_VERSION
    )

    st.title(f"{model}を使ったオーディオファイルの文字起こし＆テキストファイルダウンロード")
    st.write("以下のファイルアップローダーからオーディオファイルをアップロードし、【文字起こし開始】ボタンを押してください。")

    prompt = """
    以下の音声ファイルの内容を、できるだけ忠実に文字起こししてください。

    複数の話者が含まれている場合、各話者の発言をセットで出力してください。
    話者の名前が明確でない場合は、自動的に"話者A" "話者B"などの名前を割り当て、それぞれの発言内容を記録してください。
    最終的な文字起こしは、話者と発言内容をタブ区切りで出力してください。
    """

    if "transcript_df" not in st.session_state:
        st.session_state["transcript_df"] = pd.DataFrame(columns=["話者", "発言"])
    if "audio_chunks" not in st.session_state:
        st.session_state["audio_chunks"] = []
    if "current_chunk_index" not in st.session_state:
        st.session_state["current_chunk_index"] = 0

    uploaded_file = st.file_uploader("オーディオファイルを選択", type=["mp3", "wav", "m4a", "mp4", "webm", "ogg"])

    if uploaded_file is not None:
        st.write(f"アップロードされたファイル名: {uploaded_file.name}")
        uploaded_pdf = st.file_uploader("文字起こしの参考になるPDFをアップロード(省略可)", type=["pdf"])
        if uploaded_pdf is not None:
            pdf_text = get_text_from_pdf(uploaded_pdf)
            st.write(pdf_text)
            prompt += f"\n\n参考PDFの内容:\n{pdf_text}"

        uploaded_txt = st.file_uploader("文字起こしの参考になるテキストをアップロード(省略可)", type=["txt"])
        if uploaded_txt is not None:
            txt_text = get_text_from_txt(uploaded_txt)
            prompt += f"\n\n参考テキストの内容:\n{txt_text}"
            st.write(txt_text)

        additional_text = st.text_area("補足事項など", height=200)

        if st.button("文字起こし開始"):
            try:
                if uploaded_pdf:
                    prompt += f"\n\n参考PDFの内容:\n{additional_text}"
                if uploaded_txt:
                    prompt += f"\n\n参考テキストの内容:\n{additional_text}"
                if additional_text:
                    prompt += f"\n\nその他補足事項:\n{additional_text}"

                _, ext = os.path.splitext(uploaded_file.name)
                audio_format = ext.lower().strip(".")
                audio = AudioSegment.from_file(uploaded_file, format=audio_format).set_channels(1)

                total_duration_ms = len(audio)
                chunk_duration_ms = duration * 1000
                num_chunks = (total_duration_ms + chunk_duration_ms - 1) // chunk_duration_ms
                st.session_state["audio_chunks"] = []

                for i in range(num_chunks):
                    start_ms = i * chunk_duration_ms
                    end_ms = min((i + 1) * chunk_duration_ms + offset * 1000, total_duration_ms)
                    chunk = audio[start_ms:end_ms]
                    st.session_state["audio_chunks"].append(chunk)

                st.session_state["current_chunk_index"] = 0
                st.success("音声の分割が完了しました。\n再開ボタンを押して推論を開始してください。")
            except Exception as e:
                st.error(f"エラーが発生しました: {str(e)}")

        if st.button("再開") and st.session_state["current_chunk_index"] < len(st.session_state["audio_chunks"]):
            try:
                i = st.session_state["current_chunk_index"]
                chunk = st.session_state["audio_chunks"][i]

                temp_path = f"chunk_{i}.wav"
                chunk.export(temp_path, format="wav")

                with open(temp_path, "rb") as wav_file:
                    encoded_chunk = base64.b64encode(wav_file.read()).decode("utf-8")

                if not st.session_state["transcript_df"].empty:
                    context_text = "\n".join([f"{row['話者']}\t{row['発言']}" for _, row in st.session_state["transcript_df"].iterrows()])
                    contextual_prompt = prompt + f"\n\nこれまでの文脈:\n{context_text}"
                else:
                    contextual_prompt = prompt

                completion = client.chat.completions.create(
                    model=model,
                    modalities=["text"],
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": contextual_prompt + f"\n\n（{i+1}つ目のセクション）"},
                                {
                                    "type": "input_audio",
                                    "input_audio": {
                                        "data": encoded_chunk,
                                        "format": "wav"
                                    }
                                }
                            ]
                        },
                    ]
                )

                transcript_part = completion.choices[0].message.content
                os.remove(temp_path)

                lines = [line.strip().split("\t") for line in transcript_part.strip().splitlines() if "\t" in line]
                df = pd.DataFrame(lines, columns=["話者", "発言"])
                combined_df = pd.concat([st.session_state["transcript_df"], df], ignore_index=True)
                st.session_state["transcript_df"] = combined_df.drop_duplicates(subset=["話者", "発言"]).reset_index(drop=True)

                st.session_state["current_chunk_index"] += 1
                st.success(f"セクション {i+1} の文字起こしが完了しました。テーブルを編集し、必要に応じて再開してください。")

            except Exception as e:
                st.error(f"エラーが発生しました: {str(e)}")

        st.subheader("全体文字起こし結果（編集可能）")
        st.session_state["transcript_df"] = st.data_editor(
            st.session_state["transcript_df"], key="full_transcript_editor", num_rows="dynamic",use_container_width=True,
            column_config={
                "話者": st.column_config.TextColumn(label="話者"),
                "発言": st.column_config.TextColumn(label="発言"),
            },
        )

        if st.button("全文整形・要約"):
            # 文を結合して文脈を整えるための最終整形処理
            raw_transcript = "".join([f"{row['話者']}	{row['発言']}" for _, row in st.session_state["transcript_df"].iterrows()])

            # ChatGPTを使って会話の流れを自然に整形
            refine_model = "gpt-4o-mini"  # 音声不要の整形には通常のgptを使用
            refine_prompt = f"""
            以下は複数話者の会話をタブ区切りで記録したものです。不自然な会話の切れ目や重複を取り除き、全体を読みやすく自然な流れに整えてください。

            {raw_transcript}
            """
            refine_completion = client.chat.completions.create(
                model=refine_model,
                messages=[
                    {"role": "user", "content": refine_prompt}
                ]
            )
            final_transcript = refine_completion.choices[0].message.content

            transcript_bytes = final_transcript.encode("shift_jis")
            st.download_button(
                label="全文の文字起こしをダウンロード",
                data=transcript_bytes,
                file_name="transcript.tsv",
                mime="text/plain; charset=shift_jis"
            )

            st.text_area("全文表示", final_transcript, height=400)
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
    app_selection = st.sidebar.selectbox("文字起こしライブラリまたはアプリを選択", ["whisper", "gpt-4o-audio-preview","gpt-4o-mini-audio-preview","動画->MP3切り出し"])
    duration = st.sidebar.number_input("1推論当たりの時間(sec)", min_value=0, max_value=1800,value=180, step=1)
    offset = st.sidebar.number_input("推論単位の重複させる時間(sec)", min_value=0, max_value=300,value=10, step=1)

    if app_selection == "whisper":
        mojiokoshi(duration, offset)
    elif app_selection in ("gpt-4o-audio-preview","gpt-4o-mini-audio-preview"):
        mojiokoshi_gpt4o_audio_api(app_selection,duration,offset)

    elif app_selection == "動画->MP3切り出し":
        mp3maker()

if __name__ == "__main__":
    main()
