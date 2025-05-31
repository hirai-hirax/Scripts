import os
import streamlit as st
st.set_page_config(layout="wide")
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
from dotenv import load_dotenv
import json

# .envファイルから環境変数を読み込む
load_dotenv()

# 環境変数から設定を取得（Azure OpenAI のエンドポイント・API キーを設定してください）
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
API_VERSION = "2023-09-01-preview"  # ご利用の API バージョンに合わせてください

summary_prompt = """
    ユーザーから、PDFから抽出されたテキストを渡されます。当該のテキストの内容を読んだ上で、150文字程度の要約を生成してください。
"""

def generate_keywords(model, FilePathOfPdf):
    # クライアント初期化
    text = get_text_from_pdf(FilePathOfPdf)
    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=API_VERSION,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": summary_prompt},
            {"role": "user", "content": text},
        ],
    )
    return response.choices[0].message.content

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

    # セッションファイルのパス
    session_file = "transcription/last_session.json"
    # セッションファイルが存在し、session_state未設定なら復元（初回のみ）
    if os.path.exists(session_file):
        try:
            with open(session_file, "r", encoding="utf-8") as f:
                session_data = json.load(f)
            if "seg_df" in session_data and "seg_df" not in st.session_state:
                st.session_state["seg_df"] = pd.DataFrame(session_data["seg_df"])
            if "results" in session_data and "results" not in st.session_state:
                st.session_state["results"] = session_data["results"]
            if "full_transcript" in session_data and "full_transcript" not in st.session_state:
                st.session_state["full_transcript"] = session_data["full_transcript"]
        except Exception as e:
            st.warning(f"セッションの復元に失敗しました: {e}")

    st.title("オーディオファイルの文字起こし")
    # --- セッションリセットボタン（サイドバーに移動） ---
    if st.sidebar.button("すべてのセッションをリセット"):
        st.session_state.clear()
        session_file = "transcription/last_session.json"
        try:
            if os.path.exists(session_file):
                os.remove(session_file)
            st.sidebar.success("セッションをリセットしました。")
        except Exception as e:
            st.sidebar.warning(f"セッションファイルの削除に失敗しました: {e}")
    st.write("会議資料とオーディオファイルをアップロードしてください。")

    # PDFファイルアップロード
    pdf_file = st.file_uploader("要約に使うPDFファイルを選択", type=["pdf"])
    if pdf_file is not None:
        st.session_state["pdf_file_name"] = pdf_file.name
        st.write(f"アップロードされたPDFファイル名: {pdf_file.name}")
    elif "pdf_file_name" in st.session_state:
        st.write(f"アップロードされたPDFファイル名: {st.session_state['pdf_file_name']}")
    # オーディオファイルアップロード
    uploaded_file = st.file_uploader("オーディオファイルを選択", type=["mp3", "wav", "m4a", "mp4", "webm", "ogg"])

    if uploaded_file is not None:
        st.session_state["uploaded_file_name"] = uploaded_file.name
        st.write(f"アップロードされたファイル名: {uploaded_file.name}")

        # 操作ログ表示用
        log_placeholder = st.empty()

        if st.button("文字起こし開始"):
            # 新しい文字起こし開始時にseg_dfリセットフラグを立てる
            st.session_state["reset_seg_df"] = True
            try:
                pdf_summary = ""
                if pdf_file is not None:
                    pdf_summary = generate_keywords("gpt-4o-mini", pdf_file)
                    st.subheader("PDF要約")
                    st.text_area("要約", pdf_summary, height=200)
                log_messages = []
                log_messages.append("文字起こしを開始します。")
                log_placeholder.info("\n".join(log_messages))

                audio_format = uploaded_file.name.split(".")[-1]
                audio = AudioSegment.from_file(uploaded_file, format=audio_format)
                chunk_duration_ms = duration * 1000  # 15分 = 900000ミリ秒
                total_duration_ms = len(audio)
                num_chunks = (total_duration_ms + chunk_duration_ms - 1) // chunk_duration_ms

                results = []
                full_transcript = ""
                for i in range(num_chunks):
                    start_ms = i * chunk_duration_ms
                    end_ms = min((i + 1) * chunk_duration_ms, total_duration_ms)
                    chunk = audio[start_ms:end_ms]

                    log_messages.append(f"チャンク {i+1}/{num_chunks} を処理中...（{start_ms//1000}秒～{end_ms//1000}秒）")
                    log_placeholder.info("\n".join(log_messages))

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        chunk.export(tmp.name, format="wav")
                        file_tuple = (f"chunk_{i+1}.wav", open(tmp.name, "rb"), "audio/wav")

                        # initial_promptにPDF要約を使う
                        prompt_text = pdf_summary if pdf_summary else "こんにちは。\n\nはい、こんにちは。\n\nお元気ですか？\n\nはい、元気です。\n\nそれは何よりです。では早速始めましょう。\n\nはい、よろしくお願いいたします。"
                        transcript = client.audio.transcriptions.create(
                            model=model,
                            file=file_tuple,
                            language="ja",
                            prompt=prompt_text,
                            response_format="verbose_json"
                        )
                        transcript_dict = transcript.model_dump()
                        chunk_text = transcript_dict.get("text", "")
                        # セグメント情報も保存
                        segments = transcript_dict.get("segments", [])
                        results.append({
                            "chunk": i + 1,
                            "start_sec": start_ms // 1000,
                            "end_sec": end_ms // 1000,
                            "text": chunk_text,
                            "segments": segments
                        })
                        full_transcript += f"\n{chunk_text}"

                    log_messages.append(f"チャンク {i+1}/{num_chunks} の文字起こし完了。")
                    log_placeholder.info("\n".join(log_messages))

                log_messages.append("全チャンクの文字起こしが完了しました。")
                log_placeholder.success("\n".join(log_messages))

                # テーブル表示（seg_dfのみ表示・編集対象に変更）
                df = pd.DataFrame(results)
                # --- セッション保存 ---
                # 必要な情報をファイルに保存
                try:
                    os.makedirs("transcription", exist_ok=True)
                    st.session_state["results"] = results
                    st.session_state["full_transcript"] = full_transcript
                    # seg_dfは後で保存
                except Exception as e:
                    st.warning(f"セッションの保存に失敗しました: {e}")

                # セグメントJSONをテーブル表示
                all_segments = []
                for r in results:
                    if "segments" in r and r["segments"]:
                        for seg in r["segments"]:
                            seg = seg.copy()
                            seg["start"] += r["start_sec"]
                            seg["end"] += r["start_sec"]
                            all_segments.append(seg)
                if all_segments:
                    seg_df = pd.DataFrame(all_segments)
                    # start, end, text, speaker列を表示
                    seg_df = seg_df.loc[:, ["start", "end", "text"]]
                    # 既存のseg_dfがあればspeaker列を引き継ぐ
                    if "seg_df" in st.session_state and len(st.session_state["seg_df"]) == len(seg_df):
                        seg_df["speaker"] = st.session_state["seg_df"]["speaker"].values
                    else:
                        seg_df["speaker"] = ""

                    # --- セッション状態でseg_dfを管理 ---
                    # 文字起こし直後のみseg_dfを初期化
                    if "seg_df" not in st.session_state or st.session_state.get("reset_seg_df", False):
                        st.session_state["seg_df"] = seg_df.copy()
                        st.session_state["reset_seg_df"] = False
                    # --- seg_dfもセッションファイルに保存 ---
                    try:
                        session_data = {
                            "results": st.session_state.get("results", []),
                            "full_transcript": st.session_state.get("full_transcript", ""),
                            "seg_df": st.session_state["seg_df"].to_dict(orient="records") if "seg_df" in st.session_state else [],
                        }
                        with open(session_file, "w", encoding="utf-8") as f:
                            json.dump(session_data, f, ensure_ascii=False, indent=2)
                    except Exception as e:
                        st.warning(f"セッションの保存に失敗しました: {e}")

                else:
                    st.info("Whisper APIのセグメント情報（segments）は取得できませんでした。")

            except Exception as e:
                log_messages.append(f"エラーが発生しました: {str(e)}")
                log_placeholder.error("\n".join(log_messages))
    else:
        st.info("オーディオファイルをアップロードしてください。")

    # --- セッション状態があればセグメント選択・全文表示UIを常に表示 ---
    if "seg_df" in st.session_state:
        # data_editorのkeyをセッションで管理し、初期化
        if "seg_editor_key" not in st.session_state:
            import time
            st.session_state["seg_editor_key"] = str(time.time())
        st.subheader("文字起こし結果")

        # st.multiselectで行選択
        display_df = st.session_state["seg_df"].loc[:, ["speaker", "text", "start"]]
        def on_seg_df_change():
            # 編集内容をセッションファイルに保存
            try:
                session_data = {
                    "results": st.session_state.get("results", []),
                    "full_transcript": st.session_state.get("full_transcript", ""),
                    "seg_df": st.session_state["seg_df"].to_dict(orient="records"),
                }
                with open("transcription/last_session.json", "w", encoding="utf-8") as f:
                    json.dump(session_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                st.warning(f"data_editor編集内容の保存に失敗しました: {e}")

        edited_df = st.data_editor(
            display_df,
            num_rows="dynamic",
            use_container_width=True,
            key=st.session_state["seg_editor_key"],
            on_change=on_seg_df_change
        )
        st.session_state["seg_df"].loc[:, ["speaker", "text", "start"]] = edited_df

        # --- data_editorの下にマージ・分割ウィジェットを配置 ---
        # 行選択用のmultiselect
        selected_indices = st.multiselect(
            "複数行のテキストを1行に纏めることができます（マージ）。その場合、纏めたい行を選択してください（複数選択可）",
            options=list(st.session_state["seg_df"].index),
            format_func=lambda i: f"{st.session_state['seg_df'].loc[i, 'speaker']} | {st.session_state['seg_df'].loc[i, 'text'][:20]}...",
        )

        # seg_dfの編集内容をdfにも反映
        seg_df = st.session_state["seg_df"]
        df_from_seg = pd.DataFrame({
            "chunk": range(1, len(seg_df) + 1),
            "start_sec": seg_df["start"],
            "end_sec": seg_df["end"],
            "text": seg_df["text"],
            "segments": [None] * len(seg_df)
        })
        st.session_state["df"] = df_from_seg

        if st.button("選択した行をマージ"):
            if selected_indices:
                selected_rows = st.session_state["seg_df"].loc[selected_indices]
                selected = selected_rows.sort_values("start")
                merged_start = selected["start"].min()
                merged_end = selected["end"].max()
                merged_text = "".join(selected["text"].str.replace("\n", ""))
                # speaker列の値が全て同じならその値、異なる場合はカンマ区切りで全話者名を格納
                if selected["speaker"].nunique() == 1:
                    merged_speaker = selected["speaker"].iloc[0]
                else:
                    merged_speaker = ",".join([s for s in selected["speaker"].drop_duplicates() if s])
                # 新しい行を作成
                merged_row = pd.DataFrame([{
                    "speaker": merged_speaker,
                    "text": merged_text,
                    "start": merged_start,
                    "end": merged_end,
                }])
                # 選択行を削除し、纏めた行を元の最初の位置に挿入
                insert_at = min(selected_indices)
                dropped_df = st.session_state["seg_df"].drop(selected_indices)
                before = dropped_df.iloc[:insert_at]
                after = dropped_df.iloc[insert_at:]
                st.session_state["seg_df"] = pd.concat(
                    [before, merged_row, after],
                    ignore_index=True
                )
                # full_transcriptもseg_dfから再生成して更新
                st.session_state["full_transcript"] = "\n".join(st.session_state["seg_df"]["text"])
                # セッションファイルも再保存
                try:
                    session_data = {
                        "results": st.session_state.get("results", []),
                        "full_transcript": st.session_state["full_transcript"],
                        "seg_df": st.session_state["seg_df"].to_dict(orient="records"),
                    }
                    with open("transcription/last_session.json", "w", encoding="utf-8") as f:
                        json.dump(session_data, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    st.warning(f"セッションの保存に失敗しました: {e}")
                # data_editorのkeyを変更して強制再描画（マージ時のみ）
                import time
                st.session_state["seg_editor_key"] = str(time.time())
                st.success("選択した行をマージしてテーブルに反映しました。")
                st.dataframe(st.session_state["seg_df"])
                st.rerun()

        # --- 任意の場所に新しい行を挿入 ---
        seg_df = st.session_state["seg_df"]
        insert_options = []
        insert_labels = []

        # 先頭
        insert_options.append(0)
        insert_labels.append("先頭")
        # 各行の後
        for i in range(len(seg_df)):
            speaker = str(seg_df.iloc[i]["speaker"])
            text = str(seg_df.iloc[i]["text"])
            label = f"{i}: {speaker}｜{text[:20]}"
            insert_options.append(i+1)
            insert_labels.append(label)
        insert_position = st.selectbox(
            "1行のテキストを、指定位置で分割して2行にすることができます。分割する場合は、その対象の行を選んでください",
            options=insert_options,
            format_func=lambda i: insert_labels[insert_options.index(i)],
            index=1
        )
        split_keyword = st.text_input("分割キーワードを入力してください（当該の行を、マッチした箇所から分割します。キーワード自体も新しい行に含めます）", key="split_keyword")
        if st.button("新しい行を挿入", key="insert_row_button"):
            if insert_position > 0 and split_keyword:
                prev_row = st.session_state["seg_df"].iloc[insert_position-1]
                prev_text = str(prev_row["text"])
                prev_start = prev_row["start"]
                prev_end = prev_row["end"]
                idx = prev_text.find(split_keyword)
                if idx != -1:
                    before_text = prev_text[:idx]
                    after_text = prev_text[idx:]  # キーワード自体も含める
                    total_len = len(prev_text)
                    len_before = len(before_text)
                    len_after = len(after_text)
                    # 線形補間でstart/endを概算
                    if total_len > 0:
                        before_end = prev_start + (prev_end - prev_start) * (len_before / total_len)
                        after_start = before_end
                    else:
                        before_end = prev_start
                        after_start = prev_end
                    # 既存行を更新
                    st.session_state["seg_df"].iloc[insert_position-1, st.session_state["seg_df"].columns.get_loc("text")] = before_text
                    st.session_state["seg_df"].iloc[insert_position-1, st.session_state["seg_df"].columns.get_loc("end")] = before_end
                    # 新しい行
                    new_row = {"speaker": "", "text": after_text, "start": after_start, "end": prev_end}
                    before = st.session_state["seg_df"].iloc[:insert_position]
                    after = st.session_state["seg_df"].iloc[insert_position:]
                    st.session_state["seg_df"] = pd.concat(
                        [before, pd.DataFrame([new_row]), after],
                        ignore_index=True
                    )
                else:
                    new_row = {"speaker": "", "text": "", "start": prev_end, "end": prev_end}
                    before = st.session_state["seg_df"].iloc[:insert_position]
                    after = st.session_state["seg_df"].iloc[insert_position:]
                    st.session_state["seg_df"] = pd.concat(
                        [before, pd.DataFrame([new_row]), after],
                        ignore_index=True
                    )
            else:
                new_row = {"speaker": "", "text": "", "start": 0, "end": 0}
                before = st.session_state["seg_df"].iloc[:insert_position]
                after = st.session_state["seg_df"].iloc[insert_position:]
                st.session_state["seg_df"] = pd.concat(
                    [before, pd.DataFrame([new_row]), after],
                    ignore_index=True
                )
            # 追加後にセッションファイルへ保存
            try:
                session_data = {
                    "results": st.session_state.get("results", []),
                    "full_transcript": st.session_state.get("full_transcript", ""),
                    "seg_df": st.session_state["seg_df"].to_dict(orient="records"),
                }
                with open("transcription/last_session.json", "w", encoding="utf-8") as f:
                    json.dump(session_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                st.warning(f"セッションの保存に失敗しました: {e}")
            import time
            st.session_state["seg_editor_key"] = str(time.time())
            st.success(f"{insert_position}番目に新しい行を挿入しました。")
            st.rerun()
        # 纏めたテーブルをダウンロード
        st.subheader("纏めたセグメントテーブルをダウンロード")

        # 全セグメントを1ファイルでダウンロード（話者名付き）
        all_text_lines = []
        for _, row in st.session_state["seg_df"].iterrows():
            speaker = f"\n（{row['speaker']}）\n" if str(row['speaker']).strip() != "" else ""
            text = str(row['text']).strip()
            if text != "":
                if speaker == "":
                    # 前の行に追記
                    if all_text_lines:
                        all_text_lines[-1] += text
                    else:
                        all_text_lines.append(text)
                else:
                    all_text_lines.append(f"{speaker}{text}")
        all_text = "\n".join(all_text_lines)
        st.download_button(
            label="TXTでダウンロード(speaker付き)",
            data=all_text.encode("utf-8"),
            file_name="all_speaker_text.txt",
            mime="text/plain"
        )
        # speaker列を()で囲った形式でCSV出力
        csv_df = st.session_state["seg_df"].copy()
        csv_df["speaker"] = csv_df["speaker"].apply(lambda x: f"（{x}）" if pd.notnull(x) and str(x).strip() != "" else "")
        csv_df = csv_df.loc[:, ["speaker", "text", "start", "end"]]
        seg_csv_bytes = csv_df.to_csv(index=False).encode("shift_jis")
        st.download_button(
            label="CSVでダウンロード",
            data=seg_csv_bytes,
            file_name="merged_segments.csv",
            mime="text/csv"
        )

        seg_json_bytes = st.session_state["seg_df"].to_json(orient="records", force_ascii=False).encode("utf-8")
        st.download_button(
            label="JSONでダウンロード",
            data=seg_json_bytes,
            file_name="merged_segments.json",
            mime="application/json"
        )
        st.subheader("文字起こし結果（全文）")
        st.text_area("結果", st.session_state["full_transcript"].strip(), height=400)
        st.download_button(
            label="TXTで全文ダウンロード",
            data=st.session_state["full_transcript"].strip().encode("utf-8"),
            file_name="plain_transcript.txt",
            mime="text/plain"
        )

def main():
    # サイドバーの幅を広げるカスタムCSSを挿入
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            width: 500px;
        }
        /* Deployボタン非表示 */
        [data-testid="stDeployButton"] {
            display: none !important;
        }
        /* 3点リーダ（ツールバー）非表示 */
        [data-testid="stHeader"] [data-testid="stToolbar"] {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    app_selection = st.sidebar.selectbox("文字起こしライブラリまたはアプリを選択", ["whisper"])
    duration = st.sidebar.number_input("1推論当たりの時間(sec)", min_value=0, max_value=1800, value=180, step=1)
    #offset = st.sidebar.number_input("推論単位の重複させる時間(sec)", min_value=0, max_value=300,value=10, step=1)

    if app_selection == "whisper":
        mojiokoshi(duration, offset=0)

if __name__ == "__main__":
    main()
