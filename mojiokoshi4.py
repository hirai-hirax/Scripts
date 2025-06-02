import os
import streamlit as st
from openai import AzureOpenAI
import tempfile
from io import BytesIO
from pydub import AudioSegment
import fitz
import pandas as pd
from dotenv import load_dotenv
from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np
from pathlib import Path
import zipfile

# .envファイルから環境変数を読み込む
load_dotenv()

# 環境変数から設定を取得（Azure OpenAI のエンドポイント・API キーを設定してください）
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
API_VERSION = "2023-09-01-preview"  # ご利用の API バージョンに合わせてください

summarizing_prompt1 = """
    ユーザーからテキストを渡されます。当該のテキストの内容を読んだ上で、150文字程度の要約を生成してください。
"""

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

def generate_summary(model, prompt, text):
    # クライアント初期化
    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=API_VERSION,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ],
    )
    print(f"Response: {response}")
    return response.choices[0].message.content

def transcribe_audio_to_dataframe(uploaded_file: BytesIO, duration: int, pdf_file: BytesIO = None):
    model = "whisper"
    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=API_VERSION
    )

    pdf_summary = ""
    if pdf_file is not None:
        pdf_text = get_text_from_pdf(pdf_file)
        pdf_summary = generate_summary("gpt-4o-mini",
                                        summarizing_prompt1,
                                        pdf_text)

    audio_format = uploaded_file.name.split(".")[-1]
    audio = AudioSegment.from_file(uploaded_file, format=audio_format)
    chunk_duration_ms = duration * 1000
    total_duration_ms = len(audio)
    num_chunks = (total_duration_ms + chunk_duration_ms - 1) // chunk_duration_ms

    all_segments = []
    # Use TemporaryDirectory for more robust cleanup
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(num_chunks):
            start_ms = i * chunk_duration_ms
            end_ms = min((i + 1) * chunk_duration_ms, total_duration_ms)
            chunk = audio[start_ms:end_ms]

            # st.info(f"チャンク {i+1}/{num_chunks} を処理中...（{start_ms//1000}秒～{end_ms//1000}秒）") # Streamlit UI removed

            # Create a unique temporary file path within the temporary directory
            temp_audio_path = os.path.join(tmpdir, f"chunk_{i+1}.wav")

            chunk.export(temp_audio_path, format="wav")

            # Open the file by path for reading
            with open(temp_audio_path, "rb") as audio_file_for_api:
                file_tuple = (f"chunk_{i+1}.wav", audio_file_for_api, "audio/wav")

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
                segments = transcript_dict.get("segments", [])

                for seg in segments:
                    seg = seg.copy()
                    seg["start"] += start_ms // 1000
                    seg["end"] += start_ms // 1000
                    all_segments.append(seg)

            # The temporary file will be automatically removed when the TemporaryDirectory context exits

        # st.success(f"チャンク {i+1}/{num_chunks} の文字起こし完了。") # Streamlit UI removed

    if all_segments:
        seg_df = pd.DataFrame(all_segments)
        # start, end, text列を表示
        seg_df = seg_df.loc[:, ["start", "end", "text"]]
        
        # Insert an empty 'speaker' column between 'end' and 'text'
        text_col_index = seg_df.columns.get_loc("text")
        seg_df.insert(text_col_index, "speaker", "")
        
        return seg_df
    else:
        # Return empty DataFrame with 'speaker' column
        return pd.DataFrame(columns=["start", "end", "speaker", "text"])

@st.cache_resource
def load_voice_encoder():
    """Caches the VoiceEncoder model."""
    return VoiceEncoder()

@st.cache_data
def extract_embedding(audio_content):
    """Extracts embedding from audio content."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav_file:
        temp_wav_file.write(audio_content.read())
        temp_wav_file_path = temp_wav_file.name

    try:
        wav = preprocess_wav(temp_wav_file_path)
        encoder = load_voice_encoder()
        embedding = encoder.embed_utterance(wav)
        return embedding
    finally:
        os.remove(temp_wav_file_path)

@st.cache_data
def load_speaker_embeddings_from_files(uploaded_files):
    """Loads known speaker embeddings from uploaded files."""
    speaker_embeddings = {}
    if not uploaded_files:
        st.warning("話者埋め込みファイルがアップロードされていません。")
        return speaker_embeddings

    for uploaded_file in uploaded_files:
        try:
            # Extract speaker name from filename (without extension)
            speaker_name = Path(uploaded_file.name).stem
            # Load the embedding from the uploaded file
            embedding = np.load(uploaded_file)
            speaker_embeddings[speaker_name] = embedding
        except Exception as e:
            st.error(f"埋め込みファイルの読み込み中にエラーが発生しました {uploaded_file.name}: {e}")
    return speaker_embeddings

@st.cache_data
def identify_speakers_in_dataframe(audio_file, df: pd.DataFrame, uploaded_embedding_files, similarity_threshold: float) -> pd.DataFrame:
    known_embeddings = load_speaker_embeddings_from_files(uploaded_embedding_files)
    if not known_embeddings:
        st.warning("既知の話者埋め込みが見つかりませんでした。識別を実行できません。")
        df['speaker'] = None
        return df

    st.info(f"Loaded embeddings for speakers: {list(known_embeddings.keys())}")

    # Save uploaded audio to a temporary file for pydub
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
        temp_audio_file.write(audio_file.getvalue())
        temp_audio_file_path = temp_audio_file.name

    try:
        audio = AudioSegment.from_file(temp_audio_file_path)
        df['speaker'] = None # Initialize speaker column
        encoder = load_voice_encoder()

        progress_bar = st.progress(0)
        status_text = st.empty()

        for index, row in df.iterrows():
            start_time_ms = row['start'] * 1000
            end_time_ms = row['end'] * 1000

            # Extract audio segment
            segment = audio[start_time_ms:end_time_ms]

            # Save the segment to a temporary file for preprocess_wav
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_segment_file:
                segment.export(temp_segment_file.name, format="wav")
                temp_segment_file_path = temp_segment_file.name

            try:
                wav = preprocess_wav(temp_segment_file_path)
                segment_embedding = encoder.embed_utterance(wav)

                highest_similarity = -1
                identified_speaker = None

                for speaker_name, known_embedding in known_embeddings.items():
                    similarity = np.dot(segment_embedding, known_embedding) / (np.linalg.norm(segment_embedding) * np.linalg.norm(known_embedding))
                    print(f"Comparing segment with {speaker_name}: similarity = {similarity:.4f}")
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        identified_speaker = speaker_name

                if highest_similarity >= similarity_threshold:
                    df.at[index, 'speaker'] = identified_speaker
                    status_text.text(f"Processed segment {index + 1}/{len(df)}: Identified as {identified_speaker}")

                    # --- Add code here to save the embedding ---
                    # Get the first 10 characters of the text and sanitize
                    text_snippet_10 = str(row['text'])[:10].replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')
                    # Construct the filename
                    output_filename = f"{identified_speaker}_{text_snippet_10}.npy"
                    # Define the full path
                    output_filepath = os.path.join("embed", output_filename)
                    # Save the embedding
                    try:
                        np.save(output_filepath, segment_embedding)
                        print(f"Saved embedding for {identified_speaker} as {output_filepath}")
                    except Exception as save_e:
                        st.error(f"Error saving embedding file {output_filepath}: {save_e}")
                    # --- End of added code ---

                else:
                    df.at[index, 'speaker'] = ""
                    status_text.text(f"Processed segment {index + 1}/{len(df)}: Similarity below threshold, speaker not identified.")

            except Exception as e:
                st.error(f"Error processing segment {row['start']}-{row['end']}s: {e}")
                df.at[index, 'speaker'] = "Error" # Mark segments that failed
            finally:
                os.remove(temp_segment_file_path)

            progress_bar.progress((index + 1) / len(df))

        status_text.text("Speaker identification complete.")
        return df

    except Exception as e:
        st.error(f"Error loading or processing audio file: {e}")
        return df # Return original df in case of audio error
    finally:
        os.remove(temp_audio_file_path)

def mojiokoshi(duration):
    st.title("オーディオファイルの文字起こし")
    st.write("会議資料とオーディオファイルをアップロードしてください。")  

    st.sidebar.write("""
    このアプリは、アップロードされた音声ファイルを文字起こしし、必要に応じて話者識別を行います。
    また、文字起こし結果を整形して議事録形式に変換する機能も含まれています。
    """)

    # PDFファイルアップロード
    pdf_file = st.file_uploader("要約に使うPDFファイルを選択", type=["pdf"])
    # オーディオファイルアップロード
    uploaded_file = st.file_uploader("オーディオファイルを選択", type=["mp3", "wav", "m4a", "mp4", "webm", "ogg"])

    seg_df = pd.DataFrame() # Initialize seg_df outside the button click

    if uploaded_file is not None:
        uploaded_file_name = uploaded_file.name # Store the uploaded file name
        st.write(f"アップロードされたファイル名: {uploaded_file_name}")

        if st.button("文字起こし開始"):
            try:
                # Call the new function to get the DataFrame
                seg_df = transcribe_audio_to_dataframe(uploaded_file, duration, pdf_file)

                st.subheader("文字起こし結果")
                if not seg_df.empty:
                    st.dataframe(seg_df)
                else:
                    st.info("文字起こし結果がありませんでした。")

            except Exception as e:
                st.error(f"エラーが発生しました: {str(e)}")
    else:
        st.info("オーディオファイルをアップロードしてください。")

    # Add download button for Excel after the dataframe is potentially created
    if not seg_df.empty:
        # Generate Excel file content
        excel_buffer = BytesIO()
        seg_df.to_excel(excel_buffer, index=False)
        excel_buffer.seek(0)

        # Determine the download file name
        if uploaded_file is not None:
             # Remove extension and add .xlsx
            base_name = os.path.splitext(uploaded_file.name)[0]
            download_file_name = f"{base_name}.xlsx"
        else:
            download_file_name = "transcription_result.xlsx" # Fallback name

        st.download_button(
            label="Download as Excel",
            data=excel_buffer,
            file_name=download_file_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

def gijiroku_seikei():
    st.title("議事録整形アプリ")
    st.sidebar.write("「文字起こし」アプリで生成した文字起こしExcelファイルに、手動で話者記入を行ってから、このアプリに記入済みのExcelファイルをアップロードしてください。")

    st.sidebar.write("""
    連続する同じ話者の発言を結合し、議事録として読みやすい形式に変換します。
    変換後のデータを再びExcelファイルとしてダウンロードできます。
    """)

    uploaded_excel_file = st.file_uploader(
        "話者記入済みExcelファイルを選択してください",
        type=["xlsx"],
        key="upload_excel_for_merge" # keyを追加
    )

    if uploaded_excel_file is not None:
        st.success("Excelファイルがアップロードされました。")

        try:
            df_original = pd.read_excel(uploaded_excel_file)

            st.subheader("アップロードされたExcelファイルのプレビュー")
            st.dataframe(df_original)

            if st.button("議事録を整形", key="merge_button"): # keyを追加
                if 'speaker' not in df_original.columns or 'text' not in df_original.columns or \
                   'start' not in df_original.columns or 'end' not in df_original.columns:
                    st.error("Excelファイルに必要な列 ('speaker', 'text', 'start', 'end') が含まれていません。")
                else:
                    with st.spinner("議事録整形中..."):
                        df_processed = df_original.copy()

                        # 1. 話者列の前処理: 空欄を前後の話者で埋める
                        df_processed['speaker_filled'] = df_processed['speaker'].replace('', pd.NA)
                        df_processed['speaker_filled'] = df_processed['speaker_filled'].bfill().ffill()

                        # 2. グループ化キーの作成: 話者が変わるごとに新しいグループIDを割り当てる
                        #    NaNが埋められた後のspeaker_filled列で比較
                        df_processed['group_id'] = (df_processed['speaker_filled'] != df_processed['speaker_filled'].shift()).cumsum()

                        # 3. データの集約: グループごとにテキストを結合し、開始・終了時刻を調整
                        df_merged = df_processed.groupby('group_id').agg(
                            start=('start', 'min'),
                            end=('end', 'max'),
                            speaker=('speaker_filled', 'first'), # グループの最初の話者を採用
                            text=('text', ' '.join)
                        ).reset_index(drop=True) # 一時的なgroup_id列を削除

                        st.subheader("整形された議事録 (プレビュー)")
                        st.dataframe(df_merged)

                        st.subheader("整形された議事録 (JSON形式)")
                        st.json(df_merged.to_dict(orient='records'))

                        output_excel = BytesIO()
                        with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
                            df_merged.to_excel(writer, index=False, sheet_name="整形後議事録")
                        processed_excel_data = output_excel.getvalue()

                        st.download_button(
                            label="整形済みExcelファイルをダウンロード",
                            data=processed_excel_data,
                            file_name=f"{os.path.splitext(uploaded_excel_file.name)[0]}_整形済み議事録.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="download_merged_excel" # keyを追加
                        )
                        st.success("議事録の整形が完了しました！")

        except Exception as e:
            st.error(f"ファイルの処理中にエラーが発生しました: {e}")
            st.info("アップロードされたExcelファイルが正しい形式であり、必要な列 ('speaker', 'text', 'start', 'end') が含まれているか確認してください。")

def generate_embeddings():
    st.title("音声埋め込み抽出")
    st.write("音声ファイルとセグメントのタイムスタンプを含むExcelをアップロードして、各セグメントの埋め込みを抽出します。")

    st.sidebar.write("""
    この機能では、音声ファイルとセグメント情報を含むExcelファイルから、各セグメントの音声埋め込みを抽出します。
    セグメント情報を含むExcelファイルは、「議事録整形」アプリで生成したものを使うことを想定しています(少なくとも'start', 'end', 'text'列が必要です)。
    抽出された埋め込みは、「文字起こしに話者情報を追加」アプリにに利用できます。
    """)

    uploaded_audio_file_extract = st.file_uploader("音声ファイルをアップロード", type=["wav", "mp3", "flac", "ogg", "mp4"], key="extract_audio_uploader")
    uploaded_excel_file_extract = st.file_uploader("Excelファイルをアップロード（'start'と'end'列が秒単位であること）", type=["xlsx"], key="extract_excel_uploader")

    if uploaded_audio_file_extract is not None and uploaded_excel_file_extract is not None:
        if st.button("埋め込みを抽出"):
            with st.spinner("埋め込みを抽出中..."):
                try:
                    dataframe_segments = pd.read_excel(uploaded_excel_file_extract)

                    if 'start' not in dataframe_segments.columns or 'end' not in dataframe_segments.columns or 'text' not in dataframe_segments.columns:
                        st.error("Excelファイルには'start'列、'end'列、および'text'列を含める必要があります。")
                    else:
                        # Save uploaded audio to a temporary file for pydub
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
                            temp_audio_file.write(uploaded_audio_file_extract.getvalue())
                            temp_audio_file_path = temp_audio_file.name

                        try:
                            audio = AudioSegment.from_file(temp_audio_file_path)
                            encoder = load_voice_encoder()

                            generated_embeddings = []

                            progress_bar = st.progress(0)
                            status_text = st.empty()

                            for index, row in dataframe_segments.iterrows():
                                start_time_ms = row['start'] * 1000
                                end_time_ms = row['end'] * 1000

                                # Extract audio segment
                                segment = audio[start_time_ms:end_time_ms]

                                # Save the segment to a temporary file for preprocess_wav
                                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_segment_file:
                                    segment.export(temp_segment_file.name, format="wav")
                                    temp_segment_file_path = temp_segment_file.name

                                try:
                                    wav = preprocess_wav(temp_segment_file_path)
                                    segment_embedding = encoder.embed_utterance(wav)

                                    # Determine output filename based on index and text column
                                    text_snippet = str(row['text'])[:20].replace('/', '_').replace('\\', '_') # Get first 20 chars and sanitize
                                    output_filename = f"{index}_{text_snippet}.npy"

                                    np.save(output_filename, segment_embedding)
                                    generated_embeddings.append((output_filename, segment_embedding))

                                    status_text.text(f"Processed segment {index + 1}/{len(dataframe_segments)}: Saved as {output_filename}")

                                except Exception as e:
                                    st.error(f"セグメント {row['start']}-{row['end']}s の処理中にエラーが発生しました: {e}")
                                    # Optionally mark this segment as failed in the list
                                    generated_embeddings.append((f"segment_{index}_error.npy", None))
                                finally:
                                    os.remove(temp_segment_file_path)

                                progress_bar.progress((index + 1) / len(dataframe_segments))

                            status_text.text("埋め込み抽出が完了しました。")

                            st.subheader("生成された埋め込みファイル:")
                            if generated_embeddings:
                                # Create a zip file in memory
                                zip_buffer = BytesIO()
                                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                                    for filename, embedding in generated_embeddings:
                                        if embedding is not None:
                                            zip_file.write(filename, arcname=filename)
                                            os.remove(filename) # Clean up the individual file after adding to zip

                                zip_buffer.seek(0)

                                st.download_button(
                                    label="埋め込みファイルをまとめてダウンロード (ZIP)",
                                    data=zip_buffer,
                                    file_name='generated_embeddings.zip',
                                    mime='application/zip',
                                )
                            else:
                                st.text("生成された埋め込みファイルはありません。")


                        except Exception as e:
                            st.error(f"音声ファイルの処理中にエラーが発生しました: {e}")
                        finally:
                            os.remove(temp_audio_file_path)

                except Exception as e:
                    st.error(f"Excelファイルの読み込み中にエラーが発生しました: {e}")

def speaker_identification_in_mojiokoshi():
    st.title("話者識別のアプリ")

    st.write("文字起こし結果に話者識別結果を記入します。音声ファイル、話者埋め込みファイル、文字起こし結果のExcelをアップロードしてください。")

    st.sidebar.write("""
    この機能では、文字起こし結果のExcelファイルと話者埋め込みファイルを使用して、文字起こし結果に話者情報を追加します。
    アップロードされた音声ファイルと埋め込みを比較し、各セグメントの話者を識別します。
    話者埋め込みファイルは、「話者埋め込み作成」アプリで生成できます。
    """)

    uploaded_audio_file = st.file_uploader("音声ファイルをアップロード", type=["wav", "mp3", "flac", "ogg", "mp4"], key="identify_audio_uploader")
    uploaded_embedding_files = st.file_uploader("話者埋め込みファイルをアップロード（複数選択可）", type=["npy"], accept_multiple_files=True, key="identify_embeddings_uploader")
    uploaded_excel_file = st.file_uploader("Excelファイルをアップロード（'start'と'end'列が秒単位であること）", type=["xlsx"], key="identify_excel_uploader")

    similarity_threshold = st.number_input(
        "話者識別の類似度閾値",
        min_value=0.0,
        max_value=1.0,
        value=0.7, # Default threshold
        step=0.01,
        help="この閾値以下の類似度の場合、話者は「判定不可」として空欄になります。"
    )

    if uploaded_audio_file is not None and uploaded_excel_file is not None and uploaded_embedding_files is not None:
        try:
            dataframe_segments = pd.read_excel(uploaded_excel_file)

            if 'start' not in dataframe_segments.columns or 'end' not in dataframe_segments.columns or 'text' not in dataframe_segments.columns:
                st.error("Excelファイルには'start'列、'end'列、および'text'列を含める必要があります。")
            else:
                st.write("話者を識別中...")
                # Store result_df in session state
                st.session_state.result_df = identify_speakers_in_dataframe(uploaded_audio_file, dataframe_segments.copy(), uploaded_embedding_files, similarity_threshold)

                st.write("結果:")

                # Callback function to update the dataframe in session state and regenerate transcript
                def update_dataframe():
                    st.session_state.result_df = st.session_state.edited_result_df
                    # Regenerate transcript
                    transcript_lines = []
                    current_speaker = None
                    current_text_block = []

                    for index, row in st.session_state.result_df.iterrows():
                        speaker = row['speaker']
                        text = str(row['text']) # Ensure text is string

                        # Handle None or empty speaker by treating it as the previous speaker
                        if speaker is None or str(speaker).strip() == "":
                            if current_speaker is not None:
                                # Append to the current speaker's block
                                current_text_block.append(text)
                            else:
                                # If no current speaker (first segment is unknown), start a new block without speaker
                                if current_text_block: # If there's already text in the block (from previous unknown speakers)
                                     current_text_block.append(text)
                                else: # First segment and unknown speaker
                                     current_text_block.append(text)
                                # current_speaker remains None

                        elif current_speaker is None:
                             # First segment with a known speaker
                             current_speaker = speaker
                             current_text_block.append(text)

                        elif speaker != current_speaker:
                            # Speaker changed, finalize the previous block
                            if current_text_block:
                                if current_speaker is not None:
                                    transcript_lines.append(f"({current_speaker}) {' '.join(current_text_block)}")
                                else:
                                    transcript_lines.append(' '.join(current_text_block)) # Should not happen based on logic, but as a fallback
                            
                            # Start a new block with the new speaker
                            current_speaker = speaker
                            current_text_block = [text]

                        else: # Same speaker as previous
                            current_text_block.append(text)

                    # Finalize the last block after the loop
                    if current_text_block:
                        if current_speaker is not None:
                            transcript_lines.append(f"({current_speaker}) {' '.join(current_text_block)}")
                        else:
                            transcript_lines.append(' '.join(current_text_block))

                    st.session_state.transcript_content = "\n".join(transcript_lines)


                # Use st.data_editor with on_change callback
                st.session_state.edited_result_df = st.data_editor(
                    st.session_state.result_df,
                    use_container_width=True,
                    hide_index=True,
                    on_change=update_dataframe,
                    key='result_df_editor' # Add a key for the data editor
                )

                # Use the edited dataframe for Excel output
                excel_buffer = BytesIO()
                st.session_state.edited_result_df.to_excel(excel_buffer, index=False)
                excel_buffer.seek(0)

                st.download_button(
                    label="結果をExcelとしてダウンロード",
                    data=excel_buffer,
                    file_name='speaker_identified_segments.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                )

                # Regenerate transcript using the edited dataframe
                transcript_lines = []
                current_speaker = None
                current_text_block = []

                for index, row in st.session_state.edited_result_df.iterrows():
                    speaker = row['speaker']
                    text = str(row['text']) # Ensure text is string

                    # Handle None or empty speaker by treating it as the previous speaker
                    if speaker is None or str(speaker).strip() == "":
                        if current_speaker is not None:
                            # Append to the current speaker's block
                            current_text_block.append(text)
                        else:
                            # If no current speaker (first segment is unknown), start a new block without speaker
                            if current_text_block: # If there's already text in the block (from previous unknown speakers)
                                 current_text_block.append(text)
                            else: # First segment and unknown speaker
                                 current_text_block.append(text)
                            # current_speaker remains None

                    elif current_speaker is None:
                         # First segment with a known speaker
                         current_speaker = speaker
                         current_text_block.append(text)

                    elif speaker != current_speaker:
                        # Speaker changed, finalize the previous block
                        if current_text_block:
                            if current_speaker is not None:
                                transcript_lines.append(f"({current_speaker}) {' '.join(current_text_block)}")
                            else:
                                transcript_lines.append(' '.join(current_text_block)) # Should not happen based on logic, but as a fallback
                        
                        # Start a new block with the new speaker
                        current_speaker = speaker
                        current_text_block = [text]

                    else: # Same speaker as previous
                        current_text_block.append(text)

                # Finalize the last block after the loop
                if current_text_block:
                    if current_speaker is not None:
                        transcript_lines.append(f"({current_speaker}) {' '.join(current_text_block)}")
                    else:
                        transcript_lines.append(' '.join(current_text_block))

                transcript_content = "\n".join(transcript_lines)

                st.download_button(
                    label="議事録テキストファイルをダウンロード",
                    data=transcript_content.encode('utf-8'),
                    file_name='meeting_transcript.txt',
                    mime='text/plain',
                )

        except Exception as e:
            st.error(f"エラーが発生しました: {e}")

def generate_transcript_text():
    st.title("発言録テキスト生成")
    st.sidebar.write("「文字起こしExcelの整え」で作成したExcelファイルをアップロードしてください。"
                     "Excelファイルから、発言録テキストを生成します。")

    uploaded_excel_file = st.file_uploader(
        "Excelファイルを選択してください",
        type=["xlsx"],
        key="upload_excel_for_transcript"
    )

    # Add a selectbox for format options
    format_option = st.selectbox(
        "話者表示形式を選択してください",
        ["（話者）テキスト", "（話者）テキスト/前後改行あり", "話者＞テキスト", "話者：テキスト"],
        index=0, # Default to the first option
        key="transcript_format_selector"
    )

    if uploaded_excel_file is not None:
        st.success("Excelファイルがアップロードされました。")

        try:
            df = pd.read_excel(uploaded_excel_file)

            if 'speaker' not in df.columns or 'text' not in df.columns:
                st.error("Excelファイルに必要な列 ('speaker', 'text') が含まれていません。")
            else:
                transcript_lines = []
                for index, row in df.iterrows():
                    speaker = row.get('speaker', '') # Use .get for safety
                    text = row.get('text', '') # Use .get for safety

                    # Ensure speaker and text are strings and handle potential NaN
                    speaker_str = str(speaker) if pd.notna(speaker) else ""
                    text_str = str(text) if pd.notna(text) else ""

                    # Format the line based on the selected option
                    if speaker_str:
                        if format_option == "（話者）テキスト":
                            transcript_lines.append(f"（{speaker_str}）{text_str}")
                        elif format_option == "（話者）テキスト/前後改行あり":
                            transcript_lines.append(f"\n（{speaker_str}）\n{text_str}")
                        elif format_option == "話者＞テキスト":
                            transcript_lines.append(f"{speaker_str}＞{text_str}")
                        elif format_option == "話者 テキスト":
                            transcript_lines.append(f"{speaker_str}：{text_str}")
                        else:
                            # Fallback to default format
                            transcript_lines.append(f"（{speaker_str}）{text_str}")
                    else:
                        # If no speaker, just include the text
                        transcript_lines.append(text_str)


                transcript_content = "\n".join(transcript_lines)

                st.subheader("生成された発言録テキスト")
                st.text_area("プレビュー", transcript_content, height=300)

                st.download_button(
                    label="発言録テキストファイルをダウンロード",
                    data=transcript_content.encode('utf-8'),
                    file_name='meeting_transcript.txt',
                    mime='text/plain',
                    key="download_transcript_text"
                )
                st.success("発言録テキストの生成が完了しました！")

        except Exception as e:
            st.error(f"ファイルの処理中にエラーが発生しました: {e}")
            st.info("アップロードされたExcelファイルが正しい形式であり、必要な列 ('speaker', 'text') が含まれているか確認してください。")

def main():
    st.set_page_config(layout="wide")
    mode = st.sidebar.selectbox(
        "アプリケーションを選択",
        ["(1)文字起こしExcelの生成", "(2)文字起こしExcelの整え", "(3)発言録テキスト生成","(a)話者埋め込み作成","(b)文字起こしに話者情報を追加",]
    )

    duration = st.sidebar.number_input("1推論当たりの時間(sec)", min_value=0, max_value=1800, value=600, step=1)
    if mode == "(1)文字起こしExcelの生成":
        mojiokoshi(duration)
    elif mode == "(2)文字起こしExcelの整え":
        gijiroku_seikei()
    elif mode == "(3)発言録テキスト生成":
        generate_transcript_text()
    elif mode == "(a)話者埋め込み作成":
        generate_embeddings()
    elif mode == "(b)文字起こしに話者情報を追加":
        speaker_identification_in_mojiokoshi()

if __name__ == "__main__":
    main()
