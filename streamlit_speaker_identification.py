import streamlit as st
from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np
import pandas as pd
from pydub import AudioSegment
import io
import tempfile
import os
from pathlib import Path
import zipfile

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
    """
    Identifies speakers for audio segments defined in a DataFrame.

    Args:
        audio_file: Uploaded audio file object.
        df: DataFrame with 'start' and 'end' columns (in seconds).
        uploaded_embedding_files: List of uploaded embedding file objects.

    Returns:
        DataFrame with an added 'speaker' column.
    """
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

st.sidebar.header("モード選択")
mode = st.sidebar.radio("機能を選択してください:", ("話者識別", "埋め込み抽出"))

if mode == "話者識別":
    st.title("話者識別のアプリ")

    st.write("話者を識別するために、音声ファイルとセグメントのタイムスタンプを含むCSVをアップロードしてください。")

    uploaded_audio_file = st.file_uploader("音声ファイルをアップロード", type=["wav", "mp3", "flac", "ogg", "mp4"], key="identify_audio_uploader")
    uploaded_embedding_files = st.file_uploader("話者埋め込みファイルをアップロード（複数選択可）", type=["npy"], accept_multiple_files=True, key="identify_embeddings_uploader")
    uploaded_csv_file = st.file_uploader("CSVファイルをアップロード（'start'と'end'列が秒単位であること）", type=["csv"], key="identify_csv_uploader")

    similarity_threshold = st.number_input(
        "話者識別の類似度閾値",
        min_value=0.0,
        max_value=1.0,
        value=0.7, # Default threshold
        step=0.01,
        help="この閾値以下の類似度の場合、話者は「判定不可」として空欄になります。"
    )

    if uploaded_audio_file is not None and uploaded_csv_file is not None and uploaded_embedding_files is not None:
        try:
            dataframe_segments = pd.read_csv(uploaded_csv_file, encoding="shift-jis", on_bad_lines='skip')

            if 'start' not in dataframe_segments.columns or 'end' not in dataframe_segments.columns or 'text' not in dataframe_segments.columns:
                st.error("CSVファイルには'start'列、'end'列、および'text'列を含める必要があります。")
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

                # Use the edited dataframe for CSV output
                csv_output = st.session_state.edited_result_df.to_csv(index=False).encode('shift-jis')
                st.download_button(
                    label="結果をCSVとしてダウンロード",
                    data=csv_output,
                    file_name='speaker_identified_segments.csv',
                    mime='text/csv',
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

    st.sidebar.header("使い方（話者識別）")
    st.sidebar.markdown("""
    1. **話者埋め込みを準備する:** 話者ごとに埋め込みファイル（`.npy`形式）を用意します。これらのファイルは、事前に音声データから「話者埋め込み作成」で生成したものです。
    2. **セグメントCSVを準備する:** 少なくとも2つの列（`start`と`end`）を持つCSVファイルを作成します。これらの列には、各音声セグメントの開始時刻と終了時刻を秒単位で含める必要があります。
    3. **ファイルをアップロードする:** ファイルアップローダーを使用して、会議音声、話者埋め込み、セグメントCSVを提供します。
    4. **識別を実行する:** 3種類のファイルがアップロードされると、アプリは自動的に処理を開始します。
    5. **結果を表示してダウンロードする:** 'speaker'列が追加された結果のDataFrameが表示され、CSVとしてダウンロードできます。
    """)

elif mode == "埋め込み抽出":
    st.title("音声埋め込み抽出")
    st.write("音声ファイルとセグメントのタイムスタンプを含むCSVをアップロードして、各セグメントの埋め込みを抽出します。")

    uploaded_audio_file_extract = st.file_uploader("音声ファイルをアップロード", type=["wav", "mp3", "flac", "ogg", "mp4"], key="extract_audio_uploader")
    uploaded_csv_file_extract = st.file_uploader("CSVファイルをアップロード（'start'と'end'列が秒単位であること）", type=["csv"], key="extract_csv_uploader")

    if uploaded_audio_file_extract is not None and uploaded_csv_file_extract is not None:
        if st.button("埋め込みを抽出"):
            with st.spinner("埋め込みを抽出中..."):
                try:
                    dataframe_segments = pd.read_csv(uploaded_csv_file_extract, encoding="shift-jis", on_bad_lines='skip')

                    if 'start' not in dataframe_segments.columns or 'end' not in dataframe_segments.columns or 'text' not in dataframe_segments.columns:
                        st.error("CSVファイルには'start'列、'end'列、および'text'列を含める必要があります。")
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
                                zip_buffer = io.BytesIO()
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
                    st.error(f"CSVファイルの読み込み中にエラーが発生しました: {e}")

    st.sidebar.header("使い方（埋め込み抽出）")
    st.sidebar.markdown("""
    1. **音声ファイルをアップロードする:** 埋め込みを抽出したいメインの音声ファイルをアップロードします。
    2. **CSVファイルをアップロードする:** 各音声セグメントの開始時刻と終了時刻（秒単位）を含むCSVファイルをアップロードします。
    3. **埋め込みを抽出ボタンをクリックする:** 処理が開始され、CSVで定義された各セグメントの埋め込みが抽出され、個別の.npyファイルとして一時的に保存されます。
    4. **ファイルをダウンロードする:** 抽出が完了すると、生成されたすべての埋め込みを含むZIPファイルをダウンロードできます。
    """)
