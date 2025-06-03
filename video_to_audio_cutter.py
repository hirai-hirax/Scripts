import streamlit as st
import subprocess
import os
import tempfile
from datetime import timedelta
import pandas as pd
import zipfile
from io import BytesIO

def format_time(seconds):
    """Formats seconds into HH:MM:SS."""
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def parse_time_to_seconds(time_str):
    """Converts HH:MM:SS or seconds string to total seconds."""
    if ':' in time_str:
        parts = list(map(int, time_str.split(':')))
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        elif len(parts) == 2:
            return parts[0] * 60 + parts[1]
        else:
            raise ValueError("Invalid time format. Use HH:MM:SS or MM:SS.")
    else:
        return int(time_str)

def video_to_audio_cutter_app():
    st.title("動画から音声を切り出しMP3で保存")
    st.write("動画ファイルをアップロードし、切り出したい開始時間と終了時間を指定してください。複数の区間を切り出すことができます。")

    uploaded_video = st.file_uploader("動画ファイルを選択", type=["mp4", "mov", "avi", "mkv", "webm"])

    if uploaded_video is not None:
        st.video(uploaded_video)

        st.subheader("切り出し区間の設定")
        # Use st.data_editor for multiple time range inputs
        default_data = pd.DataFrame([
            {"開始時間": "00:00:00", "終了時間": "00:00:30"}
        ])
        edited_df = st.data_editor(
            default_data,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "開始時間": st.column_config.TextColumn(
                    "開始時間 (HH:MM:SS or seconds)",
                    help="切り出し開始時間 (例: 00:00:10 または 10)",
                    default="00:00:00"
                ),
                "終了時間": st.column_config.TextColumn(
                    "終了時間 (HH:MM:SS or seconds)",
                    help="切り出し終了時間 (例: 00:00:30 または 30)",
                    default="00:00:30"
                ),
            }
        )

        if st.button("音声を切り出してMP3で保存"):
            if edited_df.empty:
                st.warning("切り出し区間が設定されていません。")
                return

            temp_video_path = ""
            output_audio_paths = [] # List to store paths of all generated MP3s
            zip_buffer = BytesIO()

            try:
                # Save uploaded video to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_video.name.split('.')[-1]}") as temp_video_file:
                    temp_video_file.write(uploaded_video.read())
                    temp_video_path = temp_video_file.name

                with st.spinner("音声の切り出しとMP3への変換中..."):
                    for index, row in edited_df.iterrows():
                        start_time_str = str(row["開始時間"])
                        end_time_str = str(row["終了時間"])

                        try:
                            start_seconds = parse_time_to_seconds(start_time_str)
                            end_seconds = parse_time_to_seconds(end_time_str)

                            if start_seconds >= end_seconds:
                                st.error(f"区間 {index+1}: 開始時間 ({start_time_str}) は終了時間 ({end_time_str}) より前に設定してください。この区間はスキップされます。")
                                continue

                            # Generate a unique output filename for each segment
                            output_audio_filename = f"{os.path.splitext(uploaded_video.name)[0]}_cut_{start_seconds}-{end_seconds}.mp3"
                            output_audio_path = os.path.join(tempfile.gettempdir(), output_audio_filename)

                            command_str = (
                                f"ffmpeg -i \"{temp_video_path}\" "
                                f"-ss {format_time(start_seconds)} -to {format_time(end_seconds)} "
                                f"-vn -ab 192k -map_metadata -1 -y \"{output_audio_path}\""
                            )

                            st.info(f"区間 {index+1} FFmpegコマンドを実行中: {command_str}")
                            
                            process = subprocess.run(command_str, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, shell=True)
                            st.success(f"区間 {index+1} の音声切り出しとMP3への変換が完了しました！")
                            st.info("FFmpegのコンソール出力は、UnicodeDecodeErrorを回避するために非表示にされています。")
                            output_audio_paths.append(output_audio_path)

                        except subprocess.CalledProcessError as e:
                            st.error(f"区間 {index+1} FFmpegの実行中にエラーが発生しました: {e}")
                            st.code(e.stdout)
                            st.code(e.stderr)
                            st.warning("FFmpegがシステムにインストールされ、PATHが通っていることを確認してください。")
                        except ValueError as e:
                            st.error(f"区間 {index+1} 時間形式エラー: {e}")
                        except Exception as e:
                            st.error(f"区間 {index+1} 処理中にエラーが発生しました: {e}")

                if output_audio_paths:
                    st.subheader("生成されたMP3ファイル")
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                        for audio_path in output_audio_paths:
                            if os.path.exists(audio_path):
                                zf.write(audio_path, os.path.basename(audio_path))
                                st.write(f"- {os.path.basename(audio_path)}")
                    zip_buffer.seek(0)

                    st.download_button(
                        label="全てのMP3ファイルをまとめてダウンロード (ZIP)",
                        data=zip_buffer,
                        file_name=f"{os.path.splitext(uploaded_video.name)[0]}_cut_audios.zip",
                        mime="application/zip"
                    )
                else:
                    st.warning("切り出されたMP3ファイルはありませんでした。")

            except Exception as e:
                st.error(f"動画ファイルの処理中にエラーが発生しました: {e}")
            finally:
                # Clean up temporary files
                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)
                for audio_path in output_audio_paths:
                    if os.path.exists(audio_path):
                        os.remove(audio_path)

# This function will be called from mojiokoshi4.py's main
if __name__ == "__main__":
    video_to_audio_cutter_app()
