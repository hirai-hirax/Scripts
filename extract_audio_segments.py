import pandas as pd
from pydub import AudioSegment
import os

def extract_segments(audio_path, data_input):
    """
    Extracts audio segments from a larger audio file based on timestamps.

    Args:
        audio_path (str): Path to the original audio file.
        data_input (str or pd.DataFrame): Path to a CSV file or a pandas DataFrame
                                          with 'start', 'end', and 'text' columns.
    """
    try:
        audio = AudioSegment.from_file(audio_path)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return

    if isinstance(data_input, str):
        try:
            dataframe = pd.read_csv(data_input)
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return
    elif isinstance(data_input, pd.DataFrame):
        dataframe = data_input
    else:
        print("Invalid data_input type. Must be a CSV file path or a pandas DataFrame.")
        return

    if not all(col in dataframe.columns for col in ['start', 'end', 'text']):
        print("DataFrame must contain 'start', 'end', and 'text' columns.")
        return

    output_dir = "extracted_segments"
    os.makedirs(output_dir, exist_ok=True)

    for index, row in dataframe.iterrows():
        start_time = row['start'] * 1000  # pydub uses milliseconds
        end_time = row['end'] * 1000
        text = row['text']

        try:
            segment = audio[start_time:end_time]
            # Sanitize text for filename
            safe_text = "".join([c for c in text[:20] if c.isalnum() or c in (' ', '_')]).rstrip()
            output_filename = os.path.join(output_dir, f"segment_{index}_{safe_text.replace(' ', '_')}.wav")
            segment.export(output_filename, format="wav")
            print(f"Exported segment {index} to {output_filename}")
        except Exception as e:
            print(f"Error processing segment {index}: {e}")

if __name__ == "__main__":
    # Create a dummy CSV file for demonstration
    csv_file=r"C:\Users\hir31\Downloads\merged_segments (8).csv"
    df = pd.read_csv(csv_file, encoding="shift-jis", on_bad_lines='skip')

    # Replace with the actual path to your audio file
    input_audio_file = r"C:\Users\hir31\Downloads\interview_aps-smp.mp3"

    # Create a dummy audio file for demonstration if it doesn't exist
    if not os.path.exists(input_audio_file):
        print(f"Creating a dummy audio file: {input_audio_file}")
        # Requires ffmpeg or librosa to be installed
        try:
            dummy_audio = AudioSegment.silent(duration=20000) # 20 seconds of silence
            dummy_audio.export(input_audio_file, format="wav")
        except Exception as e:
            print(f"Could not create dummy audio file. Please ensure ffmpeg is installed and in your PATH, or create '{input_audio_file}' manually. Error: {e}")
            exit()

    # Example usage with a CSV file path
    extract_segments(input_audio_file, df)

    # Example usage with a DataFrame (optional, uncomment to test)
    # extract_segments(input_audio_file, dummy_df)
