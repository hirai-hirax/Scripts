from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np
import sys
from pathlib import Path
import os
import pandas as pd
from pydub import AudioSegment
import io
import tempfile
import os

def extract_embedding(audio_path, output_path=None):
    wav = preprocess_wav(audio_path)
    encoder = VoiceEncoder()
    embedding = encoder.embed_utterance(wav)
    if output_path is None:
        output_path = Path(audio_path).with_suffix('.npy')
    np.save(output_path, embedding)
    print(f"Embedding saved to {output_path}")
    return embedding

def load_speaker_embeddings(embeddings_dir="embed"):
    speaker_embeddings = {}
    if not os.path.exists(embeddings_dir):
        print(f"Embeddings directory '{embeddings_dir}' not found.")
        return speaker_embeddings

    for filename in os.listdir(embeddings_dir):
        if filename.endswith(".npy"):
            speaker_name = os.path.splitext(filename)[0]
            embedding_path = os.path.join(embeddings_dir, filename)
            try:
                embedding = np.load(embedding_path)
                speaker_embeddings[speaker_name] = embedding
            except Exception as e:
                print(f"Error loading embedding from {embedding_path}: {e}")
    return speaker_embeddings

def identify_speakers_in_dataframe(audio_path: str, df: pd.DataFrame, embeddings_dir="embed") -> pd.DataFrame:
    """
    Identifies speakers for audio segments defined in a DataFrame.

    Args:
        audio_path: Path to the main audio file.
        df: DataFrame with 'start' and 'end' columns (in seconds).
        embeddings_dir: Directory containing known speaker embeddings.

    Returns:
        DataFrame with an added 'speaker' column.
    """
    known_embeddings = load_speaker_embeddings(embeddings_dir)
    if not known_embeddings:
        print("No known speaker embeddings found. Cannot perform identification.")
        df['speaker'] = None
        return df

    print(f"Loaded embeddings for speakers: {list(known_embeddings.keys())}")

    audio = AudioSegment.from_file(audio_path)
    df['speaker'] = None # Initialize speaker column

    for index, row in df.iterrows():
        start_time_ms = row['start'] * 1000
        end_time_ms = row['end'] * 1000

        # Extract audio segment
        segment = audio[start_time_ms:end_time_ms]

        # Convert segment to numpy array for preprocess_wav
        # preprocess_wav expects a numpy array or a file path
        # Save the segment to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav_file:
            segment.export(temp_wav_file.name, format="wav")
            temp_wav_file_path = temp_wav_file.name

        try:
            # preprocess_wav can take a file path
            wav = preprocess_wav(temp_wav_file_path)
            encoder = VoiceEncoder()
            segment_embedding = encoder.embed_utterance(wav)
        finally:
            # Clean up the temporary file
            os.remove(temp_wav_file_path)

        try:
            highest_similarity = -1
            identified_speaker = None

            for speaker_name, known_embedding in known_embeddings.items():
                similarity = np.dot(segment_embedding, known_embedding) / (np.linalg.norm(segment_embedding) * np.linalg.norm(known_embedding))
                # print(f"Segment {row['start']}-{row['end']}s, Similarity with {speaker_name}: {similarity:.4f}") # Optional: for debugging
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    identified_speaker = speaker_name

            df.at[index, 'speaker'] = identified_speaker
            print(f"Segment {row['start']}-{row['end']}s identified as: {identified_speaker}")

        except Exception as e:
            print(f"Error processing segment {row['start']}-{row['end']}s: {e}")
            df.at[index, 'speaker'] = "Error" # Mark segments that failed

    return df

#df = pd.read_csv(r"C:\Users\hir31\Downloads\merged_segments (9).csv", encoding="shift-jis", on_bad_lines='skip')
#df2 = identify_speakers_in_dataframe(audio_path=r"C:\Users\hir31\Downloads\interview_aps-smp.mp3",
#                                   df=df,
#                                   embeddings_dir=r"C:\Users\hir31\dataanalysis\Scripts\embed")
#print(df2)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python speaker_identification.py <mode> <args>")
        print("Modes:")
        print("  extract <audio_file> [output_file]")
        print("  identify <unknown_audio_file> [embeddings_dir]")
        print("  dataframe_identify <audio_file> <csv_file> [embeddings_dir]")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "extract":
        if len(sys.argv) < 3:
            print("Usage: python speaker_identification.py extract <audio_file> [output_file]")
            sys.exit(1)
        audio_file = sys.argv[2]
        output_file = sys.argv[3] if len(sys.argv) > 3 else None
        extract_embedding(audio_file, output_file)

    elif mode == "identify":
        if len(sys.argv) < 3:
            print("Usage: python speaker_identification.py identify <unknown_audio_file> [embeddings_dir]")
            sys.exit(1)
        unknown_audio_file = sys.argv[2]
        embeddings_dir = sys.argv[3] if len(sys.argv) > 3 else "embed"

        known_embeddings = load_speaker_embeddings(embeddings_dir)
        if not known_embeddings:
            print("No known speaker embeddings found. Cannot perform identification.")
            sys.exit(1)

        print(f"Loaded embeddings for speakers: {list(known_embeddings.keys())}")

        try:
            unknown_embedding = extract_embedding(unknown_audio_file)
        except Exception as e:
            print(f"Error extracting embedding from {unknown_audio_file}: {e}")
            sys.exit(1)

        highest_similarity = -1
        identified_speaker = None

        for speaker_name, known_embedding in known_embeddings.items():
            # Calculate cosine similarity using numpy
            similarity = np.dot(unknown_embedding, known_embedding) / (np.linalg.norm(unknown_embedding) * np.linalg.norm(known_embedding))
            print(f"Similarity with {speaker_name}: {similarity:.4f}")
            if similarity > highest_similarity:
                highest_similarity = similarity
                identified_speaker = speaker_name

        if identified_speaker:
            print(f"\nIdentified speaker: {identified_speaker} (Similarity: {highest_similarity:.4f})")
        else:
            print("\nCould not identify speaker.")

    elif mode == "dataframe_identify":
        if len(sys.argv) < 4:
            print("Usage: python speaker_identification.py dataframe_identify <audio_file> <csv_file> [embeddings_dir]")
            sys.exit(1)
        audio_file = sys.argv[2]
        csv_file = sys.argv[3]
        embeddings_dir = sys.argv[4] if len(sys.argv) > 4 else "embed"

        try:
            df = pd.read_csv(csv_file, encoding="shift-jis", on_bad_lines='skip')
        except FileNotFoundError:
            print(f"Error: CSV file not found at {csv_file}")
            sys.exit(1)
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            sys.exit(1)

        df_with_speakers = identify_speakers_in_dataframe(audio_file, df, embeddings_dir)
        print(df_with_speakers.to_csv(index=False)) # Print the updated DataFrame as CSV to standard output

    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python speaker_identification.py <mode> <args>")
        print("Modes:")
        print("  extract <audio_file> [output_file]")
        print("  identify <unknown_audio_file> [embeddings_dir]")
        print("  dataframe_identify <audio_file> <csv_file> [embeddings_dir]")
        sys.exit(1)
