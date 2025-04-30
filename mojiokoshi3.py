# transcribe_mfcc_diarization.py
# 25MB制限を回避しつつResemblyzer不要のMFCCベース話者分離付き文字起こしスクリプト

import os
import sys
import subprocess
import tempfile
from datetime import timedelta
import numpy as np
import librosa
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm
from openai import AzureOpenAI

AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
API_VERSION = "2023-09-01-preview"  # ご利用の API バージョンに合わせてください
model = "whisper"

def extract_mfcc_embeddings(audio_path, window_size=2.0, hop_length=512, n_mfcc=13, sr=16000):
    """
    音声全体を短時間フレームに分割し、各フレームのMFCCを計算して埋め込み行列を返す
    RETURN: mfcc_features (nb_frames, n_mfcc), times (list of frame start seconds)
    """
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    frame_length = int(window_size * sr)
    hop = int(hop_length)

    mfccs = []
    times = []
    for start_sample in range(0, len(y) - frame_length + 1, hop):
        frame = y[start_sample:start_sample + frame_length]
        mfcc = librosa.feature.mfcc(y=frame, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfccs.append(mfcc_mean)
        times.append(start_sample / sr)
    return np.vstack(mfccs), times


def cluster_speakers(embeddings, n_speakers=6):
    """MFCC埋め込みを階層クラスタリングしてラベル返却"""
    clustering = AgglomerativeClustering(
        n_clusters=n_speakers,
        metric="euclidean",
        linkage="ward"
    )
    labels = clustering.fit_predict(embeddings)
    return labels


def build_speaker_segments(times, labels, window_size=2.0):
    """連続した同一ラベルをまとめて話者セグメントリストを構築"""
    segments = []
    cur_label = labels[0]
    seg_start = times[0]
    for t, lab in zip(times[1:], labels[1:]):
        if lab != cur_label:
            segments.append({"start": seg_start, "end": t, "speaker": int(cur_label)})
            seg_start = t
            cur_label = lab
    segments.append({"start": seg_start, "end": times[-1] + window_size, "speaker": int(cur_label)})
    return segments


def split_audio_chunks(audio_path, chunk_sec=600.0, overlap_sec=30.0, sr=16000):
    """10分チャンク＋前後30秒オーバーラップで分割"""
    duration = librosa.get_duration(path=audio_path)
    chunks = []
    start = 0.0
    while start < duration:
        end = min(start + chunk_sec, duration)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            chunk_path = tmp.name
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                str(max(0, start - overlap_sec)),
                "-to",
                str(min(duration, end + overlap_sec)),
                "-i",
                audio_path,
                "-ac",
                "1",
                "-ar",
                str(sr),
                "-hide_banner",
                "-loglevel",
                "error",
                chunk_path,
            ],
            check=True,
        )
        chunks.append((chunk_path, start))
        start += chunk_sec
    return chunks


def transcribe_chunk(chunk_path):
    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT, api_key=AZURE_OPENAI_API_KEY, api_version=API_VERSION
    )
    with open(chunk_path, "rb") as f:
        return client.audio.transcriptions.create(
            model=model,
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["segment", "word"],
            language="ja",
        )


def assign_speakers(transcript_result, speaker_segments, offset):
    """文字起こし結果のセグメント開始時間に基づき、グローバルセグメントから話者IDをマッピング"""
    assigned = []
    for seg in transcript_result.segments:
        # 属性アクセスで start,text を取得
        t0 = offset + seg.start
        spk = next((s["speaker"] for s in speaker_segments if s["start"] <= t0 < s["end"]), None)
        label = f"SPEAKER_{spk:02d}" if spk is not None else "UNKNOWN"
        assigned.append({"start": t0, "speaker": label, "text": seg.text.strip()})
    return assigned


def merge_transcripts(assigned):
    """連続同一話者をマージしテキスト化"""
    merged = []
    for seg in assigned:
        if merged and merged[-1]["speaker"] == seg["speaker"]:
            merged[-1]["text"] += " " + seg["text"]
        else:
            merged.append(seg)
    lines = []
    for e in merged:
        ts = str(timedelta(seconds=int(e["start"]))).split(".")[0]
        txt = e["text"]
        if txt and not txt.endswith(("。", "？", "！")):
            txt += "。"
        lines.append(f"[{ts}] {e['speaker']}: {txt}")
    return "\n".join(lines)


def main(audio_path, n_speakers=6):
    print("グローバル話者モデル構築中...")
    embeddings, times = extract_mfcc_embeddings(audio_path)
    labels = cluster_speakers(embeddings, n_speakers)
    speaker_segments = build_speaker_segments(times, labels)

    print("チャンク分割中...")
    chunks = split_audio_chunks(audio_path)
    assigned_all = []
    for chunk_path, offset in tqdm(chunks, desc="Transcribing"):
        res = transcribe_chunk(chunk_path)
        assigned = assign_speakers(res, speaker_segments, offset)
        assigned_all.extend(assigned)
        os.remove(chunk_path)

    print("トランスクリプトマージ中...")
    transcript = merge_transcripts(assigned_all)
    out_file = os.path.splitext(audio_path)[0] + "_mfcc_diarized.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(transcript)
    print(f"保存完了: {out_file}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transcribe_mfcc_diarization.py <audio_file> [num_speakers]")
        sys.exit(1)
    audio_file = sys.argv[1]
    speakers = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    main(audio_file, speakers)
