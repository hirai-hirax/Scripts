import os
import json
from flask import Flask, request, Response, render_template_string
from openai import AzureOpenAI

app = Flask(__name__)

# 環境変数から設定を取得（Azure OpenAI のエンドポイント・API キーを設定してください）
#AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_ENDPOINT = "https://ai-hir3147197ai120585502465.openai.azure.com/"


AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
API_VERSION = "2023-09-01-preview"  # 適切な API バージョンに合わせてください

print(AZURE_OPENAI_ENDPOINT)
# AzureOpenAI クライアントのインスタンスを生成
client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=API_VERSION
)
# シンプルなHTMLアップロードフォーム
upload_form = """
<!doctype html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <title>オーディオファイルアップロード</title>
</head>
<body>
    <h1>オーディオファイルをアップロードして文字起こし（テキストファイルダウンロード）</h1>
    <form method="post" action="/transcribe" enctype="multipart/form-data">
        <input type="file" name="audio_file" accept="audio/*">
        <br><br>
        <input type="submit" value="文字起こし開始">
    </form>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(upload_form)

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if 'audio_file' not in request.files:
        error_text = "オーディオファイルがアップロードされていません"
        return Response(error_text.encode('shift_jis'),
                        content_type="text/plain; charset=shift_jis"), 400
    audio_file = request.files['audio_file']
    if audio_file.filename == "":
        error_text = "ファイルが選択されていません"
        return Response(error_text.encode('shift_jis'),
                        content_type="text/plain; charset=shift_jis"), 400

    try:
        # FileStorage の情報をタプル形式に変換（filename, stream, content_type）
        file_tuple = (audio_file.filename, audio_file.stream, audio_file.content_type)
        transcript = client.audio.transcriptions.create(
            model="whisper",  # Azure ポータル上のデプロイ名に合わせてください
            file=file_tuple
        )
        # transcript は pydantic モデルとして返されるので dict に変換し、"text"フィールドを取得
        transcript_text = transcript.model_dump().get("text", "")
        # ダウンロード用レスポンス作成（Content-Disposition ヘッダーでファイルとして扱われる）
        headers = {
            "Content-Disposition": "attachment; filename=transcript.txt"
        }
        return Response(transcript_text.encode('shift_jis'),
                        headers=headers,
                        content_type="text/plain; charset=shift_jis")
    except Exception as e:
        error_text = "Error: " + str(e)
        return Response(error_text.encode('shift_jis'),
                        content_type="text/plain; charset=shift_jis"), 500

if __name__ == '__main__':
    app.run(debug=True)