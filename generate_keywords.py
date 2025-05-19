import os
import base64
from openai import AzureOpenAI
import fitz  # PyMuPDF

AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
API_VERSION = "2025-01-01-preview"  # ご利用の API バージョンに合わせてください
DEPLOYMENT = "gpt-4o-mini"
model = "gpt-4o-mini"

prompt = """
# 目的
ユーザーから、PDFから抽出されたテキストを渡されます。当該のテキストの内容を読んだ上で、100文字程度の要約を生成してください。
"""

text = """

"""

def extract_text_from_pdf(file) -> str:
    """
    fitz (PyMuPDF) を使って PDF からテキストを抽出
    """
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = []
    for page in doc:
        txt = page.get_text()
        if txt:
            text.append(txt)
    return "\n".join(text)

def generate_keywords(model,FilePathOfPdf):
    # クライアント初期化
    text = extract_text_from_pdf(FilePathOfPdf)
    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=API_VERSION,
    )

    response = client.chat.completions.create(
        model=model,
         messages=[
        {"role": "system", 
        "content": prompt},
        {"role": "user",
        "content": text},
        ],
    )

    # --- テキスト部分を取り出してファイルに書き出し ---
    print(response)

if __name__ == "__main__":
    generate_keywords(model,"sample.pdf")