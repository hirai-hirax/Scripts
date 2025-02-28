import openai
import pandas as pd
import streamlit as st
import io

prompt = """
あなたはプロの校正者です。会議の発言録の草稿を校正するために雇われました。
これから発言が1個ずつ与えられるので、以下のルールに従い校正してください
##ルール##
- 冗長な表現や過剰に硬い表現は避けて、シンプルな日本語の文章にしてください
- 会話の流れを考慮して、適切な敬語を使ってください
- 会話の内容を正確に伝えるようにしてください
- 「です」「ます」調ではなくて「である」調でお願いします
- 「わかりました」「了解」のような、了承を示す言葉は、「承知した」に統一してください
- 句読点は「。」「、」のみを使用してください
- 会話の内容に関係ない部分は削除してください
-「あー」「えー」のようなフィラーは削除してください
"""

def analyze_text(text):
    client = openai.OpenAI()
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", 
        "content": prompt},
        {"role": "user",
        "content": text},])
    return response.choices[0].message.content

def create_excel_download(df: pd.DataFrame) -> bytes:
    """
    渡された DataFrame を Excel ファイルに変換してバイト列を返す
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

def execute_summarize(df):
    df["summerized_text"] = ""
    for index, row in df.iterrows():
        summerized_text = analyze_text(row["content"])
        print(summerized_text)
        df.at[index, "summerized_text"] = summerized_text
    return df

def main():
    st.title("Company Analysis App")
    st.write("議事録のCSVファイルをアップロードしてください。")

    uploaded_file = st.file_uploader("Upload the csv file", type=["csv"], key="file1")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, dtype=str)
        st.write("The uploaded file has the following data:")
        st.write(df)

        if st.button("Start summarizing"):
            df = execute_summarize(df)
            st.write("The analyzed data is as follows:")
            st.write(df)
            excel_data = create_excel_download(df)
            
            st.download_button(
                label="Excel ファイルとしてダウンロード",
                data=excel_data,
                file_name="processed_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

if __name__ == "__main__":
    main()
