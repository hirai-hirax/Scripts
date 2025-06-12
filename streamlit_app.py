import streamlit as st

st.title("ファイルアップローダー")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("📄 **ドキュメント**")
    uploaded_file1 = st.file_uploader("ドキュメントファイルを選択", type=['pdf', 'txt', 'docx'], key="doc")

with col2:
    st.markdown("🖼️ **画像**")
    uploaded_file2 = st.file_uploader("画像ファイルを選択", type=['png', 'jpg', 'jpeg'], key="img")

with col3:
    st.markdown("📊 **データ**")
    uploaded_file3 = st.file_uploader("データファイルを選択", type=['csv', 'xlsx', 'json'], key="data")

if uploaded_file1:
    st.success(f"ドキュメント: {uploaded_file1.name}")

if uploaded_file2:
    st.success(f"画像: {uploaded_file2.name}")

if uploaded_file3:
    st.success(f"データ: {uploaded_file3.name}")