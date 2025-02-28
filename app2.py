import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import tempfile

# ページタイトル
st.title("バリューチェーンのCO2排出量分析アプリ")

# セッションステートにノード・エッジ、CO2情報のテーブルを用意
if 'graph_df' not in st.session_state:
    st.session_state.graph_df = pd.DataFrame(columns=['ID', '名称', 'タイプ'])
if 'co2_df' not in st.session_state:
    st.session_state.co2_df = pd.DataFrame(columns=['名称', '発生した工程ID', '燃料消費量', '電力消費量'])

st.header("1. 工程（ノード/エッジ）の定義")

# ノードまたはエッジの追加選択
option = st.selectbox("追加する工程を選択", ["ノード追加", "エッジ追加"])

if option == "ノード追加":
    node_id = st.text_input("ノードID", key="node_id")
    node_name = st.text_input("ノード名称", key="node_name")
    if st.button("ノード追加"):
        if node_id and node_name:
            new_row = pd.DataFrame([{'ID': node_id, '名称': node_name, 'タイプ': 'ノード'}])
            st.session_state.graph_df = pd.concat([st.session_state.graph_df, new_row], ignore_index=True)
            st.success(f"ノード【{node_name}】を追加しました。")
        else:
            st.error("ID と名称の両方を入力してください。")

elif option == "エッジ追加":
    edge_id = st.text_input("エッジID", key="edge_id")
    edge_name = st.text_input("エッジ名称", key="edge_name")
    from_node = st.text_input("開始ノードID", key="from_node")
    to_node = st.text_input("終了ノードID", key="to_node")
    if st.button("エッジ追加"):
        if edge_id and edge_name and from_node and to_node:
            new_row = pd.DataFrame([{
                'ID': edge_id,
                '名称': edge_name,
                'タイプ': f'エッジ: {from_node}->{to_node}'
            }])
            st.session_state.graph_df = pd.concat([st.session_state.graph_df, new_row], ignore_index=True)
            st.success(f"エッジ【{edge_name}】（{from_node}→{to_node}）を追加しました。")
        else:
            st.error("全項目を入力してください。")

st.subheader("工程一覧（ノード・エッジ）")
st.dataframe(st.session_state.graph_df)

st.header("2. CSVファイルアップロード")
if not st.session_state.graph_df.empty:
    selected_process = st.selectbox("対象の工程IDを選択", st.session_state.graph_df['ID'])
    uploaded_file = st.file_uploader("対象工程のインプット情報（CSV）をアップロード", type="csv")
    if uploaded_file is not None:
        try:
            csv_data = pd.read_csv(uploaded_file)
            st.write("アップロードされたCSVの内容")
            st.dataframe(csv_data)
        except Exception as e:
            st.error("CSVの読み込みエラー: " + str(e))
else:
    st.info("まずは工程（ノード/エッジ）を追加してください。")

st.header("3. CO2排出量情報の入力")
co2_name = st.text_input("CO2情報名称", key="co2_name")
process_id = st.text_input("発生した工程ID", key="process_co2")
fuel = st.number_input("燃料消費量", value=0.0, key="fuel")
electricity = st.number_input("電力消費量", value=0.0, key="electricity")
if st.button("CO2情報追加"):
    if co2_name and process_id:
        new_row = pd.DataFrame([{
            '名称': co2_name,
            '発生した工程ID': process_id,
            '燃料消費量': fuel,
            '電力消費量': electricity
        }])
        st.session_state.co2_df = pd.concat([st.session_state.co2_df, new_row], ignore_index=True)
        st.success(f"CO2情報【{co2_name}】を追加しました。")
    else:
        st.error("CO2情報名称と発生工程IDを入力してください。")

st.subheader("CO2排出量情報一覧")
st.dataframe(st.session_state.co2_df)

st.header("4. バリューチェーングラフの可視化")
if st.button("グラフ表示"):
    # NetworkXで有向グラフを構築
    G = nx.DiGraph()

    # ノードの追加
    nodes_df = st.session_state.graph_df[st.session_state.graph_df['タイプ'] == 'ノード']
    for idx, row in nodes_df.iterrows():
        G.add_node(row['ID'], label=row['名称'])

    # エッジの追加（タイプに"エッジ:"が含まれる行を対象）
    edges_df = st.session_state.graph_df[st.session_state.graph_df['タイプ'].str.startswith('エッジ')]
    for idx, row in edges_df.iterrows():
        try:
            _, nodes = row['タイプ'].split(':')
            from_node, to_node = nodes.strip().split('->')
            G.add_edge(from_node, to_node, label=row['名称'])
        except Exception as e:
            st.error("エッジ情報の解析に失敗しました: " + str(e))

    # pyvisでグラフを可視化
    net = Network(height="600px", width="100%", directed=True)
    net.from_nx(G)
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
    net.show(tmp_file.name)
    st.components.v1.html(open(tmp_file.name, 'r', encoding='utf-8').read(), height=600)
