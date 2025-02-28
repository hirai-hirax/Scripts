from flask import Flask, render_template_string

app = Flask(__name__)

HTML_PAGE = '''
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>バリューチェーン分析 - データテーブル紐づけ付き Webアプリ</title>
  <style>
    body { font-family: sans-serif; }
    #controls {
      text-align: center;
      margin: 10px;
    }
    #container {
      width: 800px;
      height: 600px;
      position: relative;
      border: 1px solid #ccc;
      margin: 20px auto;
      background: #fff;
    }
    .chain-element {
      width: 100px;
      height: 50px;
      background-color: lightblue;
      border: 1px solid black;
      text-align: center;
      line-height: 50px;
      position: absolute;
      cursor: move;
      user-select: none;
    }
    .selected {
      border: 2px solid red;
    }
    /* SVGはcontainer内に絶対配置 */
    #svgOverlay {
      position: absolute;
      top: 0;
      left: 0;
      pointer-events: none;
      width: 100%;
      height: 100%;
      overflow: visible;
    }
    /* 矢印追加モード時のハイライト */
    .pending {
      border: 2px dashed green;
    }
    /* モーダルウィンドウ */
    #modalOverlay {
      position: fixed;
      top: 0; left: 0;
      width: 100%; height: 100%;
      background: rgba(0,0,0,0.5);
      display: none;
      justify-content: center;
      align-items: center;
    }
    #modal {
      background: #fff;
      padding: 20px;
      border-radius: 5px;
      width: 400px;
      max-height: 80%;
      overflow-y: auto;
    }
    #modal table {
      width: 100%;
      border-collapse: collapse;
    }
    #modal table, #modal th, #modal td {
      border: 1px solid #ccc;
    }
    #modal th, #modal td {
      padding: 5px;
      text-align: center;
    }
    #modal button {
      margin: 5px;
    }
  </style>
</head>
<body>
  <h1 style="text-align:center;">バリューチェーン分析 - データテーブル紐づけ付き Webアプリ</h1>
  <div id="controls">
    <button id="addButton">新しいボタンを作る</button>
    <button id="deleteButton">選択したボタンを消す</button>
    <button id="connectModeButton">矢印追加モード OFF</button>
  </div>
  <div id="container">
    <!-- SVGレイヤー -->
    <svg id="svgOverlay">
      <defs>
        <marker id="arrow" markerWidth="10" markerHeight="10" refX="10" refY="3" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L0,6 L9,3 z" fill="#f00" />
        </marker>
      </defs>
    </svg>
    <!-- 初期のチェーン要素 -->
    <div class="chain-element" draggable="true" id="procurement" style="left:50px; top:50px;">調達</div>
    <div class="chain-element" draggable="true" id="production" style="left:200px; top:50px;">生産</div>
    <div class="chain-element" draggable="true" id="marketing" style="left:350px; top:50px;">マーケティング</div>
    <div class="chain-element" draggable="true" id="logistics" style="left:500px; top:50px;">物流</div>
    <div class="chain-element" draggable="true" id="service" style="left:650px; top:50px;">サービス</div>
  </div>

  <!-- モーダルウィンドウ（データテーブル編集用） -->
  <div id="modalOverlay">
    <div id="modal">
      <h2>データテーブル編集</h2>
      <table id="dataTable">
        <thead>
          <tr>
            <th>キー</th>
            <th>値</th>
            <th>操作</th>
          </tr>
        </thead>
        <tbody>
          <!-- 行はJavaScriptで追加 -->
        </tbody>
      </table>
      <button id="addRowButton">行を追加</button>
      <div style="text-align:right; margin-top:10px;">
        <button id="saveTableButton">保存</button>
        <button id="cancelTableButton">キャンセル</button>
      </div>
    </div>
  </div>

  <script>
    const container = document.getElementById('container');
    const svgOverlay = document.getElementById('svgOverlay');
    let dragItem = null;
    let offsetX = 0, offsetY = 0;
    let selectedElement = null;
    let connectMode = false;
    let pendingSource = null;
    const connections = []; // { source: element, target: element, line: SVGLineElement }

    // データテーブルを各要素に紐づけるためのオブジェクト
    const elementData = {}; // キー: element.id, 値: 配列 [{ key: ..., value: ... }, ...]

    // モーダル関連変数
    const modalOverlay = document.getElementById('modalOverlay');
    const dataTable = document.getElementById('dataTable').getElementsByTagName('tbody')[0];
    let currentEditingElement = null;

    // 選択状態を解除する
    function clearSelection() {
      if (selectedElement) {
        selectedElement.classList.remove('selected');
        selectedElement = null;
      }
    }

    // ドラッグ＆選択のイベントリスナーを追加
    function addDragAndSelectListeners(item) {
      item.addEventListener('dragstart', function(e) {
        dragItem = this;
        offsetX = e.offsetX;
        offsetY = e.offsetY;
      });

      // ドラッグ終了時に矢印の再描画
      item.addEventListener('dragend', function(e) {
        updateAllConnections();
      });

      // クリックで通常選択または接続用の処理
      item.addEventListener('click', function(e) {
        e.stopPropagation();
        if (connectMode) {
          // 接続モードの場合
          if (!pendingSource) {
            pendingSource = this;
            this.classList.add('pending');
          } else if (pendingSource !== this) {
            addConnection(pendingSource, this);
            pendingSource.classList.remove('pending');
            pendingSource = null;
          }
        } else {
          clearSelection();
          selectedElement = this;
          selectedElement.classList.add('selected');
        }
      });

      // ダブルクリックでデータテーブル編集モーダルを表示
      item.addEventListener('dblclick', function(e) {
        e.stopPropagation();
        openModalForElement(this);
      });
    }

    // 初期要素にリスナーを追加
    document.querySelectorAll('.chain-element').forEach(item => {
      addDragAndSelectListeners(item);
    });

    // ドラッグオーバーとドロップの処理（container側）
    container.addEventListener('dragover', function(e) {
      e.preventDefault();
    });
    container.addEventListener('drop', function(e) {
      e.preventDefault();
      if (dragItem) {
        const rect = container.getBoundingClientRect();
        const x = e.clientX - rect.left - offsetX;
        const y = e.clientY - rect.top - offsetY;
        dragItem.style.left = x + 'px';
        dragItem.style.top = y + 'px';
        updateAllConnections();
        dragItem = null;
      }
    });

    // コンテナクリックで選択解除（通常モード）
    container.addEventListener('click', function(e) {
      if (e.target === container) {
        clearSelection();
      }
    });

    // 新しいボタン追加機能
    let newIdCounter = 1;
    document.getElementById('addButton').addEventListener('click', function() {
      const newDiv = document.createElement('div');
      newDiv.className = 'chain-element';
      newDiv.draggable = true;
      newDiv.style.left = '100px';
      newDiv.style.top = '100px';
      newDiv.textContent = '新規ボタン ' + newIdCounter;
      newDiv.id = 'newButton' + newIdCounter;
      newIdCounter++;
      container.appendChild(newDiv);
      addDragAndSelectListeners(newDiv);
    });

    // 選択したボタン削除機能（接続も削除）
    document.getElementById('deleteButton').addEventListener('click', function() {
      if (selectedElement) {
        // 接続されている矢印も削除
        for (let i = connections.length - 1; i >= 0; i--) {
          if (connections[i].source === selectedElement || connections[i].target === selectedElement) {
            svgOverlay.removeChild(connections[i].line);
            connections.splice(i, 1);
          }
        }
        // 紐づけたデータも削除
        delete elementData[selectedElement.id];
        container.removeChild(selectedElement);
        selectedElement = null;
      } else {
        alert("削除するボタンを選択してください。");
      }
    });

    // 矢印接続モードのON/OFF切替
    const connectModeButton = document.getElementById('connectModeButton');
    connectModeButton.addEventListener('click', function() {
      connectMode = !connectMode;
      connectModeButton.textContent = connectMode ? "矢印追加モード ON" : "矢印追加モード OFF";
      if (pendingSource) {
        pendingSource.classList.remove('pending');
        pendingSource = null;
      }
    });

    // 指定した2要素間に矢印を追加
    function addConnection(source, target) {
      const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      line.setAttribute("stroke", "#f00");
      line.setAttribute("stroke-width", "2");
      line.setAttribute("marker-end", "url(#arrow)");
      svgOverlay.appendChild(line);
      connections.push({ source, target, line });
      updateConnection(line, source, target);
    }

    // 1本の矢印の位置を更新
    function updateConnection(line, source, target) {
      const sourceRect = source.getBoundingClientRect();
      const targetRect = target.getBoundingClientRect();
      const containerRect = container.getBoundingClientRect();
      const x1 = sourceRect.left - containerRect.left + sourceRect.width / 2;
      const y1 = sourceRect.top - containerRect.top + sourceRect.height / 2;
      const x2 = targetRect.left - containerRect.left + targetRect.width / 2;
      const y2 = targetRect.top - containerRect.top + targetRect.height / 2;
      line.setAttribute("x1", x1);
      line.setAttribute("y1", y1);
      line.setAttribute("x2", x2);
      line.setAttribute("y2", y2);
    }

    // 全ての接続の矢印位置を更新
    function updateAllConnections() {
      connections.forEach(conn => {
        updateConnection(conn.line, conn.source, conn.target);
      });
    }

    // -----------------------------
    // モーダルウィンドウ（データテーブル編集）関連
    // -----------------------------

    // モーダルを開く
    function openModalForElement(element) {
      currentEditingElement = element;
      // 既存のデータがあればロード、なければ空の行を1行追加
      dataTable.innerHTML = "";
      const data = elementData[element.id] || [];
      if (data.length === 0) {
        addRowToTable("", "");
      } else {
        data.forEach(row => addRowToTable(row.key, row.value));
      }
      modalOverlay.style.display = "flex";
    }

    // 行を追加する
    function addRowToTable(key, value) {
      const tr = document.createElement('tr');
      const tdKey = document.createElement('td');
      const inputKey = document.createElement('input');
      inputKey.type = "text";
      inputKey.value = key;
      tdKey.appendChild(inputKey);

      const tdValue = document.createElement('td');
      const inputValue = document.createElement('input');
      inputValue.type = "text";
      inputValue.value = value;
      tdValue.appendChild(inputValue);

      const tdAction = document.createElement('td');
      const deleteBtn = document.createElement('button');
      deleteBtn.textContent = "削除";
      deleteBtn.addEventListener('click', function() {
        tr.parentNode.removeChild(tr);
      });
      tdAction.appendChild(deleteBtn);

      tr.appendChild(tdKey);
      tr.appendChild(tdValue);
      tr.appendChild(tdAction);
      dataTable.appendChild(tr);
    }

    // 「行を追加」ボタン
    document.getElementById('addRowButton').addEventListener('click', function() {
      addRowToTable("", "");
    });

    // 「保存」ボタン：テーブルの内容を保存
    document.getElementById('saveTableButton').addEventListener('click', function() {
      const rows = dataTable.getElementsByTagName('tr');
      const newData = [];
      for (let row of rows) {
        const inputs = row.getElementsByTagName('input');
        if (inputs.length >= 2) {
          newData.push({ key: inputs[0].value, value: inputs[1].value });
        }
      }
      // 現在の要素IDにデータを紐づけ
      elementData[currentEditingElement.id] = newData;
      closeModal();
    });

    // 「キャンセル」ボタン
    document.getElementById('cancelTableButton').addEventListener('click', function() {
      closeModal();
    });

    function closeModal() {
      modalOverlay.style.display = "none";
      currentEditingElement = null;
    }
  </script>
</body>
</html>

'''

@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

if __name__ == '__main__':
    app.run(debug=True)
