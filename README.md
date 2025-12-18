# 手勢音樂介面使用說明

利用攝影機與 MediaPipe 手部偵測，透過揮動手掌觸發「音階按鈕」並播放鋼琴或小提琴音色。系統支援 6 組預設音階、動態音量以及「放鬆模式」環境音淡入淡出。

## 環境需求
- Python 3.9 以上（建議 3.10）。
- 可運作的攝影機與喇叭／耳機。
- 系統可執行 FFmpeg（自動將音檔轉為 44.1kHz、16-bit WAV，缺少時會提示）。
- 字體：程式會依序嘗試 `PingFang.ttc`、`wqy-zenhei.ttc`、`msjh.ttf` 等中文字體；找不到時改用系統預設字型。

### 套件安裝
在專案根目錄執行：
```bash
pip install -r requirements.txt
```

### Windows
- 建議安裝含「Desktop development with C++」的 Visual Studio Build Tools 以協助編譯依賴。
- 安裝 FFmpeg 與 OpenCV 執行檔（擇一）：
  - 使用 [Chocolatey](https://chocolatey.org/)：`choco install ffmpeg opencv`。
  - 下載官方安裝包並將路徑加入 `PATH`。
- 首次啟動時允許終端或 `python.exe` 使用相機／麥克風（設定 → 隱私權與安全性）。

### macOS
- 先安裝 [Homebrew](https://brew.sh/) 後執行：`brew install ffmpeg opencv`。
- Apple Silicon 建議安裝 arm64 版 mediapipe，例如 `pip install "mediapipe==0.10.11"` 或 `pip install mediapipe-silicon`；如需 x86_64 相容，可用 Rosetta 虛擬環境。
- 若要自行編譯 mediapipe，可搭配 `bazelisk`、`llvm` 並設定 `CC/CXX` 以提升成功率。
- 若被系統阻擋，請在 **系統設定 → 隱私權與安全性 → 相機／麥克風** 允許 Terminal/Python。

## 音訊素材與檔案結構
- `piano_sound/`：鋼琴音色 (`c1.wav`~`b1.wav`)。
- `violin_sound/`：小提琴音色 (`c3.wav`~`b3.wav`)。
- `ambient_sound/`：環境音（預設尋找 `ocean.wav`、`wind.wav`，缺檔會自動合成內建環境聲）。
- `assets/beep.wav`：提示音，缺檔時會改用程式內建嗶聲。

非標準取樣率／位深的檔案會在播放前自動透過 FFmpeg 轉檔至臨時 WAV，無 FFmpeg 時會在終端提示。

## 執行方式
```bash
python main.py
```

啟動後流程：
1. 程式自動掃描攝影機（優先使用編號最大的外接鏡頭），失敗時會詢問你輸入索引。
2. 進入等待畫面：`s` 開始校準、`q` 離開。
3. 校準完成即進入演奏模式，畫面顯示可觸發的區塊。

## 操作方式
- **觸發區塊**：手掌停留在區塊內，讀條滿格後觸發；移出會逐步衰減，避免誤觸。
- **音階列（上方五鍵）**：顯示當前樂器預設的五個音階，手速越快音量越大。若音檔缺失會改用嗶聲。
- **樂器切換**：左右下角為上一個／下一個樂器。內建 6 組預設：
  - 鋼琴：中國五聲、律音階、呂音階
  - 小提琴：中國五聲、律音階、呂音階
- **展開／收起功能**：觸發「開始／收起功能」切換是否顯示音階列與簡化底部按鈕。
- **放鬆模式**：平均手速低於 40 px/s 並持續 5 秒會啟動，樂器音量降至 35%，環境音淡入；手速回到 65 px/s 以上會退出並淡出環境音。
- **離開程式**：觸發「結束程式 (Exit)」或按鍵盤 `q`。

## 常見問題
- **無法讀取攝影機**：確認其他程式未占用鏡頭，或在提示時手動輸入可用的攝影機索引。
- **無法播放音效**：確認喇叭可用、已安裝 FFmpeg 以便轉檔；缺少提示音或環境音檔會在終端提示並使用內建音源。
- **字體亂碼**：在系統中安裝 `FONT_PATHS` 列出的任一字型，或將自備字型路徑加入程式內的 `FONT_PATHS`。
- **效能或延遲問題**：降低相機解析度、關閉高負載應用程式，或改用以 USB 供電的外接攝影機。
