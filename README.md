# 專案使用說明

## 安裝環境

### 共通需求
- Python 3.9 以上（建議 3.10）。
- 於專案根目錄執行套件安裝：
  ```bash
  pip install -r requirements.txt
  ```

### Windows
- 建議使用安裝了「Desktop development with C++」元件的 Visual Studio Build Tools 以利部分套件編譯。
- 如需 FFmpeg / OpenCV 相關二進位：
  - 可使用 [Chocolatey](https://chocolatey.org/)：`choco install ffmpeg opencv`。
  - 或使用官方安裝包，並將安裝路徑加入 `PATH`。
- 第一次啟動時系統可能會要求相機、麥克風授權，請允許 Python / OpenCV 存取。
  - 若未跳出提示，可在 **設定 → 隱私權與安全性 → 相機/麥克風** 中允許 `python.exe` 或終端機應用程式。

### macOS
- 先確認已安裝 [Homebrew](https://brew.sh/)。
- 使用 Homebrew 安裝依賴：
  ```bash
  brew install ffmpeg opencv
  ```
- Apple Silicon 安裝 Mediapipe 建議：
  - 優先使用 arm64 wheels，例如：`pip install "mediapipe==0.10.11"`（已發佈 macOS arm64 版）或 `pip install mediapipe-silicon`。
  - 若遇到 arm64 版缺失，可改用 Rosetta x86_64 Python：
    ```bash
    arch -x86_64 /usr/bin/python3 -m venv .venv-x86
    source .venv-x86/bin/activate
    pip install mediapipe==0.10.11
    ```
  - 驗證安裝：
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    pip install "mediapipe==0.10.11"  # 或 mediapipe-silicon
    python - <<'PY'
    import mediapipe as mp; import cv2; import numpy as np; print('mediapipe', mp.__version__)
    PY
    ```
- 需要自行編譯或使用 Bazel/clang（例如調整 Mediapipe 原始碼）時，可先安裝並鎖定版本以降低失敗率：
  ```bash
  brew install bazelisk llvm
  export CC=/opt/homebrew/opt/llvm/bin/clang
  export CXX=/opt/homebrew/opt/llvm/bin/clang++
  bazelisk version  # 範例：macOS 14 (Apple Silicon) 常用的 bazelisk 7.x
  ```
- 首次執行時系統會詢問相機與麥克風權限，請選擇允許。
  - 若被阻擋，可至 **系統設定 → 隱私權與安全性 → 相機/麥克風** 勾選「Python」或「Terminal」允許存取。

## 啟動方式
在專案根目錄執行：
```bash
python main.py
```

## 常見問題與排除
- **缺少 OpenCV/FFmpeg 或找不到 DLL/動態庫**：確認已依照對應平台的安裝步驟安裝，並將可執行檔或動態庫路徑加入 `PATH`（Windows）或 `DYLD_LIBRARY_PATH`/`PATH`（macOS）。
- **相機/麥克風無法啟用或畫面黑屏**：檢查系統隱私權設定是否允許 Python/Terminal，重新啟動應用後再嘗試。
- **無法播放音訊或音效延遲**：確認音效裝置正常；如使用外接聲卡請於系統音效設定選擇正確輸出裝置。
- **啟動時出現權限被拒**：在 macOS 執行 `chmod +x` 給必要腳本，或在 Windows 以系統管理員身份執行終端。

## 資料夾解析
- `piano_sound/`: 存放鋼琴 do、re、mi、fa、so 的 WAV 檔。
- `violin_sound/`: 存放小提琴 do、re、mi、fa、so 的 WAV 檔。
- 觸發對應區塊會播放相應音檔。

## 變更紀錄
（以下為開發歷史備註）

1. 已為上方五個區塊加入音效觸發（do、re、mi、so、la）。在 main.py 引入 winsound，新增 TOP_ACTION_SOUNDS 對應各區塊到頻率，TOP_ACTION_DURATION_MS 控制音長。新增 play_action_sound，在觸發 功能一~功能五 時呼叫 winsound.Beep 撥放音效。檔案：main.py。如需改音色/音長，可調整 TOP_ACTION_SOUNDS 內的頻率或 TOP_ACTION_DURATION_MS。
2. 已將上方五個區塊的讀條參數獨立，並告知位置：新增常數：TOP_TRIGGER_THRESHOLD、TOP_ACCUMULATION_RATE、TOP_DECAY_RATE（上方區塊專用），原本的 TRIGGER_THRESHOLD、ACCUMULATION_RATE、DECAY_RATE 只管其他區塊。新增 helper：get_zone_params 依區塊名稱取回各自的讀條參數；build_zone_thresholds 生成與區塊對齊的門檻列表。draw_ui 現在支援每個區塊自己的 threshold（傳入列表）；主迴圈使用每區塊獨立的累加/衰減/觸發邏輯。相關程式碼位置：main.py 頂部參數區、get_zone_params/build_zone_thresholds 定義區，以及主偵測迴圈的累積/判斷邏輯與 draw_ui 呼叫。目前預設兩組參數數值相同；要調快/慢上方讀條，請調整 TOP_TRIGGER_THRESHOLD、TOP_ACCUMULATION_RATE、TOP_DECAY_RATE。
3. 已調整右下觸發行為：當右下（顯示/收起）區塊被觸發後，底部兩個區塊會改成「小提琴」「鋼琴」（功能暫留空）。同時會自動顯示上方五個區塊（若尚未展開）。
4. 已更新：右下「鋼琴」觸發後，上方五個區塊會改為鋼琴音階的 do、re、mi、fa、so，並播放對應音效。實作重點（main.py）：新增 TOP_ACTION_NAMES_PIANO = ["鋼琴 do", ... "鋼琴 so"] 並將音效映射加入 TOP_ACTION_SOUNDS。將上方區塊名稱集合改為可切換：TOP_ACTION_NAMES_DEFAULT、TOP_ACTION_NAMES_PIANO、TOP_ACTION_ALL。build_top_zones 支援傳入名稱列表；toggle_menu_visibility、get_zone_params 等都改用 TOP_ACTION_ALL。主迴圈中：觸發右下 鋼琴 時，設置 piano_top_requested，並在後續步驟將上方區塊替換為鋼琴音階區塊（若未展開會先展開），重建區域與讀條參數。音效播放對應新名稱，保持 do/re/mi/fa/so。
5. 已改為播放實際鋼琴 wav 檔：新增使用 simpleaudio，從 piano_sound/ 載入 c1.wav、d1.wav、e1.wav、f1.wav、g1.wav 對應 do/re/mi/fa/so，套用在原有五區塊與鋼琴模式。play_action_sound 改為以檔名查找、組合路徑並播放 wave，缺檔或播放失敗會列印訊息。requirements.txt 加入 simpleaudio。檔案：main.py、requirements.txt。若尚未安裝，請執行 pip install -r requirements.txt。
6. 已調整：右下（顯示/收起）觸發時會播放 Windows 嗶聲（winsound），並且再切換底部為鋼琴/小提琴。上方區塊：初始使用系統嗶聲（winsound.Beep）播放 do/re/mi/fa/so；當右下鋼琴被觸發後，上方區塊改為鋼琴音階並播放 piano_sound/ 中的 wav 檔。新增嗶聲/實音的分流邏輯：TOP_ACTION_FREQS（預設嗶聲）、TOP_ACTION_SOUNDS（鋼琴 wav），play_action_sound 依名稱挑選 winsound 或 simpleaudio。切換行為會額外嗶一聲提示。
7. 已調整：新增小提琴音階名稱與音檔：TOP_ACTION_NAMES_VIOLIN 及 violin_sound 下的 c3/d3/e3/f3/g3。TOP_ACTION_ALL 含預設、鋼琴、小提琴三組名稱；音效表 TOP_ACTION_SOUNDS 改存 (資料夾, 檔名)，小提琴對應 violin_sound。play_action_sound：預設名稱走 winsound 嗶聲，鋼琴/小提琴名稱改播放對應 wav。主迴圈：觸發左下「小提琴」會將上方五區塊切換成小提琴 do/re/mi/fa/so，並用小提琴音檔播放；鋼琴行為保持。仍保留右下切換時的系統嗶聲提示。檔案：main.py（已通過 python -m py_compile main.py）。
8. 完成音效切換邏輯：新增 WINDOWS_LABEL、TOP_ACTION_NAMES_WINDOWS，並將 Windows 五音加入讀條名單與嗶聲頻率。右下鋼琴/左下小提琴觸發後：該區塊會改成 Windows，並記錄來源樂器。觸發 Windows 區塊：上方五區塊切換為 Windows 嗶聲版 do/re/mi/fa/so，同時將該區塊改回原本的鋼琴或小提琴。觸發鋼琴/小提琴/Windows 上方區塊時，會播放對應的鋼琴/小提琴 wav 或 Windows 嗶聲。切換底部為樂器時會清空先前的來源記錄，避免混亂。檔案：main.py（已通過 python -m py_compile main.py）。
