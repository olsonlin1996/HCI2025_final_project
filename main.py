import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import time
import math
import wave
from datetime import datetime
import mediapipe as mp # 新增 MediaPipe 導入
import simpleaudio as sa
import winsound

# MediaPipe 手部偵測設定
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

FONT_PATHS = [
    # macOS/Linux 常用路徑或內建字體名稱
    "/System/Library/Fonts/Supplemental/Songti.ttc",  # macOS 內建宋體
    "/System/Library/Fonts/Supplemental/PingFang.ttc", # macOS 內建蘋方體
    "Arial Unicode MS.ttf",                           # 許多系統都有的通用字體名稱
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",   # Linux (Ubuntu) 常用中文字體
    # Windows 路徑
    "C:/Windows/Fonts/msjh.ttf",                      # Windows 微軟正黑體
    "C:/Windows/Fonts/mingliu.ttc",                   # Windows 細明體
    "msjh.ttf"
]
FONT_SIZE = 20

# --- 參數設定 ---
CAP_WIDTH = 1280
CAP_HEIGHT = 720
BOX_COLOR = (255, 0, 255) # 亮粉色
EXIT_LABEL = "結束程式 (Exit)"
SHOW_MENU_LABEL = "開始"
HIDE_MENU_LABEL = "收起功能"
PIANO_LABEL = "鋼琴"
VIOLIN_LABEL = "小提琴"
WINDOWS_LABEL = "Windows"
TOP_ACTION_NAMES_DEFAULT = ["do", "re", "mi", "fa", "so"]
TOP_ACTION_NAMES_PIANO = ["鋼琴 do", "鋼琴 re", "鋼琴 mi", "鋼琴 fa", "鋼琴 so"]
TOP_ACTION_NAMES_VIOLIN = ["小提琴 do", "小提琴 re", "小提琴 mi", "小提琴 fa", "小提琴 so"]
TOP_ACTION_NAMES_WINDOWS = ["Windows do", "Windows re", "Windows mi", "Windows fa", "Windows so"]
TOP_ACTION_ALL = TOP_ACTION_NAMES_DEFAULT + TOP_ACTION_NAMES_PIANO + TOP_ACTION_NAMES_VIOLIN + TOP_ACTION_NAMES_WINDOWS
PIANO_SOUND_DIR = "piano_sound"
VIOLIN_SOUND_DIR = "violin_sound"
TOP_ACTION_FREQS = {
    "do": 262,
    "re": 294,
    "mi": 330,
    "fa": 349,
    "so": 392,
    "Windows do": 262,
    "Windows re": 294,
    "Windows mi": 330,
    "Windows fa": 349,
    "Windows so": 392,
}
TOP_ACTION_SOUNDS = {
    "鋼琴 do": (PIANO_SOUND_DIR, "c1.wav"),
    "鋼琴 re": (PIANO_SOUND_DIR, "d1.wav"),
    "鋼琴 mi": (PIANO_SOUND_DIR, "e1.wav"),
    "鋼琴 fa": (PIANO_SOUND_DIR, "f1.wav"),
    "鋼琴 so": (PIANO_SOUND_DIR, "g1.wav"),
    "小提琴 do": (VIOLIN_SOUND_DIR, "c3.wav"),
    "小提琴 re": (VIOLIN_SOUND_DIR, "d3.wav"),
    "小提琴 mi": (VIOLIN_SOUND_DIR, "e3.wav"),
    "小提琴 fa": (VIOLIN_SOUND_DIR, "f3.wav"),
    "小提琴 so": (VIOLIN_SOUND_DIR, "g3.wav"),
}
TOP_ZONE_WIDTH = 200
TOP_ZONE_HEIGHT = 100
TOP_ZONE_START_Y = 0  # 貼齊上緣
COMMAND_ZONES = []


def build_top_zones(top_names=None):
    """
    建立上方五個區塊，左右貼齊邊界並平均分開。
    左側第一個 x=0，右側最後一個 x=CAP_WIDTH - W，中間等距分布。
    """
    names = top_names if top_names is not None else TOP_ACTION_NAMES_DEFAULT
    num = len(names)
    if num <= 1:
        xs = [0]
    else:
        span = max(0, CAP_WIDTH - TOP_ZONE_WIDTH)
        step = span / (num - 1)
        xs = [int(round(idx * step)) for idx in range(num)]

    return [(x, TOP_ZONE_START_Y, TOP_ZONE_WIDTH, TOP_ZONE_HEIGHT, name) for x, name in zip(xs, names)]


def ensure_toggle_label(zones, menu_visible):
    updated = []
    for x, y, w, h, name in zones:
        if name in (SHOW_MENU_LABEL, HIDE_MENU_LABEL):
            updated.append((x, y, w, h, HIDE_MENU_LABEL if menu_visible else SHOW_MENU_LABEL))
        else:
            updated.append((x, y, w, h, name))
    return updated


def toggle_menu_visibility(menu_visible, zones, top_zone_cache, top_names):
    """
    Toggle whether the top action row is visible. Keeps any edited positions for the
    top action zones by caching them while hidden.
    """
    new_menu_visible = not menu_visible

    # Keep base zones and strip out top actions for layout math.
    zones_without_top = [z for z in zones if z[4] not in TOP_ACTION_ALL]

    if new_menu_visible:
        if not top_zone_cache:
            top_zone_cache = build_top_zones(top_names)
        zones_with_menu = zones_without_top + top_zone_cache
    else:
        cached = [z for z in zones if z[4] in TOP_ACTION_ALL]
        if cached:
            top_zone_cache = cached
        zones_with_menu = zones_without_top

    zones_with_menu = ensure_toggle_label(zones_with_menu, new_menu_visible)
    return new_menu_visible, zones_with_menu, top_zone_cache


BASE_ZONES = [
    (50, 570, 200, 100, EXIT_LABEL),
    (1030, 570, 200, 100, SHOW_MENU_LABEL),
]
COMMAND_ZONES = ensure_toggle_label(BASE_ZONES.copy(), menu_visible=False)


def compute_velocity_factor(velocity: float):
    """Normalize velocity into [0, 1] for gain/pitch mapping."""
    # Velocity is in pixels/sec; clamp to avoid extreme scaling.
    normalized = max(0.0, min(velocity / 400.0, 1.0))
    return normalized


def velocity_to_color(velocity: float):
    """Map velocity to a visible color for UI feedback."""
    factor = compute_velocity_factor(velocity)
    # Blend between the base magenta and bright yellow as speed increases.
    base_color = np.array(BOX_COLOR)
    hot_color = np.array([255, 255, 0])
    blended = (base_color * (1 - factor) + hot_color * factor).astype(int)
    return tuple(int(c) for c in blended)


def load_scaled_wave(path: str, velocity: float):
    """Load a wave file and return a WaveObject scaled by velocity."""
    if not os.path.exists(path):
        print(f"找不到音檔：{path}")
        return None

    try:
        with wave.open(path, "rb") as wf:
            sample_rate = wf.getframerate()
            num_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())
    except Exception as e:
        print(f"讀取音檔失敗：{e}")
        return None

    audio_data = np.frombuffer(frames, dtype=np.int16)

    velocity_factor = compute_velocity_factor(velocity)
    gain = 0.4 + 0.6 * velocity_factor
    pitch_factor = 1.0 + 0.25 * velocity_factor

    # Apply gain with clipping protection
    scaled = np.clip(audio_data * gain, -32768, 32767).astype(np.int16)
    # Adjust playback rate to subtly pitch-shift according to movement speed
    adjusted_sample_rate = int(sample_rate * pitch_factor)

    return sa.WaveObject(scaled.tobytes(), num_channels, sample_width, adjusted_sample_rate)


def play_action_sound(action_name: str, velocity: float):
    """Play the mapped tone for the given top action if available."""
    if action_name in TOP_ACTION_FREQS:
        base_freq = TOP_ACTION_FREQS[action_name]
        pitch_factor = 1.0 + 0.25 * compute_velocity_factor(velocity)
        freq = int(base_freq * pitch_factor)
        try:
            winsound.Beep(freq, 200)
        except RuntimeError:
            print(f"無法播放系統嗶聲：{action_name}")
        return

    sound_info = TOP_ACTION_SOUNDS.get(action_name)
    if not sound_info:
        return
    sound_dir, filename = sound_info
    path = os.path.join(sound_dir, filename)

    wave_obj = load_scaled_wave(path, velocity)
    if wave_obj is None:
        return
    try:
        wave_obj.play()
    except Exception as e:
        print(f"播放失敗：{e}")

TRIGGER_THRESHOLD = 15
ACCUMULATION_RATE = 2
DECAY_RATE = 1
TOP_TRIGGER_THRESHOLD = 15
TOP_ACCUMULATION_RATE = 5
TOP_DECAY_RATE = 1


def get_zone_params(name: str):
    """Return (acc_rate, decay_rate, threshold) for zone by name."""
    if name in TOP_ACTION_ALL:
        return TOP_ACCUMULATION_RATE, TOP_DECAY_RATE, TOP_TRIGGER_THRESHOLD
    return ACCUMULATION_RATE, DECAY_RATE, TRIGGER_THRESHOLD


def build_zone_thresholds(zones):
    """Build per-zone thresholds list aligned with zones."""
    thresholds = []
    for _, _, _, _, name in zones:
        _, _, th = get_zone_params(name)
        thresholds.append(th)
    return thresholds


def build_instrument_bottom_zones():
    """Create piano/violin zones at the original bottom positions."""
    violin_zone = (BASE_ZONES[0][0], BASE_ZONES[0][1], BASE_ZONES[0][2], BASE_ZONES[0][3], VIOLIN_LABEL)
    piano_zone = (BASE_ZONES[1][0], BASE_ZONES[1][1], BASE_ZONES[1][2], BASE_ZONES[1][3], PIANO_LABEL)
    return [violin_zone, piano_zone]


def swap_bottom_to_instruments(zones):
    """Replace bottom control zones with piano/violin once triggered."""
    # Avoid duplicate swap
    if any(z[4] == PIANO_LABEL for z in zones) and any(z[4] == VIOLIN_LABEL for z in zones):
        return zones

    kept = [z for z in zones if z[4] not in (EXIT_LABEL, SHOW_MENU_LABEL, HIDE_MENU_LABEL)]
    return kept + build_instrument_bottom_zones()
GET_READY_SECONDS = 3    # 按下按鍵後的準備時間
CALIBRATION_SECONDS = 3  # 正式校準時間
VAR_THRESHOLD = 75
CAMERA_WAIT_TIMEOUT = 10 # 等待攝影機啟動的最長時間（秒）

# --- 繪圖函式 ---
def draw_ui(frame, zones, accumulators=None, threshold=None, zone_colors=None):
    # 偵錯：重新排序繪圖順序，確保框線總是可見
    for i, (x, y, w, h, name) in enumerate(zones):
        color = zone_colors[i] if zone_colors else BOX_COLOR
        # 1. 如果有提供進度，先畫進度條
        if accumulators is not None and threshold is not None:
            # threshold 可以是單一值或對應各區塊的列表
            zone_threshold = threshold[i] if isinstance(threshold, list) else threshold
            progress = min(accumulators[i] / zone_threshold, 1.0)
            if progress > 0 and zone_threshold > 0:
                cv2.rectangle(frame, (x, y), (x + int(w * progress), y + h), color, -1)

        # 2. 接著畫框線
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        
        # 3. 最後畫文字，確保文字在最上層
        # 使用 put_chinese_text 函式來顯示中文
        frame = put_chinese_text(frame, name, (x + 10, y + h - 15 - FONT_SIZE // 2), FONT_PATHS, FONT_SIZE, (255, 255, 255))
    return frame

def put_chinese_text(frame, text, position, font_paths, font_size, color):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    font = None
    for path in font_paths:
        try:
            font = ImageFont.truetype(path, font_size)
            break
        except IOError:
            continue

    if font is None:
        print(f"錯誤：找不到任何字體檔案於 {font_paths}，請確認路徑是否正確。將使用預設字體。")
        font = ImageFont.load_default()
    
    draw.text(position, text, font=font, fill=(color[2], color[1], color[0])) # OpenCV BGR to PIL RGB
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# --- 攝影機選擇函式 ---
def select_camera():
    print("正在偵測可用的攝影機...")
    available_cameras = []
    for i in range(5): # 嘗試檢查 0-4 的索引
        cap_test = cv2.VideoCapture(i)
        if cap_test.isOpened():
            available_cameras.append(i)

        cap_test.release()
    
    if not available_cameras:
        print("錯誤：找不到任何可用的攝影機。")
        return None
    
    if len(available_cameras) == 1:
        print(f"自動選擇唯一的攝影機: {available_cameras[0]}")
        return available_cameras[0]

    print("請選擇要使用的攝影機：")
    for cam_idx in available_cameras:
        print(f"  - 輸入 {cam_idx} 選擇攝影機 {cam_idx}")
    
    while True:
        try:
            choice = int(input("請輸入攝影機編號: "))
            if choice in available_cameras:
                return choice
            else:
                print("無效的選擇，請重新輸入。")
        except ValueError:
            print("請輸入數字。")

# --- 程式主體 ---
def main():
    global COMMAND_ZONES
    # --- 新增：攝影機選擇 ---
    camera_index = select_camera()
    if camera_index is None:
        return

    # --- 修改：耐心等待攝影機啟動 ---
    print(f"正在啟動攝影機 {camera_index}，這可能需要幾秒鐘...")
    # 不填第二參數，讓openCV自己選擇後端去打API（Windows與MacOS適用後端不同）
    cap = cv2.VideoCapture(camera_index)
    
    start_time = time.time()
    camera_ready = False
    while time.time() - start_time < CAMERA_WAIT_TIMEOUT:
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                camera_ready = True
                break
        # 短暫延遲，避免 CPU 占用過高
        time.sleep(0.1)

    if not camera_ready:
        print(f"錯誤：在 {CAMERA_WAIT_TIMEOUT} 秒內無法從攝影機讀取畫面。")
        print("請檢查攝影機是否被其他程式占用，或重新插拔攝影機。")
        cap.release()
        return

    print("攝影機已成功啟動！")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)

    # 介面狀態：預設隱藏上方五個功能區塊
    menu_visible = False
    current_top_names = TOP_ACTION_NAMES_DEFAULT
    top_zone_cache = build_top_zones(current_top_names)
    COMMAND_ZONES = ensure_toggle_label(BASE_ZONES.copy(), menu_visible)
    zone_thresholds = build_zone_thresholds(COMMAND_ZONES)
    window_origin = {}
    previous_hand_points = {}
    
    # 初始化 MediaPipe Hands
    hands = mp_hands.Hands(
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7)

    # 階段一：等待使用者按下按鍵
    window_name = "Hand Gesture Interface"
    cv2.namedWindow(window_name)

    while True:
        ret, frame = cap.read()
        if not ret: continue
        
        frame = cv2.flip(frame, 1)
        
        # 顯示操作提示
        cv2.putText(frame, "Press 's' to start calibration", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, "Press 'e' to edit layout", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, "Press 'q' to quit", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        frame = draw_ui(frame, COMMAND_ZONES, None, zone_thresholds)
        cv2.imshow(window_name, frame)
        
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            print("收到指令！準備開始校準...")
            break
        
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return

        elif key == ord('e'):
            COMMAND_ZONES = run_edit_mode(cap, COMMAND_ZONES, window_name, BOX_COLOR)
            if menu_visible:
                top_zone_cache = [z for z in COMMAND_ZONES if z[4] in TOP_ACTION_ALL] or top_zone_cache
            else:
                COMMAND_ZONES = [z for z in COMMAND_ZONES if z[4] not in TOP_ACTION_ALL]
            COMMAND_ZONES = ensure_toggle_label(COMMAND_ZONES, menu_visible)
            zone_thresholds = build_zone_thresholds(COMMAND_ZONES)
            # 繼續外層迴圈，等待 's' 或 'q'


    print("校準完成，可以開始操作！")
    zone_accumulators = [0] * len(COMMAND_ZONES)
    feedback_timers = [0] * len(COMMAND_ZONES)

    # 階段四：主偵測迴圈
    while True:
        zones_dirty = False
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)

        # 將 BGR 圖像轉換為 RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 處理圖像以偵測手部
        results = hands.process(frame_rgb)

        # 計算每隻手的速度
        hand_data = []
        h_frame, w_frame, _ = frame.shape
        current_time = time.time()
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label
                target_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                cx, cy = int(target_landmark.x * w_frame), int(target_landmark.y * h_frame)
                prev = previous_hand_points.get(label)
                velocity = 0.0
                if prev:
                    dist = math.hypot(cx - prev[0], cy - prev[1])
                    dt = current_time - prev[2]
                    if dt > 0:
                        velocity = dist / dt
                previous_hand_points[label] = (cx, cy, current_time)
                hand_data.append((hand_landmarks, label, velocity))

        # 繪製手部關鍵點
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        # --- 新增：處理視覺回饋 ---
        for i, start_time in enumerate(feedback_timers):
            if start_time > 0:
                elapsed_time = time.time() - start_time
                if elapsed_time < 5.0:  # 持續 5 秒
                    x, y, w, h, _ = COMMAND_ZONES[i]
                    
                    # 建立一個半透明的綠色疊加層
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), -1) # 綠色
                    alpha = 0.4  # 透明度
                    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
                else:
                    feedback_timers[i] = 0 # 重置計時器

        menu_toggle_requested = False
        instrument_swap_requested = False
        piano_top_requested = False
        violin_top_requested = False
        windows_top_requested = False
        exit_requested = False
        toggle_beep_requested = False
        zone_velocities = [0.0] * len(COMMAND_ZONES)

        for i, (x, y, w, h, name) in enumerate(COMMAND_ZONES):
            # 檢查是否有手部關鍵點在當前區域內
            hand_in_zone = False
            zone_velocity = 0.0
            if hand_data:
                for hand_landmarks, hand_label, hand_velocity in hand_data:
                    target_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                    cx, cy = int(target_landmark.x * w_frame), int(target_landmark.y * h_frame)

                    if x <= cx < x + w and y <= cy < y + h:
                        hand_in_zone = True
                        zone_velocity = max(zone_velocity, hand_velocity)

            acc_rate, decay_rate, zone_threshold = get_zone_params(name)

            if hand_in_zone:
                zone_accumulators[i] += acc_rate
            else:
                zone_accumulators[i] = max(0, zone_accumulators[i] - decay_rate)

            if zone_accumulators[i] > zone_threshold:
                print(f"指令觸發: {name}")
                feedback_timers[i] = time.time()

                if name == EXIT_LABEL:
                    exit_requested = True
                elif name in (SHOW_MENU_LABEL, HIDE_MENU_LABEL):
                    instrument_swap_requested = True
                    toggle_beep_requested = True
                elif name == PIANO_LABEL:
                    piano_top_requested = True
                    window_origin[i] = PIANO_LABEL
                    COMMAND_ZONES[i] = (x, y, w, h, WINDOWS_LABEL)
                    zones_dirty = True
                elif name == VIOLIN_LABEL:
                    violin_top_requested = True
                    window_origin[i] = VIOLIN_LABEL
                    COMMAND_ZONES[i] = (x, y, w, h, WINDOWS_LABEL)
                    zones_dirty = True
                elif name == WINDOWS_LABEL:
                    windows_top_requested = True
                    original = window_origin.get(i, PIANO_LABEL)
                    COMMAND_ZONES[i] = (x, y, w, h, original)
                    window_origin.pop(i, None)
                    zones_dirty = True
                elif name in TOP_ACTION_ALL:
                    play_action_sound(name, zone_velocity)
                    print(f"{name} 觸發（播放音效）。 速度: {zone_velocity:.1f}")

                zone_accumulators[i] = 0

            zone_velocities[i] = zone_velocity

            if exit_requested or menu_toggle_requested:
                break

        if exit_requested:
            cap.release()
            cv2.destroyAllWindows()
            return

        if toggle_beep_requested:
            try:
                winsound.Beep(880, 200)
            except RuntimeError:
                print("無法播放系統嗶聲（切換提示）。")

        if menu_toggle_requested:
            menu_visible, COMMAND_ZONES, top_zone_cache = toggle_menu_visibility(
                menu_visible, COMMAND_ZONES, top_zone_cache, current_top_names
            )
            zone_accumulators = [0] * len(COMMAND_ZONES)
            feedback_timers = [0] * len(COMMAND_ZONES)
            zone_thresholds = build_zone_thresholds(COMMAND_ZONES)
            continue

        if instrument_swap_requested:
            if not menu_visible:
                menu_visible, COMMAND_ZONES, top_zone_cache = toggle_menu_visibility(
                    menu_visible, COMMAND_ZONES, top_zone_cache, current_top_names
                )
            COMMAND_ZONES = swap_bottom_to_instruments(COMMAND_ZONES)
            window_origin = {}
            zone_accumulators = [0] * len(COMMAND_ZONES)
            feedback_timers = [0] * len(COMMAND_ZONES)
            zone_thresholds = build_zone_thresholds(COMMAND_ZONES)
            continue

        if piano_top_requested:
            current_top_names = TOP_ACTION_NAMES_PIANO
            # 確保上方區塊顯示並更新為鋼琴音階
            if not menu_visible:
                menu_visible, COMMAND_ZONES, top_zone_cache = toggle_menu_visibility(
                    menu_visible, COMMAND_ZONES, top_zone_cache, current_top_names
                )
            # 移除舊的上方區塊，替換成鋼琴音階的區塊
            COMMAND_ZONES = [z for z in COMMAND_ZONES if z[4] not in TOP_ACTION_ALL]
            top_zone_cache = build_top_zones(current_top_names)
            COMMAND_ZONES += top_zone_cache
            COMMAND_ZONES = ensure_toggle_label(COMMAND_ZONES, menu_visible)
            zone_accumulators = [0] * len(COMMAND_ZONES)
            feedback_timers = [0] * len(COMMAND_ZONES)
            zone_thresholds = build_zone_thresholds(COMMAND_ZONES)
            continue

        if violin_top_requested:
            current_top_names = TOP_ACTION_NAMES_VIOLIN
            if not menu_visible:
                menu_visible, COMMAND_ZONES, top_zone_cache = toggle_menu_visibility(
                    menu_visible, COMMAND_ZONES, top_zone_cache, current_top_names
                )
            COMMAND_ZONES = [z for z in COMMAND_ZONES if z[4] not in TOP_ACTION_ALL]
            top_zone_cache = build_top_zones(current_top_names)
            COMMAND_ZONES += top_zone_cache
            COMMAND_ZONES = ensure_toggle_label(COMMAND_ZONES, menu_visible)
            zone_accumulators = [0] * len(COMMAND_ZONES)
            feedback_timers = [0] * len(COMMAND_ZONES)
            zone_thresholds = build_zone_thresholds(COMMAND_ZONES)
            continue

        if windows_top_requested:
            current_top_names = TOP_ACTION_NAMES_WINDOWS
            if not menu_visible:
                menu_visible, COMMAND_ZONES, top_zone_cache = toggle_menu_visibility(
                    menu_visible, COMMAND_ZONES, top_zone_cache, current_top_names
                )
            COMMAND_ZONES = [z for z in COMMAND_ZONES if z[4] not in TOP_ACTION_ALL]
            top_zone_cache = build_top_zones(current_top_names)
            COMMAND_ZONES += top_zone_cache
            COMMAND_ZONES = ensure_toggle_label(COMMAND_ZONES, menu_visible)
            zone_accumulators = [0] * len(COMMAND_ZONES)
            feedback_timers = [0] * len(COMMAND_ZONES)
            zone_thresholds = build_zone_thresholds(COMMAND_ZONES)
            continue
        
        if zones_dirty:
            zone_thresholds = build_zone_thresholds(COMMAND_ZONES)

        zone_colors = [velocity_to_color(v) for v in zone_velocities]
        frame = draw_ui(frame, COMMAND_ZONES, zone_accumulators, zone_thresholds, zone_colors=zone_colors)

        cv2.imshow("Hand Gesture Interface", frame)
        # cv2.imshow("Foreground Mask", fg_mask) # 移除此行

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    hands.close() # 關閉 MediaPipe Hands 資源
    print("程式已結束。")

if __name__ == '__main__':
    main()
