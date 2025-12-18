import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import time
import math
import wave
import subprocess
import tempfile
from collections import deque
from datetime import datetime
import platform
import mediapipe as mp
import simpleaudio as sa

try:
    import winsound
except ImportError:
    winsound = None

# MediaPipe 手部偵測設定
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

FONT_PATHS = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Supplemental/PingFang.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
    "Arial Unicode MS.ttf",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "C:/Windows/Fonts/msjh.ttf",
    "C:/Windows/Fonts/mingliu.ttc",
    "msjh.ttf"
]
FONT_SIZE = 30
_FONT_CACHE = {}
_FONT_WARNING_SHOWN = False

# 參數
CAP_WIDTH = 1280
CAP_HEIGHT = 720
BOX_COLOR = (182, 175, 164),  # 霧霾藍 (Haze Blue)
COLOR_TOP_NOTE = (166, 149, 158)  # 灰紫 (Grey Purple)
COLOR_NAV_BTN  = (122, 157, 138)  # 鼠尾草綠 (Sage Green)
COLOR_SYS_BTN  = (146, 166, 186)  # 礦石灰 (Mineral Grey)

EXIT_LABEL = "結束程式 (Exit)"
SHOW_MENU_LABEL = "開始"
HIDE_MENU_LABEL = "收起功能"

# 音色資料夾
PIANO_SOUND_DIR = "piano_sound"
VIOLIN_SOUND_DIR = "violin_sound"

# 音階名稱
NAMES_CHINESE = ["宮 (Do)", "商 (Re)", "角 (Mi)", "徵 (So)", "羽 (La)"]
NAMES_RITSU = ["律 一 (Do)", "律 二 (Re)", "律 三 (Fa)", "律 四 (So)", "律 五 (La)"]
NAMES_RYO = ["呂 一 (Re)", "呂 二 (Mi)", "呂 三 (So)", "呂 四 (La)", "呂 五 (Ti)"]

# 整合名稱
TOP_ACTION_ALL = tuple(NAMES_CHINESE + NAMES_RITSU + NAMES_RYO)

# 音檔對照表
NOTE_FILES_PIANO = {
    "do": "c1.wav", "re": "d1.wav", "mi": "e1.wav", "fa": "f1.wav", 
    "so": "g1.wav", "la": "a1.wav", "ti": "b1.wav",
}

NOTE_FILES_VIOLIN = {
    "do": "c3.wav", "re": "d3.wav", "mi": "e3.wav", "fa": "f3.wav", 
    "so": "g3.wav", "la": "a3.wav", "ti": "b3.wav",
}

# 缺檔時的備用頻率
TOP_ACTION_FREQS = {
    "do": 262, # C4
    "re": 294, # D4
    "mi": 330, # E4
    "fa": 349, # F4
    "so": 392, # G4
    "la": 440, # A4
    "ti": 494  # B4
}

SCALE_PRESETS = [
    # 鋼琴系列
    {
        "key": "piano_chinese",
        "label": "[鋼琴] 中國五聲",
        "names": NAMES_CHINESE,
        "note_keys": ["do", "re", "mi", "so", "la"],
        "sound_dir": PIANO_SOUND_DIR,
        "file_map": NOTE_FILES_PIANO  # 指定用鋼琴檔名
    },
    {
        "key": "piano_ritsu",
        "label": "[鋼琴] 日本律音階",
        "names": NAMES_RITSU,
        "note_keys": ["do", "re", "fa", "so", "la"],
        "sound_dir": PIANO_SOUND_DIR,
        "file_map": NOTE_FILES_PIANO
    },
    {
        "key": "piano_ryo",
        "label": "[鋼琴] 日本呂音階",
        "names": NAMES_RYO,
        "note_keys": ["re", "mi", "so", "la", "ti"],
        "sound_dir": PIANO_SOUND_DIR,
        "file_map": NOTE_FILES_PIANO
    },
    # 小提琴系列
    {
        "key": "violin_chinese",
        "label": "[小提琴] 中國五聲",
        "names": NAMES_CHINESE,
        "note_keys": ["do", "re", "mi", "so", "la"],
        "sound_dir": VIOLIN_SOUND_DIR,
        "file_map": NOTE_FILES_VIOLIN # 指定用小提琴檔名
    },
    {
        "key": "violin_ritsu",
        "label": "[小提琴] 日本律音階",
        "names": NAMES_RITSU,
        "note_keys": ["do", "re", "fa", "so", "la"],
        "sound_dir": VIOLIN_SOUND_DIR,
        "file_map": NOTE_FILES_VIOLIN
    },
    {
        "key": "violin_ryo",
        "label": "[小提琴] 日本呂音階",
        "names": NAMES_RYO,
        "note_keys": ["re", "mi", "so", "la", "ti"],
        "sound_dir": VIOLIN_SOUND_DIR,
        "file_map": NOTE_FILES_VIOLIN
    },
]

ACTION_NOTE_KEYS = {}
TOP_ACTION_SOUNDS = {}

TIMBRE_LAYERS = []     
TIMBRE_MODES = []      
TIMBRE_MODE_LABEL = ""

TOP_ZONE_WIDTH = 200
TOP_ZONE_HEIGHT = 540
TOP_ZONE_START_Y = 0  # 貼齊上緣
COMMAND_ZONES = []
AMBIENT_SOUND_FILES = [
    ("ocean", os.path.join("ambient_sound", "ocean.wav")),
    ("wind", os.path.join("ambient_sound", "wind.wav")),
]
AMBIENT_SOUND_PATH = AMBIENT_SOUND_FILES[0][1]
AMBIENT_TARGET_GAIN = 0.85
RELAX_VELOCITY_THRESHOLD = 40.0
RELAX_EXIT_VELOCITY = 65.0
RELAX_ACCUM_THRESHOLD = 4.0
RELAX_TRIGGER_SECONDS = 5.0
GAIN_FADE_SECONDS = 2.5
VISUAL_DIM_MAX = 0.45
RELAX_INSTRUMENT_GAIN = 0.35
INSTRUMENT_MASTER_GAIN = 1.0
BEEP_SOUND_PATH = os.path.join("assets", "beep.wav")
BEEP_WAVE = None
BEEP_WARNING_SHOWN = False
IS_WINDOWS = platform.system() == "Windows"

def get_nav_labels(current_idx):
    """回傳上一個/下一個樂器文字。"""
    n = len(SCALE_PRESETS)
    prev_idx = (current_idx - 1 + n) % n
    next_idx = (current_idx + 1) % n
    prev_label = f"< {SCALE_PRESETS[prev_idx]['label']}"
    next_label = f"{SCALE_PRESETS[next_idx]['label']} >"
    
    return prev_label, next_label

def build_top_zones(top_names=None):
    """建立上方區塊，左右貼齊並等距。"""
    names = top_names if top_names is not None else TOP_ACTION_NAMES_PIANO
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


def load_active_preset_sounds(preset):
    """切換樂器時重載當前預設的音效。"""
    TOP_ACTION_SOUNDS.clear()
    ACTION_NOTE_KEYS.clear()

    current_file_map = preset["file_map"]
    
    for display_name, note_key in zip(preset["names"], preset["note_keys"]):
        ACTION_NOTE_KEYS[display_name] = note_key
        
        if preset["sound_dir"]:
            filename = current_file_map.get(note_key)
            if filename:
                TOP_ACTION_SOUNDS[display_name] = (preset["sound_dir"], filename)


def toggle_menu_visibility(menu_visible, zones, top_zone_cache, top_names):
    """切換上方區塊顯示並維持暫存位置。"""
    new_menu_visible = not menu_visible

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
    (40, 560, 200, 140, EXIT_LABEL),
    (790, 560, 200, 140, TIMBRE_MODE_LABEL),
    (1040, 560, 200, 140, SHOW_MENU_LABEL),
]
COMMAND_ZONES = ensure_toggle_label(BASE_ZONES.copy(), menu_visible=False)


def compute_velocity_factor(velocity: float):
    """把速度正規化到 0-1。"""
    normalized = max(0.0, min(velocity / 400.0, 1.0))
    return normalized

def velocity_to_color(velocity: float, base_color):
    """速度越快顏色越亮。"""
    factor = compute_velocity_factor(velocity)
    
    base_arr = np.array(base_color)
    hot_arr = np.array([255, 255, 0]) 
    
    blended = (base_arr * (1 - factor) + hot_arr * factor).astype(int)
    return tuple(int(c) for c in blended)

def clamp01(value: float):
    return max(0.0, min(value, 1.0))


# 視覺效果設定
VISUAL_BUFFER_DOWNSCALE = 4
MOTION_HISTORY_FRAMES = 30
MOTION_VELOCITY_NORM = 160.0
MOTION_VARIANCE_NORM = 8000.0


def init_visual_state():
    return {
        "shape": None,
        "phase": 0.0,
        "x_coords": None,
        "y_coords": None,
        "buffer": None,
    }


def ensure_visual_state(state, frame_shape):
    h, w = frame_shape[:2]
    if state["shape"] == (h, w):
        return state

    small_h = max(1, h // VISUAL_BUFFER_DOWNSCALE)
    small_w = max(1, w // VISUAL_BUFFER_DOWNSCALE)
    xs = np.linspace(0, 2 * math.pi, small_w, dtype=np.float32)[None, :]
    ys = np.linspace(0, 2 * math.pi, small_h, dtype=np.float32)[:, None]
    state.update(
        {
            "shape": (h, w),
            "phase": 0.0,
            "x_coords": xs,
            "y_coords": ys,
            "buffer": np.zeros((small_h, small_w, 3), dtype=np.float32),
        }
    )
    return state


def compute_motion_metrics(recent_velocities, current_velocity):
    velocity_norm = clamp01(current_velocity / MOTION_VELOCITY_NORM)
    variance = np.var(recent_velocities) if len(recent_velocities) > 1 else 0.0
    variance_norm = clamp01(variance / MOTION_VARIANCE_NORM)
    motion_intensity = clamp01(0.65 * velocity_norm + 0.35 * variance_norm)
    alpha = 0.12 + 0.5 * motion_intensity
    return {
        "velocity_norm": velocity_norm,
        "variance": variance,
        "variance_norm": variance_norm,
        "motion_intensity": motion_intensity,
        "alpha": clamp01(alpha),
    }


def update_visual_buffer(state, motion_metrics, dt):
    if state.get("buffer") is None:
        return state

    phase_speed = 0.6 + 2.4 * motion_metrics["motion_intensity"]
    state["phase"] += phase_speed * dt
    freq = 1.0 + 3.0 * motion_metrics["variance_norm"]
    amplitude = 0.35 + 0.65 * motion_metrics["motion_intensity"]

    x_term = np.sin(state["x_coords"] * freq + state["phase"])
    y_term = np.cos(state["y_coords"] * (freq * 0.6 + 0.4) + state["phase"] * 1.35)
    wave = (x_term + y_term + 2.0) * 0.25
    wave = np.power(wave, 1.2) * amplitude

    cold_color = np.array([40, 80, 140], dtype=np.float32)
    hot_color = np.array([160, 240, 255], dtype=np.float32)
    state["buffer"] = cold_color + (hot_color - cold_color) * wave[:, :, None]
    return state


def render_visual_overlay(state):
    if state.get("buffer") is None:
        return None

    h, w = state["shape"]
    overlay = cv2.resize(
        state["buffer"].astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC
    )
    return np.clip(overlay, 0, 255).astype(np.uint8)


def blend_overlay_on_zones(frame, overlay, zones, base_alpha, highlights=None):
    if overlay is None or base_alpha <= 0:
        return frame

    blended = frame.copy()
    for idx, (x, y, w, h, _) in enumerate(zones):
        zone_alpha = base_alpha
        
        # highlights 比 zones 短時也不會越界
        if highlights and idx < len(highlights) and highlights[idx]:
            zone_alpha = clamp01(base_alpha * 1.3)

        if zone_alpha <= 0:
            continue
            
        if y + h > frame.shape[0] or x + w > frame.shape[1]:
            continue

        overlay_region = overlay[y : y + h, x : x + w]
        target_region = blended[y : y + h, x : x + w]
        
        if overlay_region.shape[:2] != target_region.shape[:2]:
            continue
            
        blended[y : y + h, x : x + w] = (
            overlay_region * zone_alpha + target_region * (1 - zone_alpha)
        ).astype(np.uint8)

    return blended

def schedule_fade(current: float, target: float, duration: float):
    return {
        "start": time.time(),
        "from": current,
        "to": target,
        "duration": max(0.01, duration),
    }


def step_fade(current: float, fade_info):
    if not fade_info:
        return current, None

    elapsed = time.time() - fade_info["start"]
    duration = fade_info["duration"]
    progress = clamp01(elapsed / duration) if duration else 1.0
    new_value = fade_info["from"] + (fade_info["to"] - fade_info["from"]) * progress

    if progress >= 1.0:
        return new_value, None
    fade_info["last"] = new_value
    return new_value, fade_info


def ensure_standard_wav(input_path: str):
    """用 ffmpeg 轉成 44100Hz、16-bit、雙聲道的暫存 wav。"""
    fd, temp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    cmd = [
        "ffmpeg", "-y", "-v", "error", "-i", input_path,
        "-ar", "44100", "-ac", "2", "-sample_fmt", "s16", "-f", "wav", temp_path
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return temp_path
    except FileNotFoundError:
        print("找不到 ffmpeg，請先安裝後再試。")
        if os.path.exists(temp_path): os.remove(temp_path)
        return None
    except Exception as e:
        if os.path.exists(temp_path): os.remove(temp_path)
        return None

def load_wave_data(path: str):
    """讀取音檔，不合規就自動轉檔。"""
    if not os.path.exists(path):
        print(f"找不到音檔：{path}")
        return None

    temp_wav_path = None
    needs_convert = False

    try:
        with wave.open(path, "rb") as wf:
            if wf.getframerate() != 44100 or wf.getsampwidth() != 2:
                needs_convert = True
    except:
        needs_convert = True

    source_path = path

    if needs_convert:
        temp_wav_path = ensure_standard_wav(path)
        if not temp_wav_path:
            return None
        source_path = temp_wav_path

    try:
        with wave.open(source_path, "rb") as wf:
            sample_rate = wf.getframerate()
            num_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())
            audio_data = np.frombuffer(frames, dtype=np.int16)
            return sample_rate, num_channels, sample_width, audio_data
    except Exception as e:
        print(f"讀取音檔失敗：{e}")
        return None
    finally:
        if temp_wav_path and os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)


def load_scaled_wave(path: str, velocity: float):
    """Load a wave file and return a WaveObject scaled by velocity."""
    wave_data = load_wave_data(path)
    if not wave_data:
        return None

    sample_rate, num_channels, sample_width, audio_data = wave_data
    velocity_factor = compute_velocity_factor(velocity)
    
    gain = (0.4 + 0.6 * velocity_factor) * INSTRUMENT_MASTER_GAIN

    scaled = np.clip(audio_data * gain, -32768, 32767).astype(np.int16)
    return sa.WaveObject(scaled.tobytes(), num_channels, sample_width, sample_rate)

def build_wave_with_gain(wave_data, gain: float):
    """Scale a pre-loaded wave tuple to the desired gain and return a WaveObject."""
    if not wave_data:
        return None

    sample_rate, num_channels, sample_width, audio_data = wave_data
    scaled = np.clip(audio_data * gain, -32768, 32767).astype(np.int16)
    return sa.WaveObject(scaled.tobytes(), num_channels, sample_width, sample_rate)


def generate_fallback_ambient(duration: float = 2.5, sample_rate: int = 44100):
    """產生一段柔和環境音當備用。"""
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    slow_wave = 0.15 * np.sin(2 * math.pi * 0.35 * t)
    shimmer = 0.08 * np.sin(2 * math.pi * 0.85 * t + math.pi / 3)
    noise = np.random.normal(0, 0.04, num_samples)
    mix = slow_wave + shimmer + noise
    stereo = np.stack([mix, mix * 0.9], axis=-1)
    audio = np.clip(stereo * 32767, -32768, 32767).astype(np.int16)
    return sample_rate, 2, 2, audio


def resolve_ambient_path():
    """回傳可用的環境音檔路徑，沒有就給警告。"""
    for _label, path in AMBIENT_SOUND_FILES:
        if os.path.exists(path):
            return path, None

    expected_locations = ", ".join(path for _label, path in AMBIENT_SOUND_FILES)
    warning = (
        "找不到環境音檔，將改用內建合成環境音。"
        f"請將檔案放在：{expected_locations}"
    )
    return AMBIENT_SOUND_PATH, warning


def load_ambient_or_fallback(path: str):
    """有檔案就載入，沒有就用合成版本。"""
    wave_data = load_wave_data(path)
    if wave_data:
        return wave_data, None

    generated = generate_fallback_ambient()
    warning = f"找不到環境音檔：{path}，改用內建環境音。"
    return generated, warning


def _generate_beep_wave(freq: int = 880, duration_ms: int = 200, sample_rate: int = 44100):
    """產生一段簡短提示音。"""
    duration = duration_ms / 1000.0
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = 0.35 * np.sin(2 * math.pi * freq * t)
    audio = np.clip(tone * 32767, -32768, 32767).astype(np.int16)
    return sa.WaveObject(audio.tobytes(), 1, 2, sample_rate)


def load_beep_wave():
    """載入提示音，缺檔時就動態生成。"""
    global BEEP_WAVE
    if os.path.exists(BEEP_SOUND_PATH):
        try:
            BEEP_WAVE = sa.WaveObject.from_wave_file(BEEP_SOUND_PATH)
            return True
        except Exception as e:
            print(f"載入提示音失敗：{e}")

    try:
        BEEP_WAVE = _generate_beep_wave()
    except Exception as e:
        print(f"生成內建提示音失敗：{e}")
        BEEP_WAVE = None

    return BEEP_WAVE is not None


def ensure_beep_wave_loaded():
    global BEEP_WARNING_SHOWN
    beep_ready = BEEP_WAVE is not None or load_beep_wave()

    if not beep_ready and not BEEP_WARNING_SHOWN:
        print("警告：無法載入或生成提示音，預設嗶聲可能無法播放。")
        BEEP_WARNING_SHOWN = True

    return beep_ready


def play_beep_sound(freq: int = 880, duration_ms: int = 200):
    """能用系統嗶聲就用，否則放內建音。"""
    if IS_WINDOWS and winsound:
        try:
            winsound.Beep(freq, duration_ms)
            return
        except RuntimeError:
            print("無法播放系統嗶聲，改用內建提示音。")

    if ensure_beep_wave_loaded() and BEEP_WAVE:
        try:
            BEEP_WAVE.play()
            return
        except Exception as e:
            print(f"播放提示音失敗：{e}")


def normalize_hand_x(hand_x: float):
    if hand_x is None:
        return 0.5
    return max(0.0, min(hand_x / CAP_WIDTH, 1.0))

def play_action_sound(action_name, velocity, hand_x):
    """播放對應的 WAV，沒有就用嗶聲備援。"""
    sound_info = TOP_ACTION_SOUNDS.get(action_name)
    if sound_info:
        sound_dir, filename = sound_info
        path = os.path.join(sound_dir, filename)
        wave_obj = load_scaled_wave(path, velocity)
        if wave_obj:
            wave_obj.play()
            return

    found_key = None
    for key in TOP_ACTION_FREQS.keys():
        if key in action_name.lower() or f"({key})" in action_name.lower():
            found_key = key
            break
    
    if not found_key:
        for note in ["do", "re", "mi", "fa", "so", "la", "ti"]:
            if note in action_name.lower():
                found_key = note
                break

    if found_key:
        base_freq = TOP_ACTION_FREQS.get(found_key, 440)
        pitch_factor = 1.0 + 0.05 * compute_velocity_factor(velocity) 
        freq = int(base_freq * pitch_factor)
        play_beep_sound(freq, 200)
    else:
        print(f"警告：找不到音效或頻率定義: {action_name}")


TRIGGER_THRESHOLD = 15
ACCUMULATION_RATE = 2
DECAY_RATE = 1
TOP_TRIGGER_THRESHOLD = 15
TOP_ACCUMULATION_RATE = 5
TOP_DECAY_RATE = 1


def get_zone_params(name: str):
    """回傳區塊的累積與閾值設定。"""
    if name in TOP_ACTION_ALL:
        return TOP_ACCUMULATION_RATE, TOP_DECAY_RATE, TOP_TRIGGER_THRESHOLD
    return ACCUMULATION_RATE, DECAY_RATE, TRIGGER_THRESHOLD


def build_zone_thresholds(zones):
    """依序建立每個區塊的閾值清單。"""
    thresholds = []
    for _, _, _, _, name in zones:
        _, _, th = get_zone_params(name)
        thresholds.append(th)
    return thresholds


def get_preset_by_key(key: str):
    for idx, preset in enumerate(SCALE_PRESETS):
        if preset["key"] == key:
            return idx, preset
    return 0, SCALE_PRESETS[0]


def rebuild_top_for_preset(preset_idx, menu_visible, zones):
    """切換上方琴鍵並重建狀態。"""
    preset = SCALE_PRESETS[preset_idx]
    
    load_active_preset_sounds(preset)

    zones_without_top = [z for z in zones if z[4] not in TOP_ACTION_ALL]
    top_zone_cache = build_top_zones(preset["names"])

    if menu_visible:
        zones = zones_without_top + top_zone_cache
    else:
        zones = zones_without_top

    zones = ensure_toggle_label(zones, menu_visible)
    zone_accumulators = [0] * len(zones)
    feedback_timers = [0] * len(zones)
    zone_thresholds = build_zone_thresholds(zones)
    
    return (
        preset["names"],
        top_zone_cache,
        zones,
        zone_accumulators,
        feedback_timers,
        zone_thresholds,
    )


NAV_BTN_LEFT_RECT = (0, 0, 0, 0)
NAV_BTN_RIGHT_RECT = (0, 0, 0, 0)

def build_instrument_bottom_zones(prev_label, next_label):
    """用預先算好的座標建立底部導航。"""
    left_zone = (*NAV_BTN_LEFT_RECT, prev_label)
    right_zone = (*NAV_BTN_RIGHT_RECT, next_label)
    return [left_zone, right_zone]

def swap_bottom_to_instruments(zones, current_idx):
    """重組底部按鈕，保留上方琴鍵。"""
    prev_label, next_label = get_nav_labels(current_idx)
    nav_left = (*NAV_BTN_LEFT_RECT, prev_label)
    nav_right = (*NAV_BTN_RIGHT_RECT, next_label)
    
    center_buttons = BASE_ZONES.copy()
    
    top_kept = [z for z in zones if z[4] in TOP_ACTION_ALL]
    return [nav_left] + center_buttons + [nav_right] + top_kept

GET_READY_SECONDS = 3    # 按下按鍵後的準備時間
CALIBRATION_SECONDS = 3  # 正式校準時間
VAR_THRESHOLD = 75
CAMERA_WAIT_TIMEOUT = 10 # 等待攝影機啟動的最長時間（秒）

# 繪圖
def draw_ui(frame, zones, accumulators=None, threshold=None, zone_colors=None):
    for i, (x, y, w, h, name) in enumerate(zones):
        raw_color = zone_colors[i] if zone_colors else BOX_COLOR
        if isinstance(raw_color, (list, tuple)) and len(raw_color) > 0 and isinstance(raw_color[0], (list, tuple)):
            raw_color = raw_color[0]
        try: 
            color = tuple(int(c) for c in raw_color)
        except: 
            color = (0, 255, 0)
        
        x, y, w, h = int(x), int(y), int(w), int(h)
        
        if accumulators is not None and threshold is not None:
            if i >= len(accumulators):
                accumulators.append(0.0)
            
            t = threshold[i] if isinstance(threshold, list) else threshold
            if t > 0:
                progress = min(accumulators[i] / t, 1.0)
                if progress > 0: 
                    cv2.rectangle(frame, (x, y), (x + int(w * progress), y + h), color, -1)
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
        frame = put_chinese_text(frame, name, (x + 10, y + h - 15 - FONT_SIZE // 2), FONT_PATHS, FONT_SIZE, (255, 255, 255))
        
    return frame

def put_chinese_text(frame, text, position, font_paths, font_size, color):
    global _FONT_WARNING_SHOWN
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    font = None
    for path in font_paths:
        cache_key = (path, font_size)
        if cache_key in _FONT_CACHE:
            cached = _FONT_CACHE[cache_key]
            if cached:
                font = cached
                break
            continue
        try:
            loaded_font = ImageFont.truetype(path, font_size)
            _FONT_CACHE[cache_key] = loaded_font
            font = loaded_font
            break
        except IOError:
            _FONT_CACHE[cache_key] = None

    if font is None:
        if not _FONT_WARNING_SHOWN:
            print(f"錯誤：找不到任何字體檔案於 {font_paths}，請確認路徑是否正確。將使用預設字體。")
            _FONT_WARNING_SHOWN = True
        font = ImageFont.load_default()
    
    draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def init_four_buttons(current_scale_idx, menu_visible):
    """初始化底部按鈕：根據是否已開始，決定中間顯示幾顆按鈕"""
    global NAV_BTN_LEFT_RECT, NAV_BTN_RIGHT_RECT, BASE_ZONES
    
    btn_w = 200
    btn_h = 100
    margin_side = 20
    margin_bottom = 30
    btn_y = CAP_HEIGHT - btn_h - margin_bottom
    gap = 20

    NAV_BTN_LEFT_RECT = (margin_side, btn_y, btn_w, btn_h)
    NAV_BTN_RIGHT_RECT = (CAP_WIDTH - btn_w - margin_side, btn_y, btn_w, btn_h)

    if not menu_visible:
        total_center_width = btn_w * 2 + gap
        center_start_x = (CAP_WIDTH - total_center_width) // 2
        
        BASE_ZONES = [
            (center_start_x, btn_y, btn_w, btn_h, EXIT_LABEL),
            (center_start_x + btn_w + gap, btn_y, btn_w, btn_h, SHOW_MENU_LABEL),
        ]
    else:
        total_center_width = btn_w
        center_start_x = (CAP_WIDTH - total_center_width) // 2
        
        BASE_ZONES = [
            (center_start_x, btn_y, btn_w, btn_h, EXIT_LABEL),
        ]
    
    zones = swap_bottom_to_instruments(BASE_ZONES, current_scale_idx)
    zones = ensure_toggle_label(zones, menu_visible)
    return zones

def select_camera():
    print("正在偵測可用的攝影機...")
    available_cameras = []
    for i in range(5):
        cap_test = cv2.VideoCapture(i)
        if cap_test.isOpened():
            available_cameras.append(i)
        cap_test.release()
    
    if not available_cameras:
        print("錯誤：找不到任何可用的攝影機。")
        return None
    
    suggested_idx = available_cameras[-1]
    print(f"自動嘗試連接攝影機 {suggested_idx} (通常為外接鏡頭)...")

    cap = cv2.VideoCapture(suggested_idx)
    is_working = False
    if cap.isOpened():
        # 嘗試讀取一幀，確認真的能運作
        ret, _ = cap.read()
        if ret:
            is_working = True
    cap.release()

    if is_working:
        print(f"成功連接攝影機 {suggested_idx}！")
        return suggested_idx
    else:
        print(f"自動連接攝影機 {suggested_idx} 失敗，轉為手動選擇模式。")

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

def main():
    global COMMAND_ZONES, INSTRUMENT_MASTER_GAIN

    ensure_beep_wave_loaded()
    camera_index = select_camera()
    if camera_index is None:
        return

    print(f"正在啟動攝影機 {camera_index}，這可能需要幾秒鐘...")
    cap = cv2.VideoCapture(camera_index)
    
    start_time = time.time()
    camera_ready = False
    while time.time() - start_time < CAMERA_WAIT_TIMEOUT:
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                camera_ready = True
                break
        time.sleep(0.1)

    if not camera_ready:
        print(f"錯誤：在 {CAMERA_WAIT_TIMEOUT} 秒內無法從攝影機讀取畫面。")
        print("請檢查攝影機是否被其他程式占用，或重新插拔攝影機。")
        cap.release()
        return

    print("攝影機已成功啟動！")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)

    menu_visible = False
    current_scale_idx, current_preset = get_preset_by_key("piano_chinese")
    load_active_preset_sounds(current_preset)
    current_timbre_idx = 0
    current_top_names = current_preset["names"]
    top_zone_cache = build_top_zones(current_top_names)

    COMMAND_ZONES = init_four_buttons(current_scale_idx, menu_visible)
    zone_thresholds = build_zone_thresholds(COMMAND_ZONES)
    window_origin = {}
    previous_hand_points = {}
    recent_velocities = deque(maxlen=MOTION_HISTORY_FRAMES)
    visual_state = init_visual_state()
    last_effect_time = time.time()
    ambient_path, ambient_precheck_warning = resolve_ambient_path()
    ambient_loop_data, ambient_warning = load_ambient_or_fallback(ambient_path)
    ambient_warning = ambient_warning or ambient_precheck_warning
    if ambient_warning:
        print(ambient_warning)
    ambient_play_obj = None
    instrument_gain = 1.0
    ambient_gain = 0.0
    instrument_fade = None
    ambient_fade = None
    relax_state = "active"
    relax_candidate_start = None
    ambient_warning_shown = ambient_warning is not None
    
    hands = mp_hands.Hands(
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7)

    window_name = "Hand Gesture Interface"
    cv2.namedWindow(window_name)

    while True:
        ret, frame = cap.read()
        if not ret: continue
        
        frame = cv2.flip(frame, 1)
        
        cv2.putText(frame, "Press 's' to start calibration", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, "Press 'q' to quit", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, "Low motion will enter Relax mode (ambient)", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 200), 2)

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


    print("校準完成，可以開始操作！")

    COMMAND_ZONES = swap_bottom_to_instruments(COMMAND_ZONES, current_scale_idx)
    
    COMMAND_ZONES = ensure_toggle_label(COMMAND_ZONES, menu_visible)

    zone_accumulators = [0] * len(COMMAND_ZONES)
    feedback_timers = [0] * len(COMMAND_ZONES)
    zone_thresholds = build_zone_thresholds(COMMAND_ZONES)

    while True:
        zones_dirty = False
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

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

        avg_hand_velocity = (
            sum(v for _, _, v in hand_data) / len(hand_data)
            if hand_data
            else 0.0
        )

        recent_velocities.append(avg_hand_velocity)
        motion_metrics = compute_motion_metrics(recent_velocities, avg_hand_velocity)
        visual_state = ensure_visual_state(visual_state, frame.shape)
        now = time.time()
        dt = max(1e-3, now - last_effect_time)
        visual_state = update_visual_buffer(visual_state, motion_metrics, dt)
        last_effect_time = now

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        if len(feedback_timers) < len(COMMAND_ZONES):
            feedback_timers.extend([0] * (len(COMMAND_ZONES) - len(feedback_timers)))

        active_feedback = [False] * len(feedback_timers)
        
        for i, start_time in enumerate(feedback_timers):
            if start_time > 0:
                elapsed_time = time.time() - start_time
                if elapsed_time < 5.0:
                    active_feedback[i] = True
                else:
                    feedback_timers[i] = 0

        menu_toggle_requested = False
        instrument_swap_requested = False
        piano_top_requested = False
        violin_top_requested = False
        windows_top_requested = False
        scale_cycle_requested = False
        timbre_cycle_requested = False
        exit_requested = False
        toggle_beep_requested = False
        zone_velocities = [0.0] * len(COMMAND_ZONES)

        for i, (x, y, w, h, name) in enumerate(COMMAND_ZONES):
            hand_in_zone = False
            zone_velocity = 0.0
            last_hand_x = None
            if hand_data:
                for hand_landmarks, hand_label, hand_velocity in hand_data:
                    target_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                    cx, cy = int(target_landmark.x * w_frame), int(target_landmark.y * h_frame)

                    if x <= cx < x + w and y <= cy < y + h:
                        hand_in_zone = True
                        zone_velocity = max(zone_velocity, hand_velocity)
                        last_hand_x = cx

            acc_rate, decay_rate, zone_threshold = get_zone_params(name)

            if hand_in_zone:
                zone_accumulators[i] += acc_rate
            else:
                zone_accumulators[i] = max(0, zone_accumulators[i] - decay_rate)

            if zone_accumulators[i] > zone_threshold:
                print(f"指令觸發: {name}")
                feedback_timers[i] = time.time()
                
                current_prev_label, current_next_label = get_nav_labels(current_scale_idx)

                if name == EXIT_LABEL:
                    exit_requested = True
                
                elif name in (SHOW_MENU_LABEL, HIDE_MENU_LABEL):
                    menu_toggle_requested = True
                    toggle_beep_requested = True

                elif name == current_prev_label:
                    current_scale_idx = (current_scale_idx - 1 + len(SCALE_PRESETS)) % len(SCALE_PRESETS)
                    (current_top_names, top_zone_cache, COMMAND_ZONES, 
                     zone_accumulators, feedback_timers, zone_thresholds) = rebuild_top_for_preset(
                        current_scale_idx, menu_visible, COMMAND_ZONES
                    )
                    COMMAND_ZONES = swap_bottom_to_instruments(COMMAND_ZONES, current_scale_idx)
                    COMMAND_ZONES = ensure_toggle_label(COMMAND_ZONES, menu_visible)
                    zones_dirty = True

                elif name == current_next_label:
                    current_scale_idx = (current_scale_idx + 1) % len(SCALE_PRESETS)
                    (current_top_names, top_zone_cache, COMMAND_ZONES, 
                     zone_accumulators, feedback_timers, zone_thresholds) = rebuild_top_for_preset(
                        current_scale_idx, menu_visible, COMMAND_ZONES
                    )
                    COMMAND_ZONES = swap_bottom_to_instruments(COMMAND_ZONES, current_scale_idx)
                    COMMAND_ZONES = ensure_toggle_label(COMMAND_ZONES, menu_visible)
                    zones_dirty = True

                elif name in TOP_ACTION_ALL:
                    hx = last_hand_x if last_hand_x is not None else x + w / 2
                    play_action_sound(name, zone_velocity, hx)
                    print(f"{name} 觸發（播放音效）。")

                zone_accumulators[i] = 0

            zone_velocities[i] = zone_velocity

            if exit_requested or menu_toggle_requested:
                break

        avg_zone_accumulator = (
            sum(zone_accumulators) / len(zone_accumulators)
            if zone_accumulators
            else 0.0
        )

        if exit_requested:
            cap.release()
            cv2.destroyAllWindows()
            return

        motion_low = (
            avg_hand_velocity < RELAX_VELOCITY_THRESHOLD
            and avg_zone_accumulator < RELAX_ACCUM_THRESHOLD
        )

        if motion_low:
            if relax_candidate_start is None:
                relax_candidate_start = time.time()
            elif (
                relax_state == "active"
                and time.time() - relax_candidate_start >= RELAX_TRIGGER_SECONDS
            ):
                relax_state = "relax"
                instrument_fade = schedule_fade(
                    instrument_gain, RELAX_INSTRUMENT_GAIN, GAIN_FADE_SECONDS
                )
                ambient_fade = schedule_fade(
                    ambient_gain, AMBIENT_TARGET_GAIN, GAIN_FADE_SECONDS
                )
                if ambient_warning and not ambient_warning_shown:
                    print(ambient_warning)
                    ambient_warning_shown = True
        else:
            relax_candidate_start = None
            if relax_state == "relax" and avg_hand_velocity >= RELAX_EXIT_VELOCITY:
                relax_state = "active"
                instrument_fade = schedule_fade(
                    instrument_gain, 1.0, GAIN_FADE_SECONDS
                )
                ambient_fade = schedule_fade(ambient_gain, 0.0, GAIN_FADE_SECONDS)

        if toggle_beep_requested:
            play_beep_sound()

        if menu_toggle_requested:
            menu_visible = not menu_visible
            
            if menu_visible:
                btn_w = 220
                btn_h = 100
                margin_bottom = 30
                center_x = (CAP_WIDTH - btn_w) // 2
                btn_y = CAP_HEIGHT - btn_h - margin_bottom
                center_exit_btn = (center_x, btn_y, btn_w, btn_h, EXIT_LABEL)

                prev_label, next_label = get_nav_labels(current_scale_idx)
                nav_left = (*NAV_BTN_LEFT_RECT, prev_label)
                nav_right = (*NAV_BTN_RIGHT_RECT, next_label)

                COMMAND_ZONES = [nav_left, center_exit_btn, nav_right] + top_zone_cache
                
            else:
                COMMAND_ZONES = init_four_buttons(current_scale_idx, menu_visible)
                COMMAND_ZONES += top_zone_cache
            
            zone_accumulators = [0] * len(COMMAND_ZONES)
            feedback_timers = [0] * len(COMMAND_ZONES)
            zone_thresholds = build_zone_thresholds(COMMAND_ZONES)
            continue

        if instrument_swap_requested:
            if not menu_visible:
                menu_visible, COMMAND_ZONES, top_zone_cache = toggle_menu_visibility(
                    menu_visible, COMMAND_ZONES, top_zone_cache, current_top_names
                )
            COMMAND_ZONES = swap_bottom_to_instruments(COMMAND_ZONES, current_scale_idx)
            window_origin = {}
            zone_accumulators = [0] * len(COMMAND_ZONES)
            feedback_timers = [0] * len(COMMAND_ZONES)
            zone_thresholds = build_zone_thresholds(COMMAND_ZONES)
            continue

        if scale_cycle_requested:
            if not menu_visible:
                menu_visible, COMMAND_ZONES, top_zone_cache = toggle_menu_visibility(
                    menu_visible, COMMAND_ZONES, top_zone_cache, current_top_names
                )
            current_scale_idx = (current_scale_idx + 1) % len(SCALE_PRESETS)
            (
                current_top_names,
                top_zone_cache,
                COMMAND_ZONES,
                zone_accumulators,
                feedback_timers,
                zone_thresholds,
            ) = rebuild_top_for_preset(current_scale_idx, menu_visible, COMMAND_ZONES)
            continue

        if zones_dirty:
            zone_thresholds = build_zone_thresholds(COMMAND_ZONES)

        instrument_gain, instrument_fade = step_fade(instrument_gain, instrument_fade)
        ambient_gain, ambient_fade = step_fade(ambient_gain, ambient_fade)
        INSTRUMENT_MASTER_GAIN = instrument_gain

        if relax_state == "relax" and ambient_gain > 0 and ambient_loop_data:
            should_start = ambient_play_obj is None or not ambient_play_obj.is_playing()
            if should_start:
                wave_obj = build_wave_with_gain(
                    ambient_loop_data, max(0.05, ambient_gain)
                )
                if wave_obj:
                    ambient_play_obj = wave_obj.play()
        elif relax_state == "active" and ambient_gain <= 0.05 and ambient_play_obj:
            ambient_play_obj.stop()
            ambient_play_obj = None

        zone_colors = []
        for i, zone in enumerate(COMMAND_ZONES):
            v = zone_velocities[i] if i < len(zone_velocities) else 0.0
            
            name = zone[4]
            
            if name in TOP_ACTION_ALL:
                base = COLOR_TOP_NOTE
            elif name.startswith("<") or name.endswith(">"):
                base = COLOR_NAV_BTN
            else:
                base = COLOR_SYS_BTN
            
            zone_colors.append(velocity_to_color(v, base))

        frame = draw_ui(frame, COMMAND_ZONES, zone_accumulators, zone_thresholds, zone_colors=zone_colors)

        overlay = render_visual_overlay(visual_state)
        frame = blend_overlay_on_zones(
            frame, overlay, COMMAND_ZONES, motion_metrics["alpha"], highlights=active_feedback
        )

        if relax_state == "relax":
            dim_level = VISUAL_DIM_MAX * clamp01(ambient_gain)
            if dim_level > 0:
                frame = cv2.addWeighted(
                    frame, 1 - dim_level, np.zeros_like(frame), dim_level, 0
                )

        state_hint = "活躍模式：揮動手掌演奏與觸發音效"
        if relax_state == "relax":
            state_hint = "放鬆模式：音量降低並播放環境音，移動手掌返回演奏"

        frame = put_chinese_text(
            frame,
            state_hint,
            (50, 60),
            FONT_PATHS,
            24,
            (255, 255, 255)
        )

        if relax_candidate_start and relax_state == "active":
            countdown = max(
                0.0, RELAX_TRIGGER_SECONDS - (time.time() - relax_candidate_start)
            )
            frame = put_chinese_text(
                frame,
                f"放鬆模式倒數 {countdown:0.1f}s (保持低速)",
                (50, 95),
                FONT_PATHS,
                24,
                (200, 220, 255)
            )

        cv2.imshow("Hand Gesture Interface", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    hands.close() # 關閉 MediaPipe Hands 資源
    print("程式已結束。")

if __name__ == '__main__':
    main()
