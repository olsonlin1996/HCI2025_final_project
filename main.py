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
import mediapipe as mp # 新增 MediaPipe 導入
import simpleaudio as sa

try:
    import winsound
except ImportError:  # 非 Windows 環境無 winsound
    winsound = None

# MediaPipe 手部偵測設定
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

FONT_PATHS = [
    # macOS/Linux 常用路徑或內建字體名稱（macOS 路徑優先）
    "/System/Library/Fonts/PingFang.ttc",             # macOS 內建蘋方體
    "/System/Library/Fonts/STHeiti Medium.ttc",       # macOS 內建黑體
    "/System/Library/Fonts/Supplemental/PingFang.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",  # macOS 內建宋體
    "Arial Unicode MS.ttf",                           # 許多系統都有的通用字體名稱
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",   # Linux (Ubuntu) 常用中文字體
    # Windows 路徑
    "C:/Windows/Fonts/msjh.ttf",                      # Windows 微軟正黑體
    "C:/Windows/Fonts/mingliu.ttc",                   # Windows 細明體
    "msjh.ttf"
]
FONT_SIZE = 30
_FONT_CACHE = {}
_FONT_WARNING_SHOWN = False

# --- 參數設定 ---
CAP_WIDTH = 1280
CAP_HEIGHT = 720
BOX_COLOR = (182, 175, 164),  # 霧霾藍 (Haze Blue)
COLOR_TOP_NOTE = (166, 149, 158)  # 灰紫 (Grey Purple)
COLOR_NAV_BTN  = (122, 157, 138)  # 鼠尾草綠 (Sage Green)
COLOR_SYS_BTN  = (146, 166, 186)  # 礦石灰 (Mineral Grey)

EXIT_LABEL = "結束程式 (Exit)"
SHOW_MENU_LABEL = "開始"
HIDE_MENU_LABEL = "收起功能"
PIANO_LABEL = "鋼琴"
VIOLIN_LABEL = "小提琴"
WINDOWS_LABEL = "Windows"
TOP_ACTION_NAMES_PIANO = ["鋼琴 do", "鋼琴 re", "鋼琴 mi", "鋼琴 fa", "鋼琴 so"]
TOP_ACTION_NAMES_VIOLIN = ["小提琴 do", "小提琴 re", "小提琴 mi", "小提琴 fa", "小提琴 so"]
TOP_ACTION_NAMES_CHINESE = ["中國 宮", "中國 商", "中國 角", "中國 徵", "中國 羽"]
TOP_ACTION_NAMES_RITSU = ["律 一越", "律 斷金", "律 平調", "律 勝絹", "律 神仙"]
TOP_ACTION_NAMES_RYO = ["呂 黃鐘", "呂 太食", "呂 夾鐘", "呂 仲呂", "呂 無射"]
TOP_ACTION_ALL = (
    TOP_ACTION_NAMES_PIANO
    + TOP_ACTION_NAMES_VIOLIN
    + TOP_ACTION_NAMES_CHINESE
    + TOP_ACTION_NAMES_RITSU
    + TOP_ACTION_NAMES_RYO
)
PIANO_SOUND_DIR = "piano_sound"
VIOLIN_SOUND_DIR = "violin_sound"
HANDPAN_SOUND_DIR = "handpan_sound"
MARIMBA_SOUND_DIR = "marimba_sound"
BELL_SOUND_DIR = "bell_sound"
CHINESE_SOUND_DIR = "chinese_pentatonic_sound"
RITSU_SOUND_DIR = "japanese_ritsu_sound"
RYO_SOUND_DIR = "japanese_ryo_sound"
NOTE_FILENAMES = {
    "do": "c1.wav",
    "re": "d1.wav",
    "mi": "e1.wav",
    "fa": "f1.wav",
    "so": "g1.wav",
    "la": "a1.wav",
    "ti": "b1.wav",
    "do3": "c3.wav",
    "re3": "d3.wav",
    "mi3": "e3.wav",
    "fa3": "f3.wav",
    "so3": "g3.wav",
    "la3": "a3.wav",
    "ti3": "b3.wav",
    "gong": "c1.wav",
    "shang": "d1.wav",
    "jue": "e1.wav",
    "zhi": "g1.wav",
    "yu": "a1.wav",
    "ritsu1": "d1.wav",
    "ritsu2": "e1.wav",
    "ritsu3": "g1.wav",
    "ritsu4": "a1.wav",
    "ritsu5": "b1.wav",
    "ryo1": "c1.wav",
    "ryo2": "d1.wav",
    "ryo3": "f1.wav",
    "ryo4": "g1.wav",
    "ryo5": "a1.wav",
}
TOP_ACTION_FREQS = {
    "do": 262,
    "re": 294,
    "mi": 330,
    "fa": 349,
    "so": 392,
}
SCALE_PRESETS = [
    {
        "key": "piano",
        "label": "鋼琴五聲",
        "names": TOP_ACTION_NAMES_PIANO,
        "note_keys": ["do", "re", "mi", "fa", "so"],
        "sound_dir": PIANO_SOUND_DIR,
        "use_beep": False,
    },
    {
        "key": "violin",
        "label": "小提琴五聲",
        "names": TOP_ACTION_NAMES_VIOLIN,
        "note_keys": ["do3", "re3", "mi3", "fa3", "so3"],
        "sound_dir": VIOLIN_SOUND_DIR,
        "use_beep": False,
    },
    {
        "key": "chinese",
        "label": "中國五聲",
        "names": TOP_ACTION_NAMES_CHINESE,
        "note_keys": ["gong", "shang", "jue", "zhi", "yu"],
        "sound_dir": CHINESE_SOUND_DIR,
        "use_beep": False,
    },
    {
        "key": "ritsu",
        "label": "日本律音階",
        "names": TOP_ACTION_NAMES_RITSU,
        "note_keys": ["ritsu1", "ritsu2", "ritsu3", "ritsu4", "ritsu5"],
        "sound_dir": RITSU_SOUND_DIR,
        "use_beep": False,
    },
    {
        "key": "ryo",
        "label": "日本呂音階",
        "names": TOP_ACTION_NAMES_RYO,
        "note_keys": ["ryo1", "ryo2", "ryo3", "ryo4", "ryo5"],
        "sound_dir": RYO_SOUND_DIR,
        "use_beep": False,
    },
]
TIMBRE_LAYERS = [
    {"key": "handpan", "label": "Handpan", "dir": HANDPAN_SOUND_DIR},
    {"key": "marimba", "label": "Marimba", "dir": MARIMBA_SOUND_DIR},
    {"key": "bell", "label": "Bell", "dir": BELL_SOUND_DIR},
]
TIMBRE_MODES = ["單一音色", "水平漸層"]
TIMBRE_MODE_LABEL = "音色模式"
ACTION_NOTE_KEYS = {}
TOP_ACTION_SOUNDS = {}
TOP_ZONE_WIDTH = 200
TOP_ZONE_HEIGHT = 540
TOP_ZONE_START_Y = 0  # 貼齊上緣
COMMAND_ZONES = []
AMBIENT_SOUND_FILES = [
    ("ocean", os.path.join("ambient_sound", "ocean.mp3")),
    ("wind", os.path.join("ambient_sound", "wind.mp3")),
]
AMBIENT_SOUND_PATH = AMBIENT_SOUND_FILES[0][1]
AMBIENT_TARGET_GAIN = 0.85
RELAX_VELOCITY_THRESHOLD = 40.0
RELAX_EXIT_VELOCITY = 65.0
RELAX_ACCUM_THRESHOLD = 4.0
RELAX_TRIGGER_SECONDS = 10.0
GAIN_FADE_SECONDS = 2.5
VISUAL_DIM_MAX = 0.45
RELAX_INSTRUMENT_GAIN = 0.35
INSTRUMENT_MASTER_GAIN = 1.0
BEEP_SOUND_PATH = os.path.join("assets", "beep.wav")
BEEP_WAVE = None
BEEP_WARNING_SHOWN = False
IS_WINDOWS = platform.system() == "Windows"

def get_nav_labels(current_idx):
    """計算並回傳 (上一個樂器名稱, 下一個樂器名稱)"""
    n = len(SCALE_PRESETS)
    # 使用 % n 確保循環 (0 的前一個變成最後一個)
    prev_idx = (current_idx - 1 + n) % n
    next_idx = (current_idx + 1) % n
    
    # 加箭頭讓視覺更清楚
    prev_label = f"< {SCALE_PRESETS[prev_idx]['label']}"
    next_label = f"{SCALE_PRESETS[next_idx]['label']} >"
    
    return prev_label, next_label

def build_top_zones(top_names=None):
    """
    建立上方五個區塊，左右貼齊邊界並平均分開。
    左側第一個 x=0，右側最後一個 x=CAP_WIDTH - W，中間等距分布。
    """
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


def register_scale_sounds():
    """Populate TOP_ACTION_SOUNDS and ACTION_NOTE_KEYS based on presets."""
    TOP_ACTION_SOUNDS.clear()
    ACTION_NOTE_KEYS.clear()

    for preset in SCALE_PRESETS:
        for display_name, note_key in zip(preset["names"], preset["note_keys"]):
            ACTION_NOTE_KEYS[display_name] = note_key
            if preset["sound_dir"]:
                filename = NOTE_FILENAMES.get(note_key)
                if filename:
                    TOP_ACTION_SOUNDS[display_name] = (preset["sound_dir"], filename)


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
    # (x, y, w, h, name)
    (40, 560, 200, 140, EXIT_LABEL),
    (790, 560, 200, 140, TIMBRE_MODE_LABEL),
    (1040, 560, 200, 140, SHOW_MENU_LABEL),
]
COMMAND_ZONES = ensure_toggle_label(BASE_ZONES.copy(), menu_visible=False)
register_scale_sounds()


def compute_velocity_factor(velocity: float):
    """Normalize velocity into [0, 1] for gain/pitch mapping."""
    # Velocity is in pixels/sec; clamp to avoid extreme scaling.
    normalized = max(0.0, min(velocity / 400.0, 1.0))
    return normalized

def velocity_to_color(velocity: float, base_color):
    """
    將速度映射為顏色，從指定的 base_color 漸變到亮黃色。
    [Fix]: 強制將 numpy 數值轉為 Python int，避免 OpenCV 報錯。
    """
    factor = compute_velocity_factor(velocity)
    
    base_arr = np.array(base_color)
    hot_arr = np.array([255, 255, 0]) 
    
    # 計算漸層顏色
    blended = (base_arr * (1 - factor) + hot_arr * factor).astype(int)
    
    # [關鍵修正]: 這裡必須用 int(c) 強制轉型，不能只用 tuple(blended)
    return tuple(int(c) for c in blended)

def clamp01(value: float):
    return max(0.0, min(value, 1.0))


# --- 動態視覺效果設定 ---
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
    wave = (x_term + y_term + 2.0) * 0.25  # normalize to 0..1 range
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
        
        # --- 修改處：增加 idx < len(highlights) 的安全檢查 ---
        # 這樣就算 highlights 列表比 zones 短，也不會報錯
        if highlights and idx < len(highlights) and highlights[idx]:
            zone_alpha = clamp01(base_alpha * 1.3)

        if zone_alpha <= 0:
            continue
            
        # 確保繪圖不超出畫面邊界
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


# --- 音訊處理核心 (修正版) ---
def ensure_standard_wav(input_path: str):
    """
    利用 ffmpeg 將任何音檔（MP3 或非標準 WAV）轉為標準 44100Hz, 16-bit, 立體聲的臨時 WAV。
    這能解決 'Weird sample rates' 的錯誤。
    """
    fd, temp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    # ffmpeg 強制轉換參數：-ar 44100 (取樣率), -ac 2 (雙聲道), -sample_fmt s16 (16位元)
    cmd = [
        "ffmpeg", "-y", "-v", "error", "-i", input_path,
        "-ar", "44100", "-ac", "2", "-sample_fmt", "s16", "-f", "wav", temp_path
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return temp_path
    except FileNotFoundError:
        print("系統找不到 ffmpeg，請先執行 'brew install ffmpeg' (macOS) 或 'apt install ffmpeg' (Linux)。")
        if os.path.exists(temp_path): os.remove(temp_path)
        return None
    except Exception as e:
        # print(f"音檔標準化失敗 ({input_path}): {e}") # 可取消註解查看詳細錯誤
        if os.path.exists(temp_path): os.remove(temp_path)
        return None

def load_wave_data(path: str):
    """
    讀取音檔數據。
    如果檔案是 MP3 或取樣率不是 44100Hz (導致 simpleaudio 報錯)，會自動進行標準化轉換。
    """
    if not os.path.exists(path):
        print(f"找不到音檔：{path}")
        return None

    temp_wav_path = None
    needs_convert = False

    # 1. 檢查是否需要轉換
    try:
        with wave.open(path, "rb") as wf:
            # simpleaudio 在 macOS 上通常要求 44100Hz 且為 16-bit
            if wf.getframerate() != 44100 or wf.getsampwidth() != 2:
                needs_convert = True
    except:
        # 如果直接用 wave 打不開 (例如是 MP3 或者是損壞的檔頭)，就強制嘗試轉換
        needs_convert = True

    source_path = path

    # 2. 如果需要，執行轉換
    if needs_convert:
        temp_wav_path = ensure_standard_wav(path)
        if not temp_wav_path:
            return None # 轉換失敗
        source_path = temp_wav_path

    # 3. 讀取最終的標準 WAV
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
        # 清理暫存檔
        if temp_wav_path and os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)


def load_scaled_wave(path: str, velocity: float):
    """Load a wave file and return a WaveObject scaled by velocity."""
    wave_data = load_wave_data(path)
    if not wave_data:
        return None

    sample_rate, num_channels, sample_width, audio_data = wave_data
    velocity_factor = compute_velocity_factor(velocity)
    
    # 保留音量變化
    gain = (0.4 + 0.6 * velocity_factor) * INSTRUMENT_MASTER_GAIN
    
    # --- 修改處：移除 Pitch Factor ---
    # pitch_factor = 1.0 + 0.25 * velocity_factor
    # adjusted_sample_rate = int(sample_rate * pitch_factor)

    scaled = np.clip(audio_data * gain, -32768, 32767).astype(np.int16)
    
    # 直接使用原始 sample_rate
    return sa.WaveObject(scaled.tobytes(), num_channels, sample_width, sample_rate)

def build_wave_with_gain(wave_data, gain: float):
    """Scale a pre-loaded wave tuple to the desired gain and return a WaveObject."""
    if not wave_data:
        return None

    sample_rate, num_channels, sample_width, audio_data = wave_data
    scaled = np.clip(audio_data * gain, -32768, 32767).astype(np.int16)
    return sa.WaveObject(scaled.tobytes(), num_channels, sample_width, sample_rate)


def generate_fallback_ambient(duration: float = 2.5, sample_rate: int = 44100):
    """Generate a soft stereo noise bed as a built-in ambient fallback."""
    num_samples = int(duration * sample_rate)
    # Combine two slow sine waves and lightly randomized noise to avoid a harsh tone.
    t = np.linspace(0, duration, num_samples, endpoint=False)
    slow_wave = 0.15 * np.sin(2 * math.pi * 0.35 * t)
    shimmer = 0.08 * np.sin(2 * math.pi * 0.85 * t + math.pi / 3)
    noise = np.random.normal(0, 0.04, num_samples)
    mix = slow_wave + shimmer + noise
    stereo = np.stack([mix, mix * 0.9], axis=-1)
    audio = np.clip(stereo * 32767, -32768, 32767).astype(np.int16)
    return sample_rate, 2, 2, audio


def resolve_ambient_path():
    """Pick the first available ambient file, or return default with a warning."""
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
    """Load ambient file if present; otherwise synthesize an in-memory loop."""
    wave_data = load_wave_data(path)
    if wave_data:
        return wave_data, None

    generated = generate_fallback_ambient()
    warning = f"找不到環境音檔：{path}，改用內建環境音。"
    return generated, warning


def _generate_beep_wave(freq: int = 880, duration_ms: int = 200, sample_rate: int = 44100):
    """Generate a short mono beep tone as a WaveObject."""
    duration = duration_ms / 1000.0
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = 0.35 * np.sin(2 * math.pi * freq * t)
    audio = np.clip(tone * 32767, -32768, 32767).astype(np.int16)
    return sa.WaveObject(audio.tobytes(), 1, 2, sample_rate)


def load_beep_wave():
    """Load or synthesize the shared short beep sound for cross-platform prompts."""
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
    """Play a system beep when possible, otherwise use the shared fallback tone."""
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

def build_blended_wave(note_key: str, velocity: float, hand_x: float):
    """Blend multiple layers according to horizontal position."""
    layer_count = len(TIMBRE_LAYERS)
    if layer_count == 0:
        return None

    normalized_x = normalize_hand_x(hand_x)
    segment = normalized_x * (layer_count - 1)
    base_idx = int(math.floor(segment))
    next_idx = min(base_idx + 1, layer_count - 1)
    mix = segment - base_idx

    base_info = TIMBRE_LAYERS[base_idx]
    next_info = TIMBRE_LAYERS[next_idx]
    base_gain = 1.0 - mix
    next_gain = mix

    def scaled_layer(layer_info, gain):
        filename = NOTE_FILENAMES.get(note_key)
        if not filename:
            return None
        # 確保這裡使用的是修正後的 load_wave_data (包含 ffmpeg 標準化)
        wave_data = load_wave_data(os.path.join(layer_info["dir"], filename))
        if not wave_data:
            return None
        sr, ch, sw, audio = wave_data
        velocity_factor = compute_velocity_factor(velocity)
        gain_scale = (0.3 + 0.7 * velocity_factor) * gain * INSTRUMENT_MASTER_GAIN
        adjusted = np.clip(audio * gain_scale, -32768, 32767).astype(np.int16)
        return sr, ch, sw, adjusted

    base_wave = scaled_layer(base_info, base_gain)
    next_wave = scaled_layer(next_info, next_gain)

    if not base_wave and not next_wave:
        return None
    chosen = base_wave or next_wave
    sample_rate, num_channels, sample_width, audio_data = chosen
    blended = np.zeros_like(audio_data)

    for data in (base_wave, next_wave):
        if not data:
            continue
        _, _, _, audio = data
        if len(audio) < len(blended):
            audio = np.pad(audio, (0, len(blended) - len(audio)))
        blended[: len(audio)] = np.clip(
            blended[: len(audio)] + audio[: len(blended)], -32768, 32767
        )

    # --- 修改處：移除 Pitch Factor ---
    # pitch_factor = 1.0 + 0.25 * compute_velocity_factor(velocity)
    # adjusted_sample_rate = int(sample_rate * pitch_factor)
    
    # 直接使用原始 sample_rate
    return sa.WaveObject(blended.tobytes(), num_channels, sample_width, sample_rate)

def play_action_sound(action_name: str, velocity: float, hand_x: float, timbre_mode: str):
    """Play the mapped tone for the given top action if available."""
    if action_name in TOP_ACTION_FREQS:
        base_freq = TOP_ACTION_FREQS[action_name]
        pitch_factor = 1.0 + 0.25 * compute_velocity_factor(velocity)
        freq = int(base_freq * pitch_factor)
        play_beep_sound(freq, 200)
        return

    sound_info = TOP_ACTION_SOUNDS.get(action_name)
    note_key = ACTION_NOTE_KEYS.get(action_name)
    if timbre_mode == TIMBRE_MODES[1] and note_key:
        wave_obj = build_blended_wave(note_key, velocity, hand_x)
    else:
        wave_obj = None

    if not wave_obj and sound_info:
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


def get_preset_by_key(key: str):
    for idx, preset in enumerate(SCALE_PRESETS):
        if preset["key"] == key:
            return idx, preset
    return 0, SCALE_PRESETS[0]


def rebuild_top_for_preset(
    preset_idx: int,
    menu_visible: bool,
    zones: list,
):
    """Replace the top row with the specified preset and rebuild caches."""
    preset = SCALE_PRESETS[preset_idx]
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

# --- 在 main 計算出的左右按鈕座標，將存放在這裡供全域使用 ---
NAV_BTN_LEFT_RECT = (0, 0, 0, 0)  # (x, y, w, h)
NAV_BTN_RIGHT_RECT = (0, 0, 0, 0) # (x, y, w, h)

def build_instrument_bottom_zones(prev_label, next_label):
    """建立底部導航按鈕，使用 main 計算好的動態座標"""
    # 使用全域變數中的座標
    left_zone = (*NAV_BTN_LEFT_RECT, prev_label)
    right_zone = (*NAV_BTN_RIGHT_RECT, next_label)
    return [left_zone, right_zone]

def swap_bottom_to_instruments(zones, current_idx):
    """
    確保底部顯示：[上一頁] + [BASE_ZONES] + [下一頁]
    並保留原本已存在的上方音階按鈕 (Top Actions)
    """
    # 1. 準備左右導航按鈕
    prev_label, next_label = get_nav_labels(current_idx)
    nav_left = (*NAV_BTN_LEFT_RECT, prev_label)
    nav_right = (*NAV_BTN_RIGHT_RECT, next_label)
    
    # 2. 準備中間的功能按鈕 (直接從全域 BASE_ZONES 拿，不受文字變化影響)
    center_buttons = BASE_ZONES.copy()
    
    # 3. 保留上方音階區塊 (如果存在的話)
    # 過濾出屬於上方音階 (TOP_ACTION_ALL) 的區塊
    top_kept = [z for z in zones if z[4] in TOP_ACTION_ALL]
    
    # 4. 組合所有按鈕：[左] + [中間...] + [右] + [上方...]
    return [nav_left] + center_buttons + [nav_right] + top_kept

GET_READY_SECONDS = 3    # 按下按鍵後的準備時間
CALIBRATION_SECONDS = 3  # 正式校準時間
VAR_THRESHOLD = 75
CAMERA_WAIT_TIMEOUT = 10 # 等待攝影機啟動的最長時間（秒）

# --- 繪圖函式 ---
def draw_ui(frame, zones, accumulators=None, threshold=None, zone_colors=None):
    for i, (x, y, w, h, name) in enumerate(zones):
        # 處理顏色 (防呆)
        raw_color = zone_colors[i] if zone_colors else BOX_COLOR
        if isinstance(raw_color, (list, tuple)) and len(raw_color) > 0 and isinstance(raw_color[0], (list, tuple)):
            raw_color = raw_color[0]
        try: 
            color = tuple(int(c) for c in raw_color)
        except: 
            color = (0, 255, 0)
        
        x, y, w, h = int(x), int(y), int(w), int(h)
        
        # 繪製進度條 (這裡加上了防當機機制)
        if accumulators is not None and threshold is not None:
            # 【關鍵修正】：如果計時器列表比區域少，自動補 0，防止 IndexError
            if i >= len(accumulators):
                accumulators.append(0.0)
            
            t = threshold[i] if isinstance(threshold, list) else threshold
            if t > 0:
                progress = min(accumulators[i] / t, 1.0)
                if progress > 0: 
                    cv2.rectangle(frame, (x, y), (x + int(w * progress), y + h), color, -1)
        
        # 繪製外框
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
        # 繪製文字
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
    
    draw.text(position, text, font=font, fill=(color[2], color[1], color[0])) # OpenCV BGR to PIL RGB
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def init_four_buttons(current_scale_idx, menu_visible):
    """初始化底部按鈕：左右貼邊，中間居中"""
    global NAV_BTN_LEFT_RECT, NAV_BTN_RIGHT_RECT, BASE_ZONES
    
    # 參數設定
    btn_w = 200
    btn_h = 100
    margin_side = 20  # 距離螢幕左右邊界的距離
    margin_bottom = 30
    btn_y = CAP_HEIGHT - btn_h - margin_bottom

    # 1. 左側按鈕 (貼左)
    NAV_BTN_LEFT_RECT = (margin_side, btn_y, btn_w, btn_h)
    
    # 2. 右側按鈕 (貼右)
    NAV_BTN_RIGHT_RECT = (CAP_WIDTH - btn_w - margin_side, btn_y, btn_w, btn_h)

    # 3. 中間的系統按鈕 (結束、開始)，計算置中位置
    gap = 20
    # 兩個按鈕寬度 + 間距
    total_center_width = btn_w * 2 + gap
    center_start_x = (CAP_WIDTH - total_center_width) // 2
    
    BASE_ZONES = [
        (center_start_x, btn_y, btn_w, btn_h, EXIT_LABEL),
        (center_start_x + btn_w + gap, btn_y, btn_w, btn_h, SHOW_MENU_LABEL),
    ]
    
    # 組合：[左] + [中] + [右]
    zones = swap_bottom_to_instruments(BASE_ZONES, current_scale_idx)
    zones = ensure_toggle_label(zones, menu_visible)
    return zones

# --- 攝影機選擇函式 ---
def select_camera():
    print("正在偵測可用的攝影機...")
    available_cameras = []
    # 掃描 0-4
    for i in range(5):
        cap_test = cv2.VideoCapture(i)
        if cap_test.isOpened():
            available_cameras.append(i)
        cap_test.release()
    
    if not available_cameras:
        print("錯誤：找不到任何可用的攝影機。")
        return None
    
    # --- 自動選擇邏輯 ---
    # 策略：優先嘗試列表中的「最後一個」攝影機 (通常外接鏡頭 index 較大)
    suggested_idx = available_cameras[-1]
    print(f"自動嘗試連接攝影機 {suggested_idx} (通常為外接鏡頭)...")

    # 進行快速測試：嘗試開啟並讀取一幀畫面
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

    # --- 手動選擇備案 ---
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
    global COMMAND_ZONES, INSTRUMENT_MASTER_GAIN

    ensure_beep_wave_loaded()
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
    current_scale_idx, current_preset = get_preset_by_key("piano")
    current_timbre_idx = 0
    current_top_names = current_preset["names"]
    top_zone_cache = build_top_zones(current_top_names)

    # 呼叫函式來初始化 COMMAND_ZONES
    COMMAND_ZONES = init_four_buttons(current_scale_idx, menu_visible)
    
    # (接下來的 zone_thresholds 等維持原樣)
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
    
    # 確保 toggle 文字正確 (例如 "開始" vs "收起功能")
    COMMAND_ZONES = ensure_toggle_label(COMMAND_ZONES, menu_visible)

    zone_accumulators = [0] * len(COMMAND_ZONES)
    feedback_timers = [0] * len(COMMAND_ZONES)
    zone_thresholds = build_zone_thresholds(COMMAND_ZONES) # 記得重新建立閾值

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

        # 繪製手部關鍵點
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        # 自動補齊 feedback_timers 長度，防止 IndexError
        if len(feedback_timers) < len(COMMAND_ZONES):
            feedback_timers.extend([0] * (len(COMMAND_ZONES) - len(feedback_timers)))

        # 處理視覺回饋狀態，用於動態著色
        active_feedback = [False] * len(feedback_timers)
        
        for i, start_time in enumerate(feedback_timers):
            if start_time > 0:
                elapsed_time = time.time() - start_time
                if elapsed_time < 5.0:  # 持續 5 秒
                    active_feedback[i] = True
                else:
                    feedback_timers[i] = 0 # 重置計時器

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
            # 檢查是否有手部關鍵點在當前區域內
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
                    # 索引減 1 (循環)
                    current_scale_idx = (current_scale_idx - 1 + len(SCALE_PRESETS)) % len(SCALE_PRESETS)
                    # 重新建構上方音階 與 底部按鈕
                    (current_top_names, top_zone_cache, COMMAND_ZONES, 
                     zone_accumulators, feedback_timers, zone_thresholds) = rebuild_top_for_preset(
                        current_scale_idx, menu_visible, COMMAND_ZONES
                    )
                    # 更新底部按鈕
                    COMMAND_ZONES = swap_bottom_to_instruments(COMMAND_ZONES, current_scale_idx)
                    # 強制檢查並修正按鈕文字 (開始 -> 收起功能)
                    COMMAND_ZONES = ensure_toggle_label(COMMAND_ZONES, menu_visible)
                    zones_dirty = True
                elif name == current_next_label:
                    # 索引加 1 (循環)
                    current_scale_idx = (current_scale_idx + 1) % len(SCALE_PRESETS)
                    # 重新建構上方音階 與 底部按鈕
                    (current_top_names, top_zone_cache, COMMAND_ZONES, 
                     zone_accumulators, feedback_timers, zone_thresholds) = rebuild_top_for_preset(
                        current_scale_idx, menu_visible, COMMAND_ZONES
                    )
                    # 更新底部按鈕
                    COMMAND_ZONES = swap_bottom_to_instruments(COMMAND_ZONES, current_scale_idx)
                    # 強制檢查並修正按鈕文字 (開始 -> 收起功能)
                    COMMAND_ZONES = ensure_toggle_label(COMMAND_ZONES, menu_visible)
                    
                    zones_dirty = True

                elif name == PIANO_LABEL:
                    piano_top_requested = True
                elif name == VIOLIN_LABEL:
                    violin_top_requested = True
                elif name == TIMBRE_MODE_LABEL:
                    timbre_cycle_requested = True
                elif name in TOP_ACTION_ALL:
                    hand_x = last_hand_x if last_hand_x is not None else x + w / 2
                    play_action_sound(
                        name,
                        zone_velocity,
                        hand_x,
                        TIMBRE_MODES[current_timbre_idx],
                    )
                    print(f"{name} 觸發（播放音效）。 速度: {zone_velocity:.1f}")

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
            # 切換狀態
            menu_visible = not menu_visible
            
            if menu_visible:
                # 【模式 A：已開始 (演奏模式)】
                # 目標：底部顯示 [ < 上一個 ]   [ 結束程式 ]   [ 下一個 > ]
                
                # 1. 計算中間「結束按鈕」的置中位置
                btn_w = 220
                btn_h = 100
                margin_bottom = 30
                
                # 讓結束按鈕絕對置中
                center_x = (CAP_WIDTH - btn_w) // 2
                btn_y = CAP_HEIGHT - btn_h - margin_bottom
                center_exit_btn = (center_x, btn_y, btn_w, btn_h, EXIT_LABEL)

                # 2. 準備左右按鈕 (使用全域變數的座標)
                prev_label, next_label = get_nav_labels(current_scale_idx)
                nav_left = (*NAV_BTN_LEFT_RECT, prev_label)
                nav_right = (*NAV_BTN_RIGHT_RECT, next_label)

                # 3. 組合所有按鈕：左 + 中(結束) + 右 + 上方樂器
                COMMAND_ZONES = [nav_left, center_exit_btn, nav_right] + top_zone_cache
                
            else:
                # 【模式 B：已結束 (主選單模式)】
                COMMAND_ZONES = init_four_buttons(current_scale_idx, menu_visible)
                # 加上上方樂器顯示
                COMMAND_ZONES += top_zone_cache
            
            # 4. 重置相關狀態
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

        if timbre_cycle_requested:
            current_timbre_idx = (current_timbre_idx + 1) % len(TIMBRE_MODES)

        if piano_top_requested:
            if not menu_visible:
                menu_visible, COMMAND_ZONES, top_zone_cache = toggle_menu_visibility(
                    menu_visible, COMMAND_ZONES, top_zone_cache, current_top_names
                )
            current_scale_idx, current_preset = get_preset_by_key("piano")
            current_top_names, top_zone_cache, COMMAND_ZONES, zone_accumulators, feedback_timers, zone_thresholds = rebuild_top_for_preset(
                current_scale_idx, menu_visible, COMMAND_ZONES
            )
            continue

        if violin_top_requested:
            if not menu_visible:
                menu_visible, COMMAND_ZONES, top_zone_cache = toggle_menu_visibility(
                    menu_visible, COMMAND_ZONES, top_zone_cache, current_top_names
                )
            current_scale_idx, current_preset = get_preset_by_key("violin")
            current_top_names, top_zone_cache, COMMAND_ZONES, zone_accumulators, feedback_timers, zone_thresholds = rebuild_top_for_preset(
                current_scale_idx, menu_visible, COMMAND_ZONES
            )
            continue

        if windows_top_requested:
            if not menu_visible:
                menu_visible, COMMAND_ZONES, top_zone_cache = toggle_menu_visibility(
                    menu_visible, COMMAND_ZONES, top_zone_cache, current_top_names
                )
            current_scale_idx, current_preset = get_preset_by_key("windows")
            current_top_names, top_zone_cache, COMMAND_ZONES, zone_accumulators, feedback_timers, zone_thresholds = rebuild_top_for_preset(
                current_scale_idx, menu_visible, COMMAND_ZONES
            )
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
            # 1. 安全取得速度：如果 zone_velocities 長度跟不上新的 zones，就預設為 0.0
            v = zone_velocities[i] if i < len(zone_velocities) else 0.0
            
            name = zone[4]
            
            if name in TOP_ACTION_ALL:
                base = COLOR_TOP_NOTE
            elif name.startswith("<") or name.endswith(">"):
                base = COLOR_NAV_BTN
            else:
                base = COLOR_SYS_BTN
            
            zone_colors.append(velocity_to_color(v, base))

        # 繪圖
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
        # cv2.imshow("Foreground Mask", fg_mask) # 移除此行

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    hands.close() # 關閉 MediaPipe Hands 資源
    print("程式已結束。")

if __name__ == '__main__':
    main()