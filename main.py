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
TOP_ACTION_NAMES_CHINESE = ["中國 宮", "中國 商", "中國 角", "中國 徵", "中國 羽"]
TOP_ACTION_NAMES_RITSU = ["律 一越", "律 斷金", "律 平調", "律 勝絹", "律 神仙"]
TOP_ACTION_NAMES_RYO = ["呂 黃鐘", "呂 太食", "呂 夾鐘", "呂 仲呂", "呂 無射"]
TOP_ACTION_ALL = (
    TOP_ACTION_NAMES_DEFAULT
    + TOP_ACTION_NAMES_PIANO
    + TOP_ACTION_NAMES_VIOLIN
    + TOP_ACTION_NAMES_WINDOWS
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
    "Windows do": 262,
    "Windows re": 294,
    "Windows mi": 330,
    "Windows fa": 349,
    "Windows so": 392,
}
SCALE_PRESETS = [
    {
        "key": "default",
        "label": "預設嗶聲",
        "names": TOP_ACTION_NAMES_DEFAULT,
        "note_keys": ["do", "re", "mi", "fa", "so"],
        "sound_dir": None,
        "use_beep": True,
    },
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
        "note_keys": ["do", "re", "mi", "fa", "so"],
        "sound_dir": VIOLIN_SOUND_DIR,
        "use_beep": False,
    },
    {
        "key": "windows",
        "label": "Windows 嗶聲",
        "names": TOP_ACTION_NAMES_WINDOWS,
        "note_keys": ["do", "re", "mi", "fa", "so"],
        "sound_dir": None,
        "use_beep": True,
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
SCALE_CYCLE_LABEL = "切換音階"
TIMBRE_MODE_LABEL = "音色模式"
ACTION_NOTE_KEYS = {}
TOP_ACTION_SOUNDS = {}
TOP_ZONE_WIDTH = 200
TOP_ZONE_HEIGHT = 100
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
RELAX_TRIGGER_SECONDS = 5.0
GAIN_FADE_SECONDS = 2.5
VISUAL_DIM_MAX = 0.45
RELAX_INSTRUMENT_GAIN = 0.35
INSTRUMENT_MASTER_GAIN = 1.0
BEEP_SOUND_PATH = os.path.join("assets", "beep.wav")
BEEP_WAVE = None
IS_WINDOWS = platform.system() == "Windows"


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
    (50, 570, 200, 100, EXIT_LABEL),
    (300, 570, 200, 100, SCALE_CYCLE_LABEL),
    (550, 570, 200, 100, TIMBRE_MODE_LABEL),
    (1030, 570, 200, 100, SHOW_MENU_LABEL),
]
COMMAND_ZONES = ensure_toggle_label(BASE_ZONES.copy(), menu_visible=False)
register_scale_sounds()


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
        if highlights and highlights[idx]:
            zone_alpha = clamp01(base_alpha * 1.3)

        if zone_alpha <= 0:
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


def convert_mp3_to_wav(mp3_path: str):
    """Transcode an MP3 file to a temporary WAV so it can be loaded."""
    fd, temp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        mp3_path,
        "-ar",
        "44100",
        "-ac",
        "2",
        temp_path,
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return temp_path
    except FileNotFoundError:
        print("找不到 ffmpeg，請先安裝或手動將 MP3 轉成 WAV 放在相同路徑。")
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors="ignore") if e.stderr else str(e)
        print(f"轉換 MP3 失敗：{stderr.strip()}")

    if os.path.exists(temp_path):
        os.remove(temp_path)
    return None


def load_wave_data(path: str):
    """Load raw wave data for blending.

    Supports WAV directly and will attempt to transcode MP3 files via ffmpeg
    when available so users can drop in downloaded ambient tracks.
    """
    if not os.path.exists(path):
        print(f"找不到音檔：{path}")
        return None

    temp_wav_path = None
    try:
        source_path = path
        if path.lower().endswith(".mp3"):
            temp_wav_path = convert_mp3_to_wav(path)
            if not temp_wav_path:
                return None
            source_path = temp_wav_path

        with wave.open(source_path, "rb") as wf:
            sample_rate = wf.getframerate()
            num_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())
    except Exception as e:
        print(f"讀取音檔失敗：{e}")
        return None
    finally:
        if temp_wav_path and os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)

    if sample_width != 2:
        print(f"不支援的位寬：{sample_width}，預期 16-bit wav")
        return None

    audio_data = np.frombuffer(frames, dtype=np.int16)
    return sample_rate, num_channels, sample_width, audio_data


def load_scaled_wave(path: str, velocity: float):
    """Load a wave file and return a WaveObject scaled by velocity."""
    wave_data = load_wave_data(path)
    if not wave_data:
        return None

    sample_rate, num_channels, sample_width, audio_data = wave_data
    velocity_factor = compute_velocity_factor(velocity)
    gain = (0.4 + 0.6 * velocity_factor) * INSTRUMENT_MASTER_GAIN
    pitch_factor = 1.0 + 0.25 * velocity_factor

    scaled = np.clip(audio_data * gain, -32768, 32767).astype(np.int16)
    adjusted_sample_rate = int(sample_rate * pitch_factor)

    return sa.WaveObject(scaled.tobytes(), num_channels, sample_width, adjusted_sample_rate)


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
            return
        except Exception as e:
            print(f"載入提示音失敗：{e}")

    try:
        BEEP_WAVE = _generate_beep_wave()
    except Exception as e:
        print(f"生成內建提示音失敗：{e}")
        BEEP_WAVE = None


def play_beep_sound(freq: int = 880, duration_ms: int = 200):
    """Play a system beep when possible, otherwise use the shared fallback tone."""
    if IS_WINDOWS and winsound:
        try:
            winsound.Beep(freq, duration_ms)
            return
        except RuntimeError:
            print("無法播放系統嗶聲，改用內建提示音。")

    if BEEP_WAVE:
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

    pitch_factor = 1.0 + 0.25 * compute_velocity_factor(velocity)
    adjusted_sample_rate = int(sample_rate * pitch_factor)
    return sa.WaveObject(blended.tobytes(), num_channels, sample_width, adjusted_sample_rate)


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


def build_instrument_bottom_zones():
    """Create piano/violin zones at reserved bottom positions."""
    violin_zone = (50, 460, 200, 100, VIOLIN_LABEL)
    piano_zone = (1030, 460, 200, 100, PIANO_LABEL)
    return [violin_zone, piano_zone]


def swap_bottom_to_instruments(zones):
    """Ensure piano/violin selectors are present without removing controls."""
    if any(z[4] == PIANO_LABEL for z in zones) and any(z[4] == VIOLIN_LABEL for z in zones):
        return zones

    kept = [z for z in zones if z[4] not in (PIANO_LABEL, VIOLIN_LABEL)]
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
    global COMMAND_ZONES, INSTRUMENT_MASTER_GAIN
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
    current_scale_idx, current_preset = get_preset_by_key("default")
    current_timbre_idx = 0
    current_top_names = current_preset["names"]
    top_zone_cache = build_top_zones(current_top_names)
    COMMAND_ZONES = ensure_toggle_label(BASE_ZONES.copy(), menu_visible)
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
        cv2.putText(frame, "Press 'e' to edit layout", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, "Press 'q' to quit", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, "Low motion will enter Relax mode (ambient)", (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 200), 2)

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

        # --- 新增：處理視覺回饋狀態，用於動態著色 ---
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
                elif name == SCALE_CYCLE_LABEL:
                    scale_cycle_requested = True
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

        zone_colors = [velocity_to_color(v) for v in zone_velocities]
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

        cv2.putText(
            frame,
            state_hint,
            (50, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        if relax_candidate_start and relax_state == "active":
            countdown = max(
                0.0, RELAX_TRIGGER_SECONDS - (time.time() - relax_candidate_start)
            )
            cv2.putText(
                frame,
                f"放鬆模式倒數 {countdown:0.1f}s (保持低速)",
                (50, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 220, 255),
                2,
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
