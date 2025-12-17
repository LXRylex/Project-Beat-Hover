from ursina import *
import random, os, math, wave, audioop, json, hashlib
import time as pytime
from statistics import median
from tkinter import Tk, filedialog
from pydub import AudioSegment

try:
    import librosa
    HAVE_LIBROSA = True
except ImportError:
    HAVE_LIBROSA = False

try:
    from pydub.playback import _play_with_simpleaudio
    HAVE_SIMPLEAUDIO = True
except ImportError:
    HAVE_SIMPLEAUDIO = False


# ----------------- GAME STATE -----------------
_initialized = False
_on_exit_to_menu = None

score = 0
combo = 0
max_combo = 0
combo_scale_time = 0.0

COMBO_BASE_SCALE = 2.5
combo_scale_current = COMBO_BASE_SCALE

spawn_timer = 0.0
game_started = False
game_over = False

CLOSE_Z_LIMIT   = 1.0
NOTE_SPEED_BASE = 10
SPAWN_INTERVAL  = 0.4

BOX_SIZE = 4
FRAME_Z  = 0
SPAWN_Z  = 10
MARGIN   = 0.3

mouse_locked_to_box = False
MOUSE_X_MIN = -0.35
MOUSE_X_MAX =  0.35
MOUSE_Y_MIN = -0.35
MOUSE_Y_MAX =  0.35

use_song_schedule = False
note_schedule = []
schedule_index = 0

audio_time = 0.0
song_start_time = 0.0

# ----------------- AUDIO PATHS -----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

SAVES_DIR = os.path.join(BASE_DIR, "saves")
os.makedirs(SAVES_DIR, exist_ok=True)

PLAY_WAV_NAME = "beat_hover_current.wav"
PLAY_WAV_PATH = os.path.join(ASSETS_DIR, PLAY_WAV_NAME)

current_song_segment = None
current_song_play_obj = None
current_song_audio = None

# map bookkeeping
current_map_path = None
current_wav_sha1 = None

# ----------------- VISUALIZER DATA -----------------
vis_amps = []
vis_chunk_sec = 0.02
vis_duration = 0.0

# ----------------- CROSSHAIR / HIT SETTINGS -----------------
CURSOR_SIZE_UI = 0.03
HIT_BOX_HALF   = 0.09
HIT_Z_MIN      = FRAME_Z - 0.4
HIT_Z_MAX      = CLOSE_Z_LIMIT + 0.8
NEAR_HIT_MARGIN = 1.5

# ----------------- HUD LAYOUT -----------------
HUD_X = -0.86
HUD_Y =  0.47

# ----------------- HEALTH -----------------
health = 100.0
MAX_HEALTH = 100.0
MISS_DAMAGE = 8.0

HIT_HEAL_NORMAL = 1.2
HIT_HEAL_FEVER  = 2.4

# ----------------- SMART SILENCE / DENSITY (SONG MODE) -----------------
SILENCE_AMP_TH = 0.030
QUIET_AMP_TH   = 0.100
LOUD_AMP_TH    = 0.250

QUIET_MIN_INTERVAL = 0.50
LOUD_MIN_INTERVAL  = 0.18

MIN_KEEP_ABS = 60

# ----------------- SPIRAL RULES (classic: spiral blocks normal spawns) -----------------
SPIRAL_NOTES_COUNT     = 8
SPIRAL_BASE_CHANCE     = 0.20
SPIRAL_THRESHOLD_COUNT = 200
SPIRAL_EVERY_N_NOTES   = 10

SPIRAL_CHAIN_CHANCE = 0.30
SPIRAL_CHAIN_MAX_EXTRA = 2
SPIRAL_CHAIN_Z_DELAY = 4.0

total_notes_spawned = 0
spiral_active = False
spiral_chain_left = 0
spiral_chain_side = 1
spiral_chain_speed = NOTE_SPEED_BASE

# ----------------- FEVER -----------------
fever_text = None
fever_active = False
fever_mult = 1.0
FEVER_START = 25
FEVER_MAX_MULT = 2.0

# Fever tunnel (inside box wireframe layers)
ENABLE_TUNNEL_EFFECT = True
fever_hold_time = 0.0
FEVER_TUNNEL_AFTER_SEC = 5.0

tunnel_root = None
tunnel_layers = []
tunnel_alpha = 0.0
tunnel_phase = 0.0

TUNNEL_LAYERS = 18
TUNNEL_MAX_ALPHA = 0.32
TUNNEL_FADE_IN_SPEED = 0.10
TUNNEL_FADE_OUT_SPEED = 0.12
TUNNEL_ROT_SPEED_BASE = 35.0
TUNNEL_ROT_SPEED_ADD  = 85.0

# ----------------- OBJECT REFERENCES -----------------
game_parent = None
notes_parent = None
particles_parent = None
vis_parent = None

box_outline = None

left_bars = []
right_bars = []

score_text = None
combo_parent = None
combo_text = None
combo_label = None
max_combo_text = None
lock_text = None
song_text = None
cursor = None

loading_parent = None
loading_text = None

# Health UI
health_parent = None
health_bg = None
health_fill = None
health_label = None

# Game over overlay
gameover_parent = None
gameover_title = None
gameover_stats = None
gameover_hint = None


# ----------------- MAP SAVE/LOAD -----------------
def _sha1_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def _map_path_for_wav_sha1(wav_sha1: str) -> str:
    return os.path.join(SAVES_DIR, f"{wav_sha1}.json")

def _seed_from_sha1_hex(sha1_hex: str) -> int:
    try:
        return int(sha1_hex[:8], 16)
    except Exception:
        return 12345678

def _save_map(map_path: str, wav_sha1: str, wav_file: str, schedule: list):
    payload = {
        "version": 1,
        "wav_sha1": wav_sha1,
        "wav_file": wav_file,
        "note_schedule": schedule,
    }
    tmp = map_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp, map_path)

def _load_map(map_path: str):
    with open(map_path, "r", encoding="utf-8") as f:
        return json.load(f)

def _validate_loaded_map(data: dict, wav_sha1: str) -> list:
    if not isinstance(data, dict):
        return []
    if data.get("version") != 1:
        return []
    if data.get("wav_sha1") != wav_sha1:
        return []
    sch = data.get("note_schedule")
    if not isinstance(sch, list) or not sch:
        return []
    out = []
    for it in sch:
        if not isinstance(it, dict):
            continue
        if "time" not in it or "x" not in it or "y" not in it:
            continue
        out.append({
            "time": float(it["time"]),
            "x": float(it["x"]),
            "y": float(it["y"]),
            "speed": float(it.get("speed", NOTE_SPEED_BASE)),
        })
    return out


# ----------------- COLOR / UTILS -----------------
def _mix(c1, c2, t):
    return (
        c1[0] + (c2[0] - c1[0]) * t,
        c1[1] + (c2[1] - c1[1]) * t,
        c1[2] + (c2[2] - c1[2]) * t,
    )

def _palette_cycle(t):
    PURPLE = (168/255, 85/255, 247/255)
    BLUE   = (59/255, 130/255, 246/255)
    RED    = (239/255, 68/255, 68/255)
    t = t % 3.0
    if t < 1.0: return _mix(PURPLE, BLUE, t)
    if t < 2.0: return _mix(BLUE, RED, t - 1.0)
    return _mix(RED, PURPLE, t - 2.0)

def _neon_rgb(r, g, b, intensity):
    rr = r * intensity
    gg = g * intensity
    bb = b * intensity
    m = max(rr, gg, bb)
    if m > 1.0:
        rr /= m; gg /= m; bb /= m
    return color.rgb(rr, gg, bb)

def _soft(a, k=3.2):
    return 1.0 - math.exp(-k * a)

def _clamp(v, a, b):
    return max(a, min(b, v))

# ---- SPEEDUP AFTER 50 COMBO (up to +10%) ----
def game_speed_multiplier(note_speed: float) -> float:
    global combo
    if combo <= 50:
        return 1.0
    combo_norm = _clamp((combo - 50) / 50.0, 0.0, 1.0)
    base = 1.0 + 0.10 * combo_norm
    ratio = (note_speed / NOTE_SPEED_BASE) if NOTE_SPEED_BASE else 1.0
    note_norm = _clamp((ratio - 1.0) / 0.7, 0.0, 1.0)
    tweak = (note_norm - 0.5) * 0.02 * combo_norm
    return _clamp(base + tweak, 1.0, 1.10)


# ----------------- AUDIO / ANALYSIS -----------------
def analyze_wav_amplitudes(path, chunk_seconds=0.02):
    wf = wave.open(path, 'rb')
    framerate = wf.getframerate()
    sampwidth = wf.getsampwidth()
    nframes = wf.getnframes()
    exact_duration = (nframes / framerate) if framerate else 0.0

    chunk_size = max(1, int(framerate * chunk_seconds))

    amps = []
    while True:
        frames = wf.readframes(chunk_size)
        if not frames:
            break
        amps.append(audioop.rms(frames, sampwidth))

    wf.close()
    return amps, exact_duration, chunk_seconds

def normalize_amps(amps):
    if not amps:
        return []
    m = max(amps)
    if m <= 0:
        return [0.0] * len(amps)
    return [a / m for a in amps]

def get_amp_at(t: float) -> float:
    global vis_amps, vis_chunk_sec
    if not vis_amps or t < 0:
        return 0.0
    idx = int(t / vis_chunk_sec)
    if idx < 0 or idx >= len(vis_amps):
        return 0.0
    return vis_amps[idx]

def _amp_window(t: float) -> float:
    a0 = get_amp_at(t)
    a1 = get_amp_at(t - vis_chunk_sec)
    a2 = get_amp_at(t + vis_chunk_sec)
    return max(a0, a1, a2)

def smart_filter_times_by_energy(times: list) -> list:
    if not times or not vis_amps:
        return times

    out = []
    last = -999.0

    for t in times:
        a = _amp_window(t)
        if a < SILENCE_AMP_TH:
            continue

        if LOUD_AMP_TH <= QUIET_AMP_TH:
            norm = 1.0
        else:
            norm = _clamp((a - QUIET_AMP_TH) / (LOUD_AMP_TH - QUIET_AMP_TH), 0.0, 1.0)
        norm = norm ** 0.75

        min_iv = lerp(QUIET_MIN_INTERVAL, LOUD_MIN_INTERVAL, norm)

        if t - last >= min_iv:
            out.append(t)
            last = t

    if len(out) < max(MIN_KEEP_ABS, int(len(times) * 0.25)):
        return times

    return out

def frange(start, stop, step):
    t = start
    while t < stop:
        yield t
        t += step

def build_beat_grid(amplitudes, duration, chunk_seconds=0.05):
    if not amplitudes or duration <= 0:
        return []
    max_amp = max(amplitudes)
    if max_amp == 0:
        return []

    amps = [a / max_amp for a in amplitudes]
    avg_amp = sum(amps) / len(amps)
    thresh = max(avg_amp * 1.2, 0.3)

    peak_times = []
    for i in range(1, len(amps) - 1):
        if amps[i] > thresh and amps[i] >= amps[i - 1] and amps[i] >= amps[i + 1]:
            peak_times.append(i * chunk_seconds)

    if len(peak_times) < 2:
        return list(frange(0.5, duration, 0.5))

    intervals = [peak_times[i+1] - peak_times[i] for i in range(len(peak_times)-1)]
    intervals = [iv for iv in intervals if 0.25 <= iv <= 1.0]
    if not intervals:
        return list(frange(0.5, duration, 0.5))

    beat_interval = median(intervals)
    beat_interval = max(0.25, min(1.0, beat_interval))

    t = peak_times[0]
    out = []
    while t < duration:
        out.append(t)
        t += beat_interval
    return out

def onsets_from_amplitudes(amplitudes, chunk_seconds=0.02):
    if not amplitudes:
        return []
    max_amp = max(amplitudes)
    if max_amp == 0:
        return []

    amps = [a / max_amp for a in amplitudes]
    avg_amp = sum(amps) / len(amps)
    thresh = max(avg_amp * 1.0, 0.25)

    onset_times = []
    for i in range(1, len(amps) - 1):
        if amps[i] > thresh and amps[i] >= amps[i-1] and amps[i] >= amps[i+1]:
            onset_times.append(i * chunk_seconds)

    filtered = []
    last_t = -999
    min_spacing = 0.06
    for t in onset_times:
        if t - last_t >= min_spacing:
            filtered.append(t)
            last_t = t
    return filtered

def librosa_onsets(path):
    y, sr = librosa.load(path, sr=None, mono=True)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, backtrack=True, units='frames'
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr).tolist()
    return [t for t in onset_times if t > 0.05]

def _moving_average(arr, win):
    if win <= 1:
        return arr[:]
    out, s, q = [], 0.0, []
    for v in arr:
        q.append(v); s += v
        if len(q) > win:
            s -= q.pop(0)
        out.append(s / len(q))
    return out

def drop_times_from_amplitudes(amplitudes, chunk_seconds=0.02):
    if not amplitudes:
        return []
    amps = normalize_amps(amplitudes)
    n = len(amps)
    if n < 10:
        return []

    win = max(3, int(0.60 / chunk_seconds))
    base = _moving_average(amps, win)

    score_arr = [0.0] * n
    for i in range(1, n):
        jump = max(0.0, amps[i] - amps[i-1])
        above = max(0.0, amps[i] - base[i])
        score_arr[i] = above * 0.8 + jump * 1.4

    m = sum(score_arr) / n
    var = sum((x - m) ** 2 for x in score_arr) / n
    sd = math.sqrt(var)
    thresh = max(0.10, m + 1.2 * sd)

    picks = []
    last_t = -999.0
    min_spacing = 0.28

    for i in range(2, n - 2):
        if score_arr[i] < thresh:
            continue
        if score_arr[i] >= score_arr[i-1] and score_arr[i] >= score_arr[i+1]:
            t = i * chunk_seconds
            if t - last_t >= min_spacing:
                picks.append(t)
                last_t = t

    return picks

def merge_time_lists(lists, merge_window=0.05):
    all_times = []
    for lst in lists:
        if lst:
            all_times.extend(lst)
    if not all_times:
        return []
    all_times.sort()

    merged = [all_times[0]]
    for t in all_times[1:]:
        if t - merged[-1] <= merge_window:
            continue
        merged.append(t)
    return merged

def thin_density(times, min_interval=0.20):
    if not times:
        return []
    times = sorted(times)
    out = [times[0]]
    last = times[0]
    for t in times[1:]:
        if t - last >= min_interval:
            out.append(t)
            last = t
    return out

def _tail_fill(times, duration, min_tail_gap=0.10):
    if not times or duration <= 0:
        return times

    times = sorted(times)

    if len(times) >= 3:
        diffs = [times[i] - times[i-1] for i in range(1, len(times))]
        tail_diffs = diffs[-min(12, len(diffs)):]
        tail_diffs = [d for d in tail_diffs if 0.10 <= d <= 1.50]
        step = median(tail_diffs) if tail_diffs else 0.5
    else:
        step = 0.5

    step = max(0.20, min(1.00, step))

    last = times[-1]
    target_end = max(0.0, duration - min_tail_gap)

    while last + step < target_end:
        last += step
        times.append(last)

    return times

def make_beat_schedule_from_file(path) -> list:
    ext = os.path.splitext(path)[1].lower()
    analysis_path = path
    tmp_analysis_path = None

    if ext != ".wav":
        tmp_analysis_path = os.path.join(BASE_DIR, "_analysis_temp.wav")
        audio = AudioSegment.from_file(path).set_channels(1)
        audio.export(tmp_analysis_path, format="wav")
        analysis_path = tmp_analysis_path

    try:
        amplitudes, duration, chunk_sec = analyze_wav_amplitudes(analysis_path, chunk_seconds=0.02)

        sources = []
        sources.append(drop_times_from_amplitudes(amplitudes, chunk_seconds=chunk_sec))

        if HAVE_LIBROSA:
            try:
                sources.append(librosa_onsets(analysis_path))
            except Exception:
                pass

        sources.append(build_beat_grid(amplitudes, duration, chunk_seconds=chunk_sec))
        sources.append(onsets_from_amplitudes(amplitudes, chunk_seconds=chunk_sec))

        merged_all = merge_time_lists(sources, merge_window=0.05)
        base_times = thin_density(merged_all, min_interval=0.16)
        filled = _tail_fill(base_times, duration, min_tail_gap=0.10)
        return thin_density(filled, min_interval=0.20)

    finally:
        if tmp_analysis_path is not None:
            try:
                os.remove(tmp_analysis_path)
            except:
                pass

def build_note_schedule(time_list, seed_int: int):
    global note_schedule
    note_schedule = []
    if not time_list:
        return

    rng = random.Random(seed_int)
    half = BOX_SIZE / 2
    n = len(time_list)

    for i, t in enumerate(time_list):
        if n == 1:
            local_dt = 0.5
        else:
            if i == 0:
                local_dt = time_list[i+1] - t
            elif i == n - 1:
                local_dt = t - time_list[i-1]
            else:
                local_dt = 0.5 * ((t - time_list[i-1]) + (time_list[i+1] - t))

        local_dt = max(0.05, min(local_dt, 1.0))
        raw_bpm = 60.0 / local_dt
        if raw_bpm > 50.0:
            raw_bpm += 15.0
        local_bpm = max(50.0, min(raw_bpm, 220.0))

        # adaptive speed (slow songs actually slower)
        norm = _clamp((local_bpm - 60.0) / (200.0 - 60.0), 0.0, 1.0)
        speed_mult = lerp(0.75, 1.70, norm)
        speed = NOTE_SPEED_BASE * speed_mult

        x = rng.uniform(-half + MARGIN, half - MARGIN)
        y = rng.uniform(-half + MARGIN, half - MARGIN)
        note_schedule.append({'time': float(t), 'x': float(x), 'y': float(y), 'speed': float(speed)})


# ----------------- PARTICLES / FX -----------------
class Particle(Entity):
    def __init__(self, pos, vel, col, size=0.08, life=0.8):
        super().__init__(parent=particles_parent, model='sphere', position=pos, scale=size, color=col)
        self.velocity = vel
        self.life = life
        self.max_life = life
        self.drag = 0.96

    def update(self):
        if not game_parent.enabled:
            destroy(self); return
        self.life -= time.dt
        self.position += self.velocity * time.dt
        self.velocity *= self.drag
        self.scale *= (1 - 0.5 * time.dt)
        alpha = max(0.0, self.life / self.max_life)
        self.color = color.rgba(self.color.r, self.color.g, self.color.b, alpha)
        if self.life <= 0:
            destroy(self)

def spawn_particles(pos, col, count=12, speed=3.0):
    for _ in range(count):
        angle = random.uniform(0, 2 * math.pi)
        elevation = random.uniform(-0.3, 0.3)
        vx = math.cos(angle) * speed * random.uniform(0.6, 1.0)
        vy = math.sin(angle) * speed * random.uniform(0.6, 1.0)
        vz = elevation * speed
        Particle(pos, Vec3(vx, vy, vz), col, random.uniform(0.06, 0.12), random.uniform(0.5, 0.9))

class PopFX(Entity):
    def __init__(self, pos, col, size=0.45, life=0.22):
        super().__init__(parent=game_parent, model='quad', billboard=True,
                         position=(pos.x, pos.y, pos.z + 0.02), scale=size, color=col)
        self.life = life
        self.max_life = life

    def update(self):
        if not game_parent.enabled:
            destroy(self); return
        self.life -= time.dt
        self.scale *= (1 + 2.8 * time.dt)
        a = max(0.0, self.life / self.max_life)
        self.color = color.rgba(self.color.r, self.color.g, self.color.b, a)
        if self.life <= 0:
            destroy(self)

def fx_hit(pos):
    PopFX(pos, color.lime, size=0.42, life=0.18)
    spawn_particles(pos, color.lime, count=15, speed=4.5)

def fx_miss(pos):
    PopFX(pos, color.red, size=0.50, life=0.24)
    spawn_particles(pos, color.red, count=10, speed=3.0)


# ----------------- NOTE -----------------
NOTE_SPAWN_FADE = 0.28
PINK = color.rgb(1.0, 0.35, 0.75)
REDC = color.rgb(1.0, 0.10, 0.15)
CYAN = color.rgb(0.2, 0.7, 1.0)

NOTE_BASE_SCALE = 0.40
NOTE_MIN_SCALE  = 0.30
NOTE_ALIGN_TIME = 0.70
NOTE_MOVE_RAMP_TIME = 1.0

def _scaled_align_time(speed: float) -> float:
    if speed <= 0:
        return NOTE_ALIGN_TIME
    ratio = NOTE_SPEED_BASE / speed
    t = NOTE_ALIGN_TIME * (ratio ** 0.35)
    return _clamp(t, 0.35, NOTE_ALIGN_TIME)

def _scaled_move_ramp(speed: float) -> float:
    if speed <= 0:
        return NOTE_MOVE_RAMP_TIME
    ratio = NOTE_SPEED_BASE / speed
    t = NOTE_MOVE_RAMP_TIME * (ratio ** 0.30)
    return _clamp(t, 0.35, NOTE_MOVE_RAMP_TIME)

class Note(Entity):
    def __init__(self, speed=None, is_spiral=False, **kwargs):
        pos = kwargs.get('position', (0, 0, SPAWN_Z))
        if isinstance(pos, Vec3):
            tx, ty, tz = pos.x, pos.y, pos.z
        else:
            tx, ty, tz = pos

        kwargs['position'] = (0, 0, tz)

        super().__init__(
            parent=notes_parent,
            model='cube',
            color=color.rgba(CYAN.r, CYAN.g, CYAN.b, 0.0),
            collider=None,
            scale=Vec3(NOTE_MIN_SCALE, NOTE_MIN_SCALE, NOTE_MIN_SCALE),
            **kwargs
        )

        self.speed = speed if speed is not None else NOTE_SPEED_BASE
        self.is_spiral = is_spiral

        self.target_x = tx
        self.target_y = ty
        self.spawn_z = tz
        self.age = 0.0

        self._align_time = _scaled_align_time(self.speed)
        self._ramp_time  = _scaled_move_ramp(self.speed)

        self.outline = Entity(parent=self, model='wireframe_cube', color=color.yellow, scale=1.15, enabled=False)

        self.timer_bar = Entity(
            parent=self,
            model='quad',
            billboard=True,
            position=(0, 0.235, 0),
            scale=(0.90, 0.14, 1),
            color=color.rgba(PINK.r, PINK.g, PINK.b, 0.0)
        )
        self.timer_bar.origin_x = 0

        self.animate_color(color.rgba(CYAN.r, CYAN.g, CYAN.b, 1.0), duration=NOTE_SPAWN_FADE, curve=curve.out_sine)

    def update(self):
        if not game_parent.enabled or game_over:
            return

        self.age += time.dt
        fade_t = _clamp(self.age / NOTE_SPAWN_FADE, 0.0, 1.0)

        if self.age < self._align_time:
            t = self.age / max(0.001, self._align_time)
            t = curve.out_cubic(t)
            self.x = lerp(0.0, self.target_x, t)
            self.y = lerp(0.0, self.target_y, t)
        else:
            self.x = self.target_x
            self.y = self.target_y

        if self.age < self._ramp_time:
            mt = curve.out_sine(self.age / max(0.001, self._ramp_time))
            move_mul = lerp(0.15, 1.0, mt)
        else:
            move_mul = 1.0

        spd_mul = game_speed_multiplier(self.speed)
        self.z -= time.dt * self.speed * spd_mul * move_mul

        self.outline.enabled = (HIT_Z_MAX < self.z <= HIT_Z_MAX + NEAR_HIT_MARGIN)

        denom = max(0.001, (self.spawn_z - HIT_Z_MAX))
        grow = 1.0 - _clamp((self.z - HIT_Z_MAX) / denom, 0.0, 1.0)
        base_s = lerp(NOTE_MIN_SCALE, NOTE_BASE_SCALE, curve.out_sine(grow))
        boost = 1.0 + 0.22 * curve.out_quad(grow)
        s = base_s * boost
        self.scale = Vec3(s, s, s)

        denom2 = max(0.001, (self.spawn_z - FRAME_Z))
        prog = 1.0 - _clamp((self.z - FRAME_Z) / denom2, 0.0, 1.0)

        remain = 1.0 - prog
        self.timer_bar.scale_x = 0.90 * remain

        r = lerp(PINK.r, REDC.r, prog)
        g = lerp(PINK.g, REDC.g, prog)
        b = lerp(PINK.b, REDC.b, prog)

        bar_alpha = 0.25 + 0.75 * fade_t
        grad = color.rgba(r, g, b, bar_alpha)

        self.timer_bar.color = grad
        self.outline.color = color.rgba(r, g, b, 1.0)

        tint_strength = prog * 0.55
        nr = lerp(CYAN.r, REDC.r, tint_strength)
        ng = lerp(CYAN.g, REDC.g, tint_strength)
        nb = lerp(CYAN.b, REDC.b, tint_strength)
        self.color = color.rgba(nr, ng, nb, self.color.a)


# ----------------- VISUALIZER -----------------
VIS_COUNT_PER_SIDE = 8
VIS_BAR_H   = 0.18
VIS_BASE_W  = 0.28
VIS_MAX_ADD = 1.50

VIS_Z = FRAME_Z - 0.75
VIS_X = (BOX_SIZE * 0.5) + 1.05
VIS_PAD_Y = 0.35

def _make_horizontal_side(side: int):
    bars = []
    x = side * VIS_X
    for i in range(VIS_COUNT_PER_SIDE):
        t = i / max(1, (VIS_COUNT_PER_SIDE - 1))
        y = lerp(-BOX_SIZE * 0.5 + VIS_PAD_Y, BOX_SIZE * 0.5 - VIS_PAD_Y, t)
        bar = Entity(parent=vis_parent, model='quad',
                     position=(x, y, VIS_Z),
                     scale=(VIS_BASE_W, VIS_BAR_H, 1),
                     color=color.white)
        bar.origin_x = -0.5 if side == 1 else 0.5
        bar._w = VIS_BASE_W
        bar._i = i
        bars.append(bar)
    return bars

def update_enhanced_visualizer(t_now: float):
    if vis_amps:
        base_amp = get_amp_at(t_now)
    else:
        base_amp = (math.sin(pytime.time() * 2.0) + 1) * 0.5 * 0.35

    base_amp = max(0.0, min(1.0, base_amp))
    base_amp = min(1.0, base_amp * 1.8)
    base_amp = base_amp ** 0.45
    base_amp = _soft(base_amp, 3.4)

    phase = (t_now * 0.85) + (base_amp * 2.5)
    r, g, b = _palette_cycle(phase)
    base_intensity = 0.6 + 2.6 * base_amp

    def _update(bars):
        for bar in bars:
            i = bar._i
            lag = i * 0.02
            a = get_amp_at(t_now - lag) if vis_amps else base_amp
            a = max(0.0, min(1.0, a * 1.7))
            a = a ** 0.35

            target_w = VIS_BASE_W + a * VIS_MAX_ADD
            bar._w = lerp(bar._w, target_w, 0.25)
            bar.scale_x = bar._w

            boost = 0.9 + 1.2 * a
            bar.color = _neon_rgb(r, g, b, base_intensity * boost)

    _update(left_bars)
    _update(right_bars)


# ----------------- FEVER + UI -----------------
def set_hud_visible(v: bool):
    score_text.enabled = v
    combo_parent.enabled = v
    lock_text.enabled = v
    song_text.enabled = v
    health_parent.enabled = v
    fever_text.enabled = v

def update_combo_display():
    global combo_scale_time
    combo_text.text = f"{combo}x"
    max_combo_text.text = f"Best: {max_combo}x"

    if combo >= 50:
        combo_text.color = color.rgb(1.0, 0.2, 0.2)
    elif combo >= 30:
        combo_text.color = color.rgb(1.0, 0.6, 0.0)
    elif combo >= 15:
        combo_text.color = color.rgb(1.0, 1.0, 0.2)
    elif combo >= 5:
        combo_text.color = color.rgb(0.2, 1.0, 0.5)
    else:
        combo_text.color = color.white

    combo_scale_time = 0.15

def show_loading(v: bool, msg="Loading song..."):
    loading_parent.enabled = v
    loading_text.text = msg

def _update_fever_state():
    global fever_active, fever_mult
    if combo >= FEVER_START:
        fever_active = True
        extra = (combo - FEVER_START) * 0.03
        fever_mult = _clamp(1.25 + extra, 1.25, FEVER_MAX_MULT)
        fever_text.text = f"Fever x{fever_mult:0.2f}"
        fever_text.color = color.rgb(1.0, 0.5, 0.2)
    else:
        fever_active = False
        fever_mult = 1.0
        fever_text.text = ""
        fever_text.color = color.white

def _update_frame_glow(t_now: float):
    if not fever_active:
        box_outline.color = color.white
        return
    pulse = (math.sin(t_now * 5.0) + 1.0) * 0.5
    r, g, b = _palette_cycle(t_now * 1.2)
    intensity = 1.0 + 2.0 * pulse
    box_outline.color = _neon_rgb(r, g, b, intensity)

def _update_health_ui():
    h = _clamp(health / MAX_HEALTH, 0.0, 1.0)
    health_fill.scale_x = 0.70 * h

    if h > 0.70:
        health_fill.color = color.rgb(0.2, 1.0, 0.35)
    elif h > 0.35:
        health_fill.color = color.rgb(1.0, 0.65, 0.15)
    else:
        health_fill.color = color.rgb(1.0, 0.15, 0.2)

    health_label.text = f"HP {int(round(health))}"


# ----------------- FEVER TUNNEL (inside square box) -----------------
def _build_box_tunnel():
    global tunnel_root, tunnel_layers
    if tunnel_root is not None or game_parent is None:
        return

    tunnel_root = Entity(parent=game_parent, enabled=False)
    tunnel_layers = []

    z0 = FRAME_Z + 0.12
    z1 = SPAWN_Z - 0.75

    for i in range(TUNNEL_LAYERS):
        t = i / max(1, (TUNNEL_LAYERS - 1))
        z = lerp(z0, z1, t)
        s = lerp(BOX_SIZE * 0.98, BOX_SIZE * 0.24, t)

        layer = Entity(
            parent=tunnel_root,
            model='wireframe_cube',
            position=(0, 0, z),
            scale=(s, s, 0.05),
            color=color.rgba(1, 1, 1, 0),
            double_sided=True
        )
        layer._t = t
        tunnel_layers.append(layer)

def _update_tunnel_inside_box(t_now: float):
    global tunnel_alpha, tunnel_phase, fever_hold_time

    if fever_active:
        fever_hold_time += time.dt
    else:
        fever_hold_time = 0.0

    _build_box_tunnel()
    if tunnel_root is None:
        return

    target = 0.0
    if ENABLE_TUNNEL_EFFECT and fever_active and fever_hold_time >= FEVER_TUNNEL_AFTER_SEC:
        ramp = _clamp((fever_hold_time - FEVER_TUNNEL_AFTER_SEC) / 2.0, 0.0, 1.0)
        target = TUNNEL_MAX_ALPHA * ramp

    if target > tunnel_alpha:
        tunnel_alpha = lerp(tunnel_alpha, target, TUNNEL_FADE_IN_SPEED)
    else:
        tunnel_alpha = lerp(tunnel_alpha, target, TUNNEL_FADE_OUT_SPEED)

    if tunnel_alpha < 0.01:
        tunnel_root.enabled = False
        return

    tunnel_root.enabled = True
    tunnel_phase += time.dt * (TUNNEL_ROT_SPEED_BASE + TUNNEL_ROT_SPEED_ADD * tunnel_alpha)

    for layer in tunnel_layers:
        t = layer._t
        direction = -1 if int(t * 10) % 2 else 1
        layer.rotation_z = tunnel_phase * direction + t * 140

        r, g, b = _palette_cycle((t_now * 0.9) + t * 1.2)
        depth_boost = (1.0 - t)
        a = tunnel_alpha * (0.35 + 0.65 * depth_boost)
        layer.color = color.rgba(r, g, b, a)


# ----------------- AUDIO CONTROL -----------------
def stop_song_audio():
    global current_song_audio, current_song_play_obj
    if current_song_audio is not None:
        current_song_audio.stop()
        current_song_audio = None
    if current_song_play_obj is not None:
        try:
            current_song_play_obj.stop()
        except Exception:
            pass
        current_song_play_obj = None


# ----------------- SPAWNING (classic spiral blocks all normal spawns) -----------------
def clear_notes():
    for n in notes_parent.children[:]:
        destroy(n)
    for p in particles_parent.children[:]:
        destroy(p)

def spawn_spiral_pattern(speed, side=None, extra_z_offset=0.0):
    global spiral_active
    spiral_active = True

    half = BOX_SIZE / 2
    if side not in (-1, 1):
        side = random.choice([-1, 1])

    base_radius  = half - MARGIN * 1.2
    inner_radius = base_radius * 0.45

    span_deg = 160.0
    tilt_deg = random.uniform(-25.0, 25.0)

    angle_start_deg = -span_deg / 2 + tilt_deg
    angle_end_deg   =  span_deg / 2 + tilt_deg

    depth_span = 12.0

    for i in range(SPIRAL_NOTES_COUNT):
        t = i / (SPIRAL_NOTES_COUNT - 1)
        angle_deg = angle_start_deg + t * (angle_end_deg - angle_start_deg)
        angle_rad = math.radians(angle_deg)
        radius    = base_radius - t * (base_radius - inner_radius)

        x = math.cos(angle_rad) * radius
        y = math.sin(angle_rad) * radius
        x *= side

        x = max(-half + MARGIN, min(half - MARGIN, x))
        y = max(-half + MARGIN, min(half - MARGIN, y))

        z = (SPAWN_Z + extra_z_offset) + (1.0 - t) * depth_span
        Note(position=(x, y, z), speed=speed, is_spiral=True)

    return side

def spawn_single_note(speed, base_x=None, base_y=None):
    half = BOX_SIZE / 2
    if base_x is None or base_y is None:
        x = random.uniform(-half + MARGIN, half - MARGIN)
        y = random.uniform(-half + MARGIN, half - MARGIN)
    else:
        x = max(-half + MARGIN, min(half - MARGIN, base_x))
        y = max(-half + MARGIN, min(half - MARGIN, base_y))
    Note(position=(x, y, SPAWN_Z), speed=speed, is_spiral=False)

def spawn_note_event(speed, base_x=None, base_y=None):
    global total_notes_spawned, spiral_active
    global spiral_chain_left, spiral_chain_side, spiral_chain_speed

    # during spiral, NOTHING else spawns
    if spiral_active:
        return

    total_notes_spawned += 1

    if total_notes_spawned < SPIRAL_THRESHOLD_COUNT:
        spawn_single_note(speed, base_x, base_y)
        return

    if total_notes_spawned % SPIRAL_EVERY_N_NOTES == 0:
        if random.random() < SPIRAL_BASE_CHANCE:
            spiral_chain_speed = speed
            spiral_chain_left = 0
            spiral_chain_side = spawn_spiral_pattern(speed, side=None, extra_z_offset=0.0)

            for _ in range(SPIRAL_CHAIN_MAX_EXTRA):
                if random.random() < SPIRAL_CHAIN_CHANCE:
                    spiral_chain_left += 1
                else:
                    break
            return

    spawn_single_note(speed, base_x, base_y)


# ----------------- PUBLIC API -----------------
def init_game(on_exit_to_menu=None):
    global _initialized, _on_exit_to_menu
    global game_parent, notes_parent, particles_parent, vis_parent
    global box_outline, left_bars, right_bars
    global score_text, combo_parent, combo_text, combo_label, max_combo_text, lock_text, song_text, cursor
    global loading_parent, loading_text
    global health_parent, health_bg, health_fill, health_label
    global fever_text
    global tunnel_root, tunnel_layers, tunnel_alpha, tunnel_phase
    global gameover_parent, gameover_title, gameover_stats, gameover_hint

    if _initialized:
        _on_exit_to_menu = on_exit_to_menu
        return

    _on_exit_to_menu = on_exit_to_menu

    game_parent = Entity(enabled=False)
    notes_parent = Entity(parent=game_parent)
    particles_parent = Entity(parent=game_parent)
    vis_parent = Entity(parent=game_parent)

    box_outline = Entity(
        parent=game_parent,
        model='wireframe_cube',
        color=color.white,
        scale=(BOX_SIZE, BOX_SIZE, 0.1),
        position=(0, 0, FRAME_Z)
    )

    Entity(parent=vis_parent, model='quad',
           position=(-VIS_X, 0, VIS_Z + 0.02),
           scale=(0.55, BOX_SIZE - VIS_PAD_Y*2, 1),
           color=color.rgba(0, 0, 0, 90))
    Entity(parent=vis_parent, model='quad',
           position=(VIS_X, 0, VIS_Z + 0.02),
           scale=(0.55, BOX_SIZE - VIS_PAD_Y*2, 1),
           color=color.rgba(0, 0, 0, 90))

    left_bars = _make_horizontal_side(-1)
    right_bars = _make_horizontal_side(+1)

    # HUD (top-left)
    score_text = Text(text="Score: 0", parent=camera.ui,
                      position=(HUD_X, HUD_Y), origin=(-0.5, 0.5), scale=1.0, enabled=False)

    lock_text  = Text(text="Mouse lock: OFF (Space)", parent=camera.ui,
                      position=(HUD_X, HUD_Y - 0.06), origin=(-0.5, 0.5), scale=0.8, enabled=False)

    song_text  = Text(text="Song: none (WAV/MP3)", parent=camera.ui,
                      position=(HUD_X, HUD_Y - 0.11), origin=(-0.5, 0.5), scale=0.7, enabled=False)

    combo_parent = Entity(parent=camera.ui, enabled=False)
    combo_text = Text(text="0x", parent=combo_parent, position=(0, 0.42), origin=(0, 0),
                      scale=COMBO_BASE_SCALE, color=color.white)
    combo_label = Text(text="COMBO", parent=combo_parent, position=(0, 0.36), origin=(0, 0),
                       scale=0.8, color=color.light_gray)
    max_combo_text = Text(text="Best: 0x", parent=combo_parent, position=(0, 0.31), origin=(0, 0),
                          scale=0.7, color=color.gray)

    cursor = Entity(parent=camera.ui, model='quad', color=color.white,
                    scale=(CURSOR_SIZE_UI, CURSOR_SIZE_UI), enabled=False)

    # HP bar (bottom-center)
    health_parent = Entity(parent=camera.ui, enabled=False, position=(0, -0.43, 0))
    health_bg = Entity(parent=health_parent, model='quad',
                       scale=(0.74, 0.045, 1),
                       color=color.rgba(0, 0, 0, 150),
                       z=0.01)
    health_fill = Entity(parent=health_parent, model='quad',
                         position=(-0.37, 0, 0),
                         scale=(0.70, 0.030, 1),
                         color=color.rgb(0.2, 1.0, 0.35))
    health_fill.origin_x = -0.5
    Entity(parent=health_parent, model='quad',
           scale=(0.75, 0.055, 1),
           color=color.rgba(255, 255, 255, 18),
           z=0.02)

    health_label = Text(text="HP 100", parent=health_parent, origin=(0, 0),
                        position=(0, -0.065, 0), scale=0.75, color=color.white)

    fever_text = Text(text="", parent=camera.ui, position=(0, 0.38), origin=(0, 0),
                      scale=0.85, color=color.white, enabled=False)

    # reset tunnel runtime (built on demand)
    tunnel_root = None
    tunnel_layers = []
    tunnel_alpha = 0.0
    tunnel_phase = 0.0

    # Loading overlay
    loading_parent = Entity(parent=camera.ui, enabled=False)
    Entity(parent=loading_parent, model='quad', color=color.black, scale=(2, 2), z=1)
    loading_text = Text(text="Loading song...", parent=loading_parent, origin=(0, 0), scale=1.5)

    # Game over overlay
    gameover_parent = Entity(parent=camera.ui, enabled=False)
    Entity(parent=gameover_parent, model='quad', color=color.rgba(0, 0, 0, 210), scale=(2, 2), z=0.9)
    gameover_title = Text(text="GAME OVER", parent=gameover_parent, origin=(0, 0), y=0.15, scale=2.3, color=color.red)
    gameover_stats = Text(text="", parent=gameover_parent, origin=(0, 0), y=-0.03, scale=1.1, color=color.white)
    gameover_hint  = Text(text="Press R to restart | ESC to menu", parent=gameover_parent, origin=(0, 0), y=-0.20, scale=0.8, color=color.light_gray)

    show_loading(False)
    set_hud_visible(False)

    _initialized = True


def start_game(play_song: bool = False):
    global game_started, spawn_timer, schedule_index, score
    global combo, max_combo, combo_scale_time, combo_scale_current
    global mouse_locked_to_box
    global song_start_time, audio_time
    global current_song_audio, current_song_play_obj
    global game_over
    global health, fever_active, fever_mult, fever_hold_time
    global tunnel_alpha, tunnel_phase
    global total_notes_spawned, spiral_active, spiral_chain_left, spiral_chain_side, spiral_chain_speed

    if not _initialized:
        raise RuntimeError("beat_hover.init_game() must be called first")

    game_started = True
    game_over = False
    gameover_parent.enabled = False

    spawn_timer = 0.0
    schedule_index = 0
    score = 0

    combo = 0
    max_combo = 0
    combo_scale_time = 0.0
    combo_scale_current = COMBO_BASE_SCALE

    health = MAX_HEALTH
    fever_active = False
    fever_mult = 1.0
    fever_hold_time = 0.0

    # tunnel reset
    tunnel_alpha = 0.0
    tunnel_phase = 0.0
    if tunnel_root:
        tunnel_root.enabled = False

    # spiral reset
    total_notes_spawned = 0
    spiral_active = False
    spiral_chain_left = 0
    spiral_chain_side = 1
    spiral_chain_speed = NOTE_SPEED_BASE

    mouse_locked_to_box = False

    score_text.text = "Score: 0"
    update_combo_display()
    lock_text.text = "Mouse lock: OFF (Space)"
    fever_text.text = ""
    _update_health_ui()

    clear_notes()
    game_parent.enabled = True
    set_hud_visible(True)

    cursor.enabled = True
    mouse.visible = False

    audio_time = 0.0
    song_start_time = pytime.time()

    stop_song_audio()
    if play_song:
        if HAVE_SIMPLEAUDIO and current_song_segment is not None:
            try:
                current_song_play_obj = _play_with_simpleaudio(current_song_segment)
            except Exception:
                current_song_play_obj = None

        if current_song_play_obj is None and os.path.exists(PLAY_WAV_PATH):
            current_song_audio = Audio(PLAY_WAV_NAME, autoplay=True, loop=False)


def go_to_menu():
    global game_started, mouse_locked_to_box, game_over
    global fever_hold_time, tunnel_alpha, tunnel_phase
    global spiral_active, spiral_chain_left

    game_started = False
    game_over = False
    mouse_locked_to_box = False

    fever_hold_time = 0.0
    tunnel_alpha = 0.0
    tunnel_phase = 0.0
    if tunnel_root:
        tunnel_root.enabled = False

    spiral_active = False
    spiral_chain_left = 0

    clear_notes()
    game_parent.enabled = False
    set_hud_visible(False)
    gameover_parent.enabled = False

    show_loading(False)
    stop_song_audio()

    cursor.enabled = False
    mouse.visible = True

    if callable(_on_exit_to_menu):
        _on_exit_to_menu()


def start_random():
    global use_song_schedule, vis_amps, current_map_path, current_wav_sha1
    use_song_schedule = False
    vis_amps = []
    current_map_path = None
    current_wav_sha1 = None
    start_game(False)


def load_song_dialog():
    global use_song_schedule, current_song_segment
    global vis_amps, vis_chunk_sec, vis_duration
    global note_schedule, current_map_path, current_wav_sha1

    root = Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        filetypes=[("Audio files", "*.wav *.mp3"), ("All files", "*.*")]
    )
    root.destroy()

    if not path:
        go_to_menu()
        return

    show_loading(True, "Loading + exporting...")

    try:
        audio = AudioSegment.from_file(path)
        audio = audio.set_channels(2).set_frame_rate(44100)
        audio.export(PLAY_WAV_PATH, format="wav")
        current_song_segment = audio
    except Exception:
        show_loading(False)
        song_text.text = "Song: export error"
        go_to_menu()
        return

    # compute sha1 for this exported wav and map filename
    try:
        current_wav_sha1 = _sha1_file(PLAY_WAV_PATH)
        current_map_path = _map_path_for_wav_sha1(current_wav_sha1)
    except Exception:
        current_wav_sha1 = None
        current_map_path = None

    # visualizer analysis (optional, map can load even if this fails)
    show_loading(True, "Analyzing song...")
    try:
        amps_raw, dur, csec = analyze_wav_amplitudes(PLAY_WAV_PATH, chunk_seconds=0.02)
        vis_amps = normalize_amps(amps_raw)
        vis_chunk_sec = csec
        vis_duration = dur
    except Exception:
        vis_amps = []
        vis_chunk_sec = 0.02
        vis_duration = 0.0

    # load map if exists, else generate and save
    show_loading(True, "Loading map...")
    try:
        if current_map_path and current_wav_sha1 and os.path.exists(current_map_path):
            data = _load_map(current_map_path)
            loaded = _validate_loaded_map(data, current_wav_sha1)
            if loaded:
                note_schedule = loaded
                use_song_schedule = True
            else:
                raise RuntimeError("bad map file")
        else:
            raise FileNotFoundError("no map")
    except Exception:
        show_loading(True, "Generating map...")
        try:
            note_times = make_beat_schedule_from_file(PLAY_WAV_PATH)
            if not note_times:
                raise RuntimeError("no timing points")

            note_times = smart_filter_times_by_energy(note_times)

            seed_int = _seed_from_sha1_hex(current_wav_sha1 or "0" * 40)
            build_note_schedule(note_times, seed_int=seed_int)

            if current_map_path and current_wav_sha1:
                _save_map(
                    current_map_path,
                    wav_sha1=current_wav_sha1,
                    wav_file=os.path.basename(PLAY_WAV_PATH),
                    schedule=note_schedule
                )

            use_song_schedule = True
        except Exception:
            use_song_schedule = False
            note_schedule = []
            show_loading(False)
            song_text.text = "Song: timing/map error"
            go_to_menu()
            return

    song_text.text = f"Song: {len(note_schedule)} notes"
    show_loading(False)
    start_game(play_song=True)


def _restart_current():
    if use_song_schedule and os.path.exists(PLAY_WAV_PATH):
        start_game(play_song=True)
    else:
        start_game(play_song=False)


def game_input(key):
    global mouse_locked_to_box

    if not game_started:
        return

    if game_over:
        if key == 'r':
            _restart_current()
        if key == 'escape':
            go_to_menu()
        return

    if key == 'space':
        mouse_locked_to_box = not mouse_locked_to_box
        lock_text.text = f"Mouse lock: {'ON' if mouse_locked_to_box else 'OFF'} (Space)"

    if key == 'r':
        _restart_current()

    if key == 'escape':
        go_to_menu()


def _set_game_over():
    global game_over
    game_over = True
    stop_song_audio()
    gameover_parent.enabled = True

    gameover_stats.text = (
        f"Score: {score}\n"
        f"Best Combo: {max_combo}x\n"
        f"HP Left: {int(round(health))}"
    )


def game_update():
    global spawn_timer, score, audio_time, schedule_index
    global combo, max_combo, combo_scale_time, combo_scale_current
    global health
    global spiral_active, spiral_chain_left, spiral_chain_side, spiral_chain_speed

    if not game_started or game_parent is None:
        return

    if game_over:
        s = 2.3 + 0.08 * math.sin(pytime.time() * 3.0)
        gameover_title.scale = Vec3(s, s, s)
        return

    cursor.position = mouse.position

    if mouse_locked_to_box:
        mouse.x = max(MOUSE_X_MIN, min(MOUSE_X_MAX, mouse.x))
        mouse.y = max(MOUSE_Y_MIN, min(MOUSE_Y_MAX, mouse.y))

    if use_song_schedule:
        audio_time = max(0.0, pytime.time() - song_start_time)

    t_now = audio_time if use_song_schedule else pytime.time()
    update_enhanced_visualizer(t_now)
    _update_fever_state()
    _update_frame_glow(t_now)
    _update_tunnel_inside_box(t_now)

    # combo pop
    if combo_scale_time > 0:
        combo_scale_time -= time.dt
        pop = (combo_scale_time / 0.15) * 0.3
        target_scale = COMBO_BASE_SCALE * (1.0 + pop)
    else:
        target_scale = COMBO_BASE_SCALE

    combo_scale_current = lerp(combo_scale_current, target_scale, 0.25)
    combo_text.scale = combo_scale_current

    # ----------------- spawning (classic spiral blocks all normal spawns) -----------------
    if use_song_schedule and note_schedule:
        if spiral_active:
            # eat schedule while spiral is running so nothing queues up
            while schedule_index < len(note_schedule) and note_schedule[schedule_index]['time'] <= audio_time:
                schedule_index += 1
        else:
            while schedule_index < len(note_schedule) and audio_time >= note_schedule[schedule_index]['time']:
                data = note_schedule[schedule_index]
                spawn_note_event(
                    speed=data.get('speed', NOTE_SPEED_BASE),
                    base_x=data.get('x', 0.0),
                    base_y=data.get('y', 0.0),
                )
                schedule_index += 1
    else:
        if spiral_active:
            spawn_timer = 0.0
        else:
            spawn_timer += time.dt
            if spawn_timer >= SPAWN_INTERVAL:
                spawn_note_event(NOTE_SPEED_BASE)
                spawn_timer = 0.0

    # ----------------- hit detect (pick best note under cursor) -----------------
    cursor_pos = cursor.position
    best_note = None
    best_score = 999999.0

    for note in notes_parent.children[:]:
        if HIT_Z_MIN <= note.z <= HIT_Z_MAX:
            sp = note.screen_position
            dx = sp.x - cursor_pos.x
            dy = sp.y - cursor_pos.y
            if abs(dx) <= HIT_BOX_HALF and abs(dy) <= HIT_BOX_HALF:
                dist2 = dx*dx + dy*dy
                z_bias = (note.z - HIT_Z_MIN) / max(0.001, (HIT_Z_MAX - HIT_Z_MIN))
                scorev = dist2 + z_bias * 0.002
                if scorev < best_score:
                    best_score = scorev
                    best_note = note

    if best_note:
        pos = best_note.position
        destroy(best_note)
        fx_hit(pos)

        add = int(round(1 * fever_mult))
        score += max(1, add)

        combo += 1
        if combo > max_combo:
            max_combo = combo

        heal_amt = HIT_HEAL_FEVER if fever_active else HIT_HEAL_NORMAL
        health = _clamp(health + heal_amt, 0.0, MAX_HEALTH)
        _update_health_ui()

        score_text.text = f"Score: {score}"
        update_combo_display()

    # miss (any note crosses frame)
    for note in notes_parent.children[:]:
        if note.z <= FRAME_Z:
            pos = note.position
            destroy(note)
            fx_miss(pos)

            combo = 0
            update_combo_display()

            health = _clamp(health - MISS_DAMAGE, 0.0, MAX_HEALTH)
            _update_health_ui()

            if health <= 0.0:
                _set_game_over()
                return

    # ----------------- spiral end -> chain followups -----------------
    if spiral_active:
        if not any(getattr(n, 'is_spiral', False) for n in notes_parent.children):
            if spiral_chain_left > 0:
                spiral_chain_left -= 1
                spiral_chain_side = -spiral_chain_side
                spawn_spiral_pattern(
                    spiral_chain_speed,
                    side=spiral_chain_side,
                    extra_z_offset=SPIRAL_CHAIN_Z_DELAY
                )
                spiral_active = True
            else:
                spiral_active = False


# ----------------- Optional standalone runner -----------------
if __name__ == "__main__":
    app = Ursina()
    window.title = "Beat Hover (Game Only)"
    window.color = color.black
    camera.position = (0, 0, -15)
    camera.look_at((0, 0, 0))

    init_game(on_exit_to_menu=lambda: application.quit())
    start_random()

    def input(key):
        game_input(key)

    def update():
        game_update()

    app.run()
