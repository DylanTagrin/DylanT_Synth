import numpy as np
import sounddevice as sd
from scipy.signal import butter, sosfilt
import json
from pathlib import Path

PRESET_FILE = Path(__file__).with_name("presets.json")

def read_presets():
    with open(PRESET_FILE, "r") as file:
        return json.load(file)

def save_presets():
    with open(PRESET_FILE, "w") as file:
        json.dump({"defaults": DEFAULT_SETTINGS, "presets": PRESETS}, file, indent=4)

library = read_presets()
DEFAULT_SETTINGS = library["defaults"]
PRESETS = library["presets"]
settings = DEFAULT_SETTINGS.copy()

SAMPLE_RATE = 44100
notes = ["C4", "E4", "G4"]

# Function that takes an input from config and a frequency in Hz and a duration to make a tone
def make_tone(freq=440, duration=None, config=None):
    config = settings if config is None else config
    duration = config["duration"] if duration is None else duration

    if isinstance(freq, str):
        freq = note_to_frequency(freq)

    envelope = make_adsr(duration, config["attack"], config["decay"], config["sustain"], config["release"])
    t = np.arange(len(envelope)) / SAMPLE_RATE

    frequency = resolve_time_series(freq, t)
    if np.any(frequency <= 0):
        raise ValueError("Oscillator frequency must be positive")

    phase = 2 * np.pi * (np.cumsum(frequency) - frequency) / SAMPLE_RATE

    waveform = config["waveform"]

    # Use the phase with a waveform to get a tone
    if waveform == "sine":
        signal = np.sin(phase)

    elif waveform == "square":
        signal = np.where(np.sin(phase) >= 0, 1.0, -1.0)

    elif waveform == "saw":
        signal = 2 * ((phase / (2 * np.pi)) % 1.0) - 1.0

    elif waveform == "triangle":
        signal = 1 - 4 * np.abs((((phase - np.pi / 2) / (2 * np.pi)) % 1) - 0.5)

    elif waveform == "pulse":
        duty_cycle = 0.3  # 30% high, 70% low
        signal = np.where((phase % (2 * np.pi)) < (duty_cycle * 2 * np.pi), 1.0, -1.0)

    else:
        raise ValueError("Unknown waveform")

    return config["gain"] * signal * envelope


# Function to convert note name to MIDI number
NOTE_OFFSETS = {
    "C": 0,
    "C#": 1,
    "D": 2,
    "D#": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "G": 7,
    "G#": 8,
    "A": 9,
    "A#": 10,
    "B": 11
}

def note_to_frequency(note):
    # Example input: "C4", "F#3", "A4"

    note_name = note[:-1]  # Everything except final character
    octave = int(note[-1]) # Last character

    semitone = NOTE_OFFSETS[note_name.upper()]

    midi = 12 * (octave + 1) + semitone
    # midi number between 0 and 127, where A4 = 69
    frequency = 440 * 2 ** ((midi - 69) / 12)

    return frequency

# Function to create a band of tones (chord) from a list of notes
def mix_tones(notes, duration=None, config=None):
    config = settings if config is None else config
    if not notes:
        raise ValueError("Must provide at least one note")

    tones = [
        make_tone(resolve_effect(config.get("pitch_effect"), note_to_frequency(note)), duration, config)
        for note in notes
    ]

    audio = np.mean(tones, axis=0)

    if config.get("filter") is not None:
        f = config["filter"]
        audio = dynamic_filter(audio, cutoff=resolve_cutoff(f["cutoff"]), filter_type=f["type"], order=f.get("order", 4))

    return audio

# Function to create an ADSR envelope
def make_adsr(duration, attack=0.1, decay=0.2, sustain=0.6, release=0.3):
    if min(duration, attack, decay, release) < 0 or not 0 <= sustain <= 1:
        raise ValueError("Invalid ADSR parameters")

    hold_samples = int(duration * SAMPLE_RATE)
    release_samples = int(release * SAMPLE_RATE)
    t = np.arange(hold_samples) / SAMPLE_RATE

    envelope = np.full(hold_samples, sustain, dtype=float)

    if attack > 0:
        mask = t < attack
        envelope[mask] = t[mask] / attack

    if decay > 0:
        mask = (t >= attack) & (t < attack + decay)
        envelope[mask] = 1 - (1 - sustain) * (t[mask] - attack) / decay

    if attack == 0 and decay == 0:
        envelope[:] = sustain

    # Release begins at the amplitude reached when the key is released
    if release_samples > 0:
        if hold_samples > 0:
            release_level = envelope[-1]
        else:
            release_level = 0

        release_env = np.linspace(release_level, 0, release_samples + 1)[1:]
        envelope = np.concatenate((envelope, release_env))

    return envelope

# Function that takes x(t) and a range of t and returns an array for the output
def resolve_time_series(value, t):
    if callable(value):
        value = value(t)

    values = np.asarray(value, dtype=float)

    if values.ndim == 0:
        values = np.full(t.shape, values.item())

    if values.shape != t.shape:
        raise ValueError("Time series must match time array")

    if not np.all(np.isfinite(values)):
        raise ValueError("Time series contains non-finite values")

    return values

# Frequency generating function for vibrato
def vibrato(base_freq, rate=5, depth=10):
    def frequency(t):
        return base_freq + depth * np.sin(2 * np.pi * rate * t)

    return frequency

# Frequency generating function for glide
def glide(start_freq, end_freq, duration):
    # Detect if end_freq is a note and not a float
    if isinstance(end_freq, str):
        end_freq = note_to_frequency(end_freq)
    
    def frequency(t):
        progress = np.clip(t / duration, 0, 1)
        return start_freq + (end_freq - start_freq) * progress

    return frequency

#Function that takes a freuqnecy and returns a callable function that modulates the frequency from one of the functions
def modulate_frequency(freq, pitch_effect):
    if not pitch_effect:
        return freq

    mod_type = pitch_effect[0]

    if mod_type == "vibrato":
        rate = float(pitch_effect[1])
        depth = float(pitch_effect[2])
        return vibrato(freq, rate, depth)

    elif mod_type == "glide":
        end_freq = (pitch_effect[1])
        duration = float(pitch_effect[2])
        return glide(freq, end_freq, duration)

    else:
        raise ValueError("Unknown modulation type")

# Dynamic Filter that can use different types of filters, variable chunk size and cutoffs
def dynamic_filter(audio, cutoff=1000, filter_type="lowpass", order=4, chunk_size=256):
    t = np.arange(len(audio)) / SAMPLE_RATE

    if filter_type == "bandpass":
        low = resolve_time_series(cutoff[0], t)
        high = resolve_time_series(cutoff[1], t)
        if np.any(low >= high):
            raise ValueError("Lower cutoff must be below upper cutoff")
        cutoffs = np.column_stack((low, high))
    else:
        cutoffs = resolve_time_series(cutoff, t)

    if np.any(cutoffs <= 0) or np.any(cutoffs >= SAMPLE_RATE / 2):
        raise ValueError("Cutoffs must lie between 0 and the Nyquist frequency")

    output = np.zeros_like(audio, dtype=float)
    zi = None

    for start in range(0, len(audio), chunk_size):
        end = min(start + chunk_size, len(audio))
        fc = cutoffs[start]

        sos = butter(order, fc, btype=filter_type, fs=SAMPLE_RATE, output="sos")

        if zi is None:
            zi = np.zeros((len(sos), 2))

        output[start:end], zi = sosfilt(sos, audio[start:end], zi=zi)

    return output

# Functions like modulate_frequency() but this time to extract json words and replace them with a function, works for filters too
def resolve_effect(effect, base=None):
    if effect is None:
        return base

    if not isinstance(effect, dict):
        return effect

    mode = effect["mode"]

    if mode == "glide":
        return glide(effect.get("start", base), effect["end"], effect["duration"])

    elif mode == "vibrato":
        return vibrato(effect.get("base", base), effect["rate"], effect["depth"])

    raise ValueError(f"Unknown effect: {mode}")

# Function to resolve cutoff values, which may be single values or tuples/lists of values, potentially with effects like glide
def resolve_cutoff(cutoff):
    if isinstance(cutoff, (list, tuple)):
        return tuple(resolve_effect(value) for value in cutoff)
    return resolve_effect(cutoff)

# Function to update a setting [name] with a [value]
def update_setting(name, value):
    if name not in DEFAULT_SETTINGS:
        raise KeyError(f"Unknown setting: {name}")

    if name == "waveform":
        if value not in ("sine", "square", "saw", "triangle", "pulse"):
            raise ValueError("Unknown waveform")

    elif name in ("filter", "pitch_effect"):
        if value is not None and not isinstance(value, dict):
            raise ValueError(f"{name} must be a dictionary or None")
        
    else:
        value = float(value)
        if not np.isfinite(value):
            raise ValueError("Setting must be finite")
        if name in ("duration", "attack", "decay", "release", "gain") and value < 0:
            raise ValueError(f"{name} must be nonnegative")
        if name == "sustain" and not 0 <= value <= 1:
            raise ValueError("Sustain must be between 0 and 1")

    settings[name] = value

# Function that calls a preset
def load_preset(name):
    if name not in PRESETS:
        raise KeyError(f"Unknown preset: {name}")

    settings.clear()
    settings.update(DEFAULT_SETTINGS)
    for key, value in PRESETS[name].items():
        update_setting(key, value)

# Use terminal to change setting
def change_settings():
    while True:
        print("\nCurrent settings:")
        for name, value in settings.items():
            print(f"  {name:<12} {value}")

        command = input("\nSetting to change (Enter to finish): ").lower().strip()
        if not command:
            break

        if command not in settings:
            print("Unknown setting")
            continue

        value = input(f"{command} [{settings[command]}]: ").strip()
        if value:
            try:
                update_setting(command, value)
            except ValueError as e:
                print(e)





print("Python Synth — type 'help' for commands")

while True:
    command = input("\nsynth> ").lower().strip()

    if command in ("quit", "exit", "q"):
        break

    elif command in ("help", "h"):
        print("Commands: play, notes, settings, presets, quit")

    elif command in ("play", "p"):
        audio = mix_tones(notes)
        sd.play(audio, SAMPLE_RATE)
        sd.wait()

    elif command == "notes":
        value = input(f"Notes [{' '.join(notes)}]: ").upper().split()
        if value:
            notes = value

    elif command == "settings":
        change_settings()

    elif command == "presets":
        print("Available presets:", ", ".join(PRESETS))
        name = input("Select preset (Enter to cancel): ").lower().strip()
        if name:
            load_preset(name)
            print(f"Loaded preset: {name}")

    else:
        print("Unknown command. Type 'help' for options.")