import numpy as np
import sounddevice as sd
from scipy.signal import butter, sosfilt


SAMPLE_RATE = 44100

attack=0.5
decay=0.3
sustain=0.8
release=0.5
duration = 4
gain = 0.2
ADSR = [0.5, 0.3, 0.8, 0.5]
waveform = "sine"
notes = ["C4", "E4", "G4"]


def make_tone(freq=440, duration=1.0, waveform="sine"):
    # Get envolope of amplitudes
    envelope = make_adsr(duration, attack, decay, sustain, release)
    # Time -> Freuqnecy pack -> phase
    t = np.arange(len(envelope)) / SAMPLE_RATE
    frequency = resolve_time_series(freq, t)
    if np.any(frequency <= 0):
        raise ValueError("Oscillator frequency must be positive")
    phase = 2 * np.pi * (np.cumsum(frequency) - frequency) / SAMPLE_RATE

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

    return gain * signal * envelope


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
def mix_tones(notes, duration=1.0, waveform="sine", pitch_effect=None):

    if not notes:
        raise ValueError("Chord must contain at least one note")

    tones = [
        make_tone(modulate_frequency(note_to_frequency(note), pitch_effect), duration, waveform)
        for note in notes
    ]

    return np.mean(tones, axis=0)

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





while True:
    

    new_waveform = input("Waveform (sine, square, saw, triangle, pulse, quit): ").lower() or waveform
    if new_waveform:
        waveform = new_waveform
    if waveform == "quit":
        break

    # Use a full cord instead of a single note
    new_notes = input("Notes (e.g. C4 E4 G4): ").upper().split() or notes
    if new_notes:
        notes = new_notes
    pitch_effect = input("Do you want pitch modulation? [vibrato(rate, depth) glide(end freq, duration)] (e.g. vibrato 5 10): ").lower().split() or []

    new_gain = float(input("Do you wish to change the gain? ") or gain)
    if new_gain:
        gain = new_gain
    new_duration = float(input("Do you wish to change the duration? ") or duration)
    if new_duration:
        duration = new_duration
    
    
    
    new_adsr = input("ADSR (A D S R), Enter to keep current: ").split()
    if new_adsr:
        ADSR = list(map(float, new_adsr))
    attack, decay, sustain, release = ADSR

    audio = mix_tones(notes, duration, waveform, pitch_effect=pitch_effect)
    # filter_choice = input("Do you want to apply a filter? [lowpass, highpass, bandpass, none]: ").lower() or "none"


    sd.play(audio, SAMPLE_RATE)
    sd.wait()