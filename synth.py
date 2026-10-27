import numpy as np
import sounddevice as sd

SAMPLE_RATE = 44100

attack=0.5
decay=0.3
sustain=0.8
release=0.5
duration = 4

def make_tone(freq=440, duration=1.0, waveform="sine"):
    # Get envolope of amplitudes
    envelope = make_adsr(duration, attack, decay, sustain, release)
    # Time -> Freuqnecy pack -> phase
    t = np.arange(len(envelope)) / SAMPLE_RATE
    frequency = resolve_frequency(freq, t)
    phase = 2 * np.pi * (np.cumsum(frequency) - frequency) / SAMPLE_RATE

    # Use the phase with a waveform to get a tone
    if waveform == "sine":
        signal = np.sin(phase)

    elif waveform == "square":
        signal = np.where(np.sin(phase) >= 0, 1.0, -1.0)

    elif waveform == "saw":
        signal = 2 * ((phase / (2 * np.pi)) % 1.0) - 1.0

    else:
        raise ValueError("Unknown waveform")

    return 0.2 * signal * envelope


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
def make_adsr(duration, attack=0.1, decay=0.2,
              sustain=0.6, release=0.3):

    # Duration is how long the key is held.
    hold_samples = int(duration * SAMPLE_RATE)

    a = int(attack * SAMPLE_RATE)
    d = int(decay * SAMPLE_RATE)
    r = int(release * SAMPLE_RATE)

    if a + d > hold_samples:
        raise ValueError("Note must be longer than attack + decay")

    # Attack: 0 -> 1
    attack_env = np.linspace(0, 1, a, endpoint=False)

    # Decay: 1 -> sustain
    decay_env = np.linspace(1, sustain, d, endpoint=False)

    # Sustain: constant amplitude
    sustain_env = np.full(hold_samples - a - d, sustain)

    # Release: sustain -> 0
    release_env = np.linspace(sustain, 0, r, endpoint=True)

    return np.concatenate([
        attack_env,
        decay_env,
        sustain_env,
        release_env
    ])

# Function that takes a frequency input (either a constant or a callable function) and returns an array of frequencies for each time step.
def resolve_frequency(freq, t):

    if callable(freq):
        frequency = freq(t)

    else:
        frequency = freq

    frequency = np.asarray(frequency, dtype=float)

    # Constant frequency -> array
    if frequency.ndim == 0:
        frequency = np.full(len(t), frequency.item())

    if frequency.shape != t.shape:
        raise ValueError("Frequency array must match time array")

    if not np.all(np.isfinite(frequency)):
        raise ValueError("Frequency must contain finite values")

    if np.any(frequency <= 0):
        raise ValueError("Frequency must be positive")

    return frequency

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


while True:
    waveform = input("Waveform (sine, square, saw, quit): ").lower() or "sine"
    if waveform == "quit":
        break

    # Use a full cord instead of a single note
    notes = input("Notes (e.g. C4 E4 G4): ").upper().split() or ["C4", "E4", "G4"]
    
    pitch_effect = input("Do you want modulation? [vibrato(rate, depth) glide(end freq, duration)] (e.g. vibrato 5 10): ").lower().split() or []
    audio = mix_tones(notes, duration, waveform, pitch_effect=pitch_effect)
    # audio = make_tone(vibrato(440, rate=5, depth=50), duration, waveform)

    sd.play(audio, SAMPLE_RATE)
    sd.wait()