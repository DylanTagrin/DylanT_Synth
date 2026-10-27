import numpy as np
import sounddevice as sd

SAMPLE_RATE = 44100

attack=0.5
decay=0.5 
sustain=0.8
release=1.0
duration = 1.5

def make_tone(freq=440, duration=1.0, waveform="sine"):

    envelope = make_adsr(duration, attack, decay, sustain, release)

    # Envelope determines how many samples we need
    t = np.arange(len(envelope)) / SAMPLE_RATE
    
    phase = 2 * np.pi * freq * t

    if waveform == "sine":
        signal = np.sin(phase)

    elif waveform == "square":
        signal = np.where(np.sin(phase) >= 0, 1.0, -1.0)

    elif waveform == "saw":
        signal = 2 * ((freq * t) % 1.0) - 1.0

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
def make_chord(notes, duration=1.0, waveform="sine"):

    if not notes:
        raise ValueError("Chord must contain at least one note")

    tones = [
        make_tone(note_to_frequency(note), duration, waveform)
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


while True:
    waveform = input("Waveform (sine, square, saw, quit): ").lower() or "sine"

    if waveform == "quit":
        break

    

    # Use a full cord instead of a single note
    notes = input("Notes (e.g. C4 E4 G4): ").upper().split()
    audio = make_chord(notes, duration, waveform)


    sd.play(audio, SAMPLE_RATE)
    sd.wait()