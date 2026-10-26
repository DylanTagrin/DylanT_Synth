import numpy as np
import sounddevice as sd

SAMPLE_RATE = 44100

def make_tone(freq=440, duration=1.0, waveform="sine"):
    t = np.arange(int(SAMPLE_RATE * duration)) / SAMPLE_RATE
    phase = 2 * np.pi * freq * t

    if waveform == "sine":
        signal = np.sin(phase)

    elif waveform == "square":
        signal = np.where(np.sin(phase) >= 0, 1.0, -1.0)

    elif waveform == "saw":
        signal = 2 * ((freq * t) % 1.0) - 1.0

    else:
        raise ValueError("Unknown waveform")

    # Short fade to reduce audible clicks
    fade = min(int(0.01 * SAMPLE_RATE), len(signal) // 2)
    envelope = np.ones(len(signal))
    envelope[:fade] = np.linspace(0, 1, fade)
    envelope[-fade:] = np.linspace(1, 0, fade)

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

def make_chord(notes, duration=1.0, waveform="sine"):
    chord = np.zeros(int(SAMPLE_RATE * duration))

    for note in notes:
        frequency = note_to_frequency(note)
        chord += make_tone(frequency, duration, waveform)

    chord /= len(notes)

    return chord




while True:
    waveform = input("Waveform (sine, square, saw, quit): ").lower() or "sine"

    if waveform == "quit":
        break

    # frequency, duration = map(float, input("Frequency (Hz) and Duration (seconds): ").split())
    # frequency = float(input("Frequency (Hz): "))
    # duration = float(input("Duration (seconds): "))
    duration = 1.0

    # Get frequency from note name
    # note = input("Note (e.g., C4, F#3, A4): ")
    # frequency = note_to_frequency(note)
    # audio = make_tone(frequency, duration, waveform)

    # Use a full cord instead of a single note
    notes = input("Notes (e.g. C4 E4 G4): ").upper().split()
    audio = make_chord(notes, duration, waveform)



    sd.play(audio, SAMPLE_RATE)
    sd.wait()