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


# waveform = input('Input waveform: sine, square, saw: ')

# # Play an A4 note
# audio = make_tone(440, 1.0, waveform)

# sd.play(audio, SAMPLE_RATE)
# sd.wait()



while True:
    waveform = input("Waveform (sine, square, saw, quit): ").lower() or "sine"

    if waveform == "quit":
        break

    # frequency, duration = map(float, input("Frequency (Hz) and Duration (seconds): ").split())
    frequency = float(input("Frequency (Hz): "))
    # duration = float(input("Duration (seconds): "))
    duration = 1.0


    audio = make_tone(frequency, duration, waveform)

    sd.play(audio, SAMPLE_RATE)
    sd.wait()