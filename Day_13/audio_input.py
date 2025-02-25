import numpy as np
import librosa

audio, sr = librosa.load("audio_3.wav", sr=16000)
audio = np.ascontiguousarray(audio, dtype=np.float32)

audio.tofile("audio_input.bin")
