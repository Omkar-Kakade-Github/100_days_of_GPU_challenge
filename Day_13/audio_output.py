import soundfile as sf
import numpy as np

filtered_audio = np.fromfile("audio_input.bin", dtype=np.float32)
sf.write("filtered_output.wav", filtered_audio, samplerate=16000)
