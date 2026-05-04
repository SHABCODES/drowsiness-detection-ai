import wave
import struct
import math

sample_rate = 44100
duration = 0.5 # 0.5 seconds
frequency = 880.0

with wave.open('alert.wav', 'w') as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(sample_rate)
    for i in range(int(duration * sample_rate)):
        value = int(32767.0 * math.sin(frequency * 2.0 * math.pi * float(i) / float(sample_rate)))
        data = struct.pack('<h', value)
        f.writeframesraw(data)
