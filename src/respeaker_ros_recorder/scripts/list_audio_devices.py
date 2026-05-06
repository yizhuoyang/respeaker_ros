#!/usr/bin/env python3
import pyaudio

pa = pyaudio.PyAudio()

print("Available audio input devices:")
for i in range(pa.get_device_count()):
    info = pa.get_device_info_by_index(i)
    max_inputs = int(info.get("maxInputChannels", 0))
    if max_inputs > 0:
        print(
            f"Index {i}: {info['name']} | "
            f"maxInputChannels={max_inputs} | "
            f"defaultSampleRate={info['defaultSampleRate']}"
        )

pa.terminate()
