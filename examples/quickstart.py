"""Transcribe one file with word timestamps and write subtitles.

python examples/quickstart.py meeting.mp4
"""

import sys

import rumblex

path = sys.argv[1] if len(sys.argv) > 1 else "audio.wav"

model = rumblex.load_model()  # Reson8/Rumble-2-Flash; GPU if available (bf16 on Ampere+, float32 otherwise)
result = rumblex.transcribe(path, model)  # language is detected; pass language="Dutch" to force it

for seg in result.segments:
    print(f"[{seg.start:7.2f} - {seg.end:7.2f}] {seg.text}")
    for w in seg.words[:5]:
        print(f"      {w.start:7.2f} {w.end:7.2f}  {w.word}")

rumblex.write_result(result, path, "out", ["srt", "vtt", "json"])
print("written to out/")
