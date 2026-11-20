"""Subtitles for every audio/video file in a folder, loading the model once.

    python examples/batch_subtitles.py videos/ --out subs/ --language Dutch --diarize

--diarize needs the [diarize] extra and access to pyannote/speaker-diarization-community-1 on Hugging Face.
"""

import argparse
from pathlib import Path

import rumblex
from rumblex.aligner import Aligner
from rumblex.audio import load_audio
from rumblex.writers import WriterOptions

EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".mp4", ".mkv", ".mov", ".webm"}

ap = argparse.ArgumentParser()
ap.add_argument("folder")
ap.add_argument("--out", default="subs")
ap.add_argument("--language", default=None)
ap.add_argument("--diarize", action="store_true")
ap.add_argument("--max_line_width", type=int, default=42)
args = ap.parse_args()

model = rumblex.load_model()
aligner = Aligner.from_transcriber(model)
diarizer = None
if args.diarize:
    from rumblex.diarizer import Diarizer, assign_speakers

    diarizer = Diarizer()
opts = WriterOptions(max_line_width=args.max_line_width)

files = sorted(p for p in Path(args.folder).iterdir() if p.suffix.lower() in EXTS)
for p in files:
    audio = load_audio(p)  # decode once; transcription, alignment and diarization share it
    result = model.transcribe(audio, language=args.language, keep_encoder_states=True)
    aligner.align(result, audio)
    if diarizer is not None:
        assign_speakers(result, diarizer(audio))
    result.metadata.pop("chunk_transcripts", None)  # encoder states: not for output files
    rumblex.write_result(result, p, args.out, ["srt", "vtt"], opts)
    print(f"{p.name}: {len(result.segments)} segments, {len(audio) / 16000:.0f} s")
