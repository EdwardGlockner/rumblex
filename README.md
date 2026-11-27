# rumblex

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/edwardgl/rumblex)

Transcription with word timestamps, speaker labels and subtitles, built on
[Reson8/Rumble-2-Flash](https://huggingface.co/Reson8/Rumble-2-Flash).

Rumble-2-Flash has a CTC head next to its decoder. rumblex uses it for word timestamps: the decoder writes
the text and the CTC head aligns it to the audio, reusing the encoder pass that already ran. So alignment
is one linear layer and a Viterbi search, not a second model.

The model is a 0.5B fine-tune of Qwen3-ASR-0.6B. Its card lists 45 languages; I've benchmarked 13 of
them (below). You can try it in the browser with the [demo Space](https://huggingface.co/spaces/edwardgl/rumblex).

![rumblex transcribing two files and writing subtitles](docs/demo.gif)

> Independent project, not affiliated with Reson8.

## Features

- Any input ffmpeg reads. Long files are split into speech chunks with Silero VAD.
- Batched transcription; a chunk's output doesn't depend on what it's batched with. Exposes the model's
  `language`, `prompt`, `digits`, `filler` and `verbatim` options.
- Hotwords: a list of names and terms the decoder is nudged towards (`--hotwords`, `--hotwords_file`).
- Word timestamps from the CTC head. If alignment fails on a chunk, its words get proportional times.
- Two per-word confidences: `probability` from the decoder and `score` from the CTC head. `probability` is
  the better at telling wrong words from right ones ([eval](docs/confidence_eval.md)); it is empty for
  `--text` and for words with proportional times.
- `ctc_match` on each transcribed word: false when the CTC head's own greedy text doesn't contain the
  word. On two English test sets such words were wrong 22% and 31% of the time, against 2% and 4% for
  the rest ([eval](docs/word_trust_eval.md)).
- Alignment of a transcript you already have (`--text`, `rumblex.align`): no decoding, only the encoder
  and the CTC head.
- Optional per-word speaker labels via pyannote.
- JSON, SRT, VTT (optionally with karaoke tags), TSV and TXT.
- An OpenAI-compatible server (`rumblex serve`).

![word timestamps and speaker labels played back over the audio](docs/words.gif)

## Install

```bash
pip install rumblex
pip install "rumblex[diarize]"   # with speaker labels
pip install "rumblex[server]"    # for rumblex serve
```

Needs Python 3.11 or 3.12 and ffmpeg on `PATH`. On Windows, PyPI's torch is CPU-only, so for a GPU install
torch and torchaudio from [pytorch.org](https://pytorch.org/get-started/locally/) first. Ampere and newer
run in bf16, older GPUs in float32 (a 6 GB GTX 1060 works at the default batch size). CPU works but is slow.

From source, `git clone https://github.com/EdwardGlockner/rumblex && cd rumblex && uv sync` (add
`--extra diarize` for speaker labels). The lock file pins CUDA 12.6 wheels, which still support Pascal; for
RTX 50-series install torch yourself and use `pip install -e .`.

## Use

```bash
rumblex meeting.mp4                                   # all formats into the current folder
rumblex talk.wav -f srt vtt --max_line_width 42
rumblex interview.m4a --language Dutch --diarize -f srt txt
rumblex call.wav --hotwords "Stifel, Paul Betz"      # names and terms to listen for
rumblex call.wav --hotwords_file terms.txt           # one per line
rumblex lecture.mp3 --text lecture.txt               # word times for your own transcript
```

```python
import rumblex

model = rumblex.load_model()
result = rumblex.transcribe("meeting.mp4", model)
for seg in result.segments:
    print(seg.start, seg.end, seg.text)
rumblex.write_result(result, "meeting.mp4", "out/", ["srt", "json"])

text = open("lecture.txt", encoding="utf-8").read()
result = rumblex.align("lecture.mp3", text, model)  # no transcribing
```

More in [examples/](examples/).

**Hotwords.** Give rumblex the names and terms you expect, and tokens that would write one of them get a
small bonus during decoding:

```python
result = rumblex.transcribe("call.wav", model, hotwords=["Stifel", "Paul Betz", "SiTime"])
```

On 857 clips of Earnings-21 calls with Rev's 1,013-entry biasing list, the share of spoken list words that
came out right went from 67.9% to 76.9%, with WER going from 5.84% to 6.02%
([docs/hotwords.md](docs/hotwords.md)). Write entries the way they should appear; a list in capitals is
handled. It is meant for a list that belongs to the recording or the customer, not a dictionary.
`--prompt` is still there for free-text context, but don't put a list in it; that made things much worse.
Not available with `--vllm_url` or in `rumblex serve` yet.

**vLLM.** On Linux, decoding can run on a vLLM server (`pip install "rumblex[vllm]"`):

```bash
vllm serve Reson8/Rumble-2-Flash --max-model-len 4096 --gpu-memory-utilization 0.5
rumblex meeting.mp4 --vllm_url http://localhost:8000/v1
```

Alignment still runs the encoder locally, so leave some GPU memory free. On LibriSpeech this went from 467×
to 601× real time with the same WER
([run](docs/bench_2026-11-20_rtx4090_release/README.md)).

**Speaker labels.** Accept the conditions on
[pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
and run `hf auth login`. rumblex turns off pyannote's telemetry unless you set
`PYANNOTE_METRICS_ENABLED=true`.

**Server.** `rumblex serve` answers OpenAI's transcription API on port 9000, so the `openai` client and
tools built on it can use rumblex by changing the base URL (`pip install "rumblex[server]"`). It listens on
127.0.0.1 unless you pass `--host`; set `--api_key` before exposing it. Formats and a client example:
[docs/server.md](docs/server.md).

## Compared with WhisperX

Same audio, one RTX 4090, one session, rumblex 0.1.0 (bf16, batch 32) vs WhisperX 3.8.6 with `large-v3`
(float16, batch 16, default aligners). Language given to both, Whisper's normalisers for WER. Speed is one
20–60 min file per dataset, models already loaded. Details and raw results:
[docs/bench_2026-11-20_rtx4090_release](docs/bench_2026-11-20_rtx4090_release/README.md).

| dataset | rumblex WER | WhisperX WER | rumblex speed | WhisperX speed |
|---|---|---|---|---|
| AMI meetings (500 utt.) | 10.9% | 17.2% | 329× real time | 104× |
| Earnings-22 calls (500 utt.) | 9.3% | 10.6% | 430× | 102× |
| LibriSpeech test-clean (200 utt.) | 2.2% | 1.6% | 467× | 92× |
| FLEURS Dutch (364 utt.) | 9.2% | 7.0% | 380× | 79× |

rumblex is 3–5× faster, mostly because alignment reuses the encoder (0.3–0.9 s per file vs 7–17 s for
WhisperX). It wins WER on meetings and calls, mostly through fewer deleted words; WhisperX wins on clean
English and Dutch. rumblex passes clips up to 20 s whole while WhisperX runs VAD on everything, which may
flatter rumblex on these short clips. A few hundred utterances per set, so treat it as rough.

### Languages

FLEURS (read speech), 350 utterances per language, same settings as above on another RTX 4090 pod
([run](docs/bench_2026-11-22_rtx4090_languages/README.md)). WER is scored with the Open ASR Leaderboard's
normaliser, as on the model card; the run also has it with Whisper's normalisers. Speed is the ratio of the
two systems on a 20-minute file per language, since that pod was slower overall than the one above.

WhisperX large-v3 has the lower WER in most of these; rumblex is level on Spanish and French and well ahead
on Hindi. English FLEURS (Whisper's normaliser) was 6.3% vs 5.2%. Word timestamps came from the CTC head in
all but 1 of 4,288 chunks outside Hindi; for Hindi see Limitations.

<details>
<summary>WER and speed per language</summary>

| language | rumblex WER | WhisperX WER | rumblex faster by |
|---|---|---|---|
| German | 5.0% | 3.3% | 3.7× |
| Spanish | 3.1% | 3.4% | 3.2× |
| French | 5.5% | 5.6% | 3.1× |
| Italian | 2.3% | 1.8% | 3.2× |
| Portuguese | 4.0% | 3.3% | 3.0× |
| Dutch | 6.2% | 4.5% | 3.1× |
| Swedish | 10.8% | 5.8% | 3.7× |
| Polish | 7.0% | 4.1% | 3.1× |
| Russian | 6.7% | 4.6% | 2.7× |
| Turkish | 6.3% | 3.8% | 3.4× |
| Arabic | 19.1% | 14.7% | 3.4× |
| Hindi | 12.6% | 26.5% | 1.9× |

</details>

### Timestamp accuracy

The encoder runs at 13 frames per second, 80 ms a frame. Since 0.6.0 a small learned head on the encoder
states places each word's start and end inside its frame ([docs/boundary_head.md](docs/boundary_head.md)).

Measured against human labels on the Buckeye corpus of spontaneous speech, test split of
[FA-Bench](https://github.com/olewave/fa-bench) (4,513 utterances), with both tools given the reference
words, i.e. `rumblex.align` against WhisperX's aligner ([docs/fabench_eval.md](docs/fabench_eval.md)).

| Buckeye test | rumblex 0.6.0 | rumblex 0.5.0 | WhisperX |
|---|---|---|---|
| word start, mean absolute error | 25 ms | 42 ms | 58 ms |
| word end, mean absolute error | 30 ms | 41 ms | 38 ms |
| starts and ends within 20 ms (F1) | 0.54 | 0.36 | 0.28 |

The head was fitted on read English and has seen no Buckeye data. Aligners built on phone models still do
better than either tool on this benchmark; see FA-Bench's leaderboard.

On LibriSpeech test-clean, against Montreal Forced Aligner references rather than human labels, the head
took the start error from 50 to 20 ms and the end error from 54 to 31 ms
([docs/boundary_head.md](docs/boundary_head.md)). An earlier LibriSpeech run that included WhisperX is in
[bench_2026-11-23_rtx4090_timing](docs/bench_2026-11-23_rtx4090_timing/README.md).

## How it works

1. ffmpeg decodes to 16 kHz mono.
2. Silero VAD packs speech into chunks of up to 20 s (shorter files go in whole).
3. Rumble-2-Flash transcribes the chunks in batches; a hook keeps the encoder states (13 frames/s).
4. The CTC head and Viterbi align the decoder's tokens to those states, grouped into words at spaces.
5. Optionally, pyannote labels speakers and each word gets the speaker it overlaps most.
6. Writers produce the output files.

Model details: [docs/model_inspection.md](docs/model_inspection.md).

## Limitations

- The encoder runs at 13 frames/s. The boundary head places words finer than that, but it was fitted and
  measured on English only, and word ends are still less precise than starts.
- Chunks are decoded independently, so a sentence crossing a chunk boundary can be broken there. Passing
  the previous chunk's text as context made transcripts worse
  ([docs/chunk_context_eval.md](docs/chunk_context_eval.md)).
- Words split at spaces, so Chinese and Japanese get one "word" per chunk.
- WhisperX large-v3 had lower WER on read speech in most languages measured (tables above).
- Hindi word times are less reliable: Hindi text runs to about 10 tokens per second of audio against the
  encoder's 13 frames, so in fast speech the CTC path doesn't fit and 30 of 360 chunks fell back to
  proportional times.
- Hotwords were measured on English earnings calls with one list. They don't work with the vLLM backend
  or the server yet.
- vLLM needs Linux and a GPU vLLM supports.

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

Benchmark scripts are in [scripts/README.md](scripts/README.md), and every number above links to its raw
results ([docs/README.md](docs/README.md)).

## License

Apache-2.0, like Rumble-2-Flash. The pyannote models have their own terms.
