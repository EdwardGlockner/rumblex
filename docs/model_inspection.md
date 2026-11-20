# Rumble-2-Flash: what's in the checkpoint

Inspected 2026-11-20 against `Reson8/Rumble-2-Flash` @ `95bd5f9`. Produced by
`scripts/inspect_rumble_ctc.py`; raw output in [inspection_cpu.json](inspection_cpu.json).

## Files

`config.json`, `model.safetensors`, tokenizer files, `chat_template.jinja`, plus the CTC head:

- `ctc_head.py` – reference code (`RumbleCTC`), greedy CTC decoding only
- `ctc/ctc_head.safetensors` – 357 MB, one `nn.Linear(896 -> 99631)`
- `ctc/ctc_vocab.json` – metadata + class -> token id map

Not gated, no HF token needed. `ctc/` sits in a subfolder so vLLM doesn't try to load it as model weights.

## Architecture

| | |
|---|---|
| Encoder | Qwen3-ASR audio tower, 18 layers, d_model 896, 14 heads, 128 mel bins |
| Decoder | Qwen3, 14 layers, hidden 1024, vocab 99,630 (custom; not Qwen's 151k) |
| Audio | 16 kHz mono, n_fft 400, hop 160 (100 mel frames/s) |
| Encoder frame rate | 13 frames/s on average (~76.9 ms/frame), confirmed: 12.98–13.04 measured. 13 frames per 1 s conv block, 80 ms apart inside the block |
| Encoder attention | 8 s windows (104 frames), counted from clip start |
| transformers | `Qwen3ASRForConditionalGeneration`, needs transformers >= 5.21 (Python >= 3.11) |

## CTC head

- Input: output of `audio_tower.ln_post` (before the projector into the LLM). Taken with a forward hook.
- 99,631 classes: classes 0..99,629 map 1:1 onto the decoder tokenizer (identity mapping); 99,630 is blank.
  So CTC and decoder share the same byte-level BPE units. There's no cross-vocabulary mapping to build.
- Trained jointly with the decoder (decoder loss + CTC loss).
- Very peaky: 60–76% of frames are blank, mean top prob ~0.91. A token occupies ~1.2 frames on average
  in the Viterbi path.
- Greedy CTC text is usable but clearly worse than the decoder ("Mr Qu quilt", "apost", "bowl" for "bull").
  Token edit distance CTC vs decoder: 3/22, 3/25, 23/147, 39/265.
- Not conditioned on the prompt: it emitted "uh" even though the decoder default drops fillers.

## Reusing the encoder pass

During `model.generate()` the encoder runs once, and hooking `ln_post` gives exactly the states
`RumbleCTC.ln_post_states` computes (max abs diff 0.0, argmax agreement 100% on all 4 clips, sdpa attention).
So alignment costs one extra linear layer plus Viterbi, not a second encoder pass.

## Forced alignment, first look

Plain CTC Viterbi over decoder tokens, on the head's log-probs: it worked on all 4 clips (6–60 s),
taking 0.004–0.43 s in pure numpy. Word starts look plausible.
Word ends do not: since a token is a ~1-frame spike, a word's span ends right after its last spike
(e.g. "middle" 2.38–2.46 s while "classes" starts 2.77 s). Consequences for `aligner.py`:

- Resolution is 77–80 ms per frame. WhisperX's wav2vec2 is 20 ms. Starts will be ±1 frame at best.
- End times need a rule, e.g. extend to the next word's start minus trailing blank frames, or use energy/VAD
  inside the gap. This has to be measured against reference timings, not tuned by eye.
- Units are BPE tokens, not characters. Word boundaries come from the `Ġ` (leading space) marker.
- The Viterbi needs to be vectorised (or run on torch) for hour-long files.

## Transcription options

Control tokens are inserted in the system prompt by the chat template. The JSON records the control tokens
for each option. The outputs below were checked by hand on `bcn_weather.mp3` and are not in the JSON output:

| option | output |
|---|---|
| default (`<digits><nofiller><verbatim>`) | ...it was 35 degrees... will go go down to minus 20 degrees. |
| `digits=False` | ...it was thirty five degrees... minus twenty degrees. |
| `filler=True` | ...it was uh 35 degrees... but today uh the temperature... minus uh 20... |
| `verbatim=False` | ...will go down to minus 20 degrees. |

Language: `language="Dutch"` forces it (`language <Name><asr_text>` appended), otherwise the model detects it and
prints that prefix itself. `processor.apply_transcription_request(audio, language=, prompt=)`: `prompt` is
free-text context/hotwords placed in the system prompt. That is the model's built-in way to bias it towards given words.

## Tokenizer fragmentation

Leading space changes the first piece only (`par` vs `Ġpar`); the rest of the split stays the same in these examples.

| word | pieces (with leading space) |
|---|---|
| paracetamol | Ġpar · acet · am · ol |
| metoprolol | Ġmet · op · rol · ol |
| ziekenhuisopname | Ġzie · ken · h · uis · op · name |
| arbeidsongeschiktheidsverzekering | Ġarbe · ids · ong · esch · ik · the · ids · ver · zek · ering (no space: ar · be · ...) |
| huisartsenpraktijk | Ġhuis · arts · en · pr · akt · ijk (no space: h · uis · ...) |
| Kubernetes | ĠKubernetes (no space: K · ubernetes) |
| deployment / pipeline | single token |
| debuggen | Ġdebug · gen |

Dutch compounds and drug names split into 4–10 pieces. English tech words are often a single token.

## Older GPUs (Pascal) and CPU

- GTX 1070 / 1060 are Pascal (sm_61): no bf16, slow fp16. torch 2.14 `+cu126` still supports them. Use float32.
- CPU float32 works: the decoder took 4.6–66 s per clip (slow), the CTC path 1.0–4.2 s.
- GPU float32, one clip at a time (no batching), GTX 1060 6GB, sdpa, torch 2.14+cu126. Raw
  output [inspection_gpu_fp32.json](inspection_gpu_fp32.json) (it does not record the GPU name). CPU
  columns from [inspection_cpu.json](inspection_cpu.json).

  | clip | audio | decoder (GPU) | decoder (CPU) | CTC path (GPU) | CTC path (CPU) | Viterbi (numpy) |
  |---|---|---|---|---|---|---|
  | librispeech_mr_quilter.wav | 5.9 s | 1.64 s* | 33.9 s | 0.92 s* | 0.96 s | 0.003 s |
  | bcn_weather.mp3 | 11.0 s | 1.26 s | 4.57 s | 0.13 s | 1.37 s | 0.007 s |
  | obama_first_45_secs.mp3 | 45.0 s | 5.98 s | 66.0 s | 0.27 s | 3.71 s | 0.18 s |
  | two_speakers.wav | 60.0 s | 14.9 s | 40.0 s | 0.37 s | 4.23 s | 0.53 s |

  \* first clip includes CUDA warm-up. "CTC path" is mel + a second encoder pass + head, i.e. the cost the
  reuse of the `generate()` states avoids. The Viterbi column is from the GPU run's JSON. Model load 7.5 s;
  peak torch allocation 4.2 GB. GPU and CPU decoder text was identical on all four clips. Captured vs
  recomputed `ln_post` states: max abs diff 1e-5 to 8e-5 on GPU (0.0 on CPU), CTC argmax agreement 100%.
  These are single-clip latencies; batched throughput is measured by `scripts/check_transcriber.py`.
