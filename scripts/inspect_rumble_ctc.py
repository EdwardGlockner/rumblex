"""Inspect Reson8/Rumble-2-Flash and its auxiliary CTC head.

Static checks (config, CTC vocab, tokenizer) plus runtime checks on real audio:
frame rate, logit shapes, CTC peakiness, CTC-vs-decoder agreement, whether the encoder
states from generate() can be reused for the head, and a rough Viterbi word alignment.

Options are checked on the first clip. docs/inspection_cpu.json came from the four files in samples/,
librispeech_mr_quilter.wav first, then bcn_weather.mp3, obama_first_45_secs.mp3, two_speakers.wav:

    python scripts/inspect_rumble_ctc.py <clips> --device cpu --out docs/inspection_cpu.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = "Reson8/Rumble-2-Flash"
ASR_TEXT_ID = 99623  # <asr_text>

FRAGMENT_WORDS = [
    "paracetamol",
    "Paracetamol",
    "ibuprofen",
    "metoprolol",
    "amoxicilline",
    "ziekenhuisopname",
    "arbeidsongeschiktheidsverzekering",
    "huisartsenpraktijk",
    "deployment",
    "pipeline",
    "debuggen",
    "Kubernetes",
    "Reson8",
    "RumbleX",
]


def load_ctc_module(snapshot: Path):
    spec = importlib.util.spec_from_file_location("rumble_ctc_head", snapshot / "ctc_head.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_audio(path: str, sr: int = 16000) -> np.ndarray:
    """Decode anything ffmpeg understands to mono float32 at `sr`."""
    import shutil
    import subprocess

    exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError("ffmpeg not found on PATH")
    cmd = [exe, "-nostdin", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def edit_distance(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def ctc_viterbi(logp: np.ndarray, targets: list[int], blank: int) -> list[tuple[int, int]] | None:
    """Plain CTC Viterbi. logp: [T, V] log-probs. Returns (first_frame, last_frame) per target token."""
    T, L = logp.shape[0], len(targets)
    S = 2 * L + 1
    ext = [blank if s % 2 == 0 else targets[s // 2] for s in range(S)]
    em = logp[:, ext]  # [T, S]
    NEG = -1e30
    dp = np.full((T, S), NEG, dtype=np.float64)
    bp = np.zeros((T, S), dtype=np.int8)
    dp[0, 0] = em[0, 0]
    if S > 1:
        dp[0, 1] = em[0, 1]
    for t in range(1, T):
        for s in range(S):
            best, arg = dp[t - 1, s], 0
            if s >= 1 and dp[t - 1, s - 1] > best:
                best, arg = dp[t - 1, s - 1], 1
            if s >= 2 and ext[s] != blank and ext[s] != ext[s - 2] and dp[t - 1, s - 2] > best:
                best, arg = dp[t - 1, s - 2], 2
            dp[t, s] = best + em[t, s]
            bp[t, s] = arg
    s = S - 1 if S == 1 or dp[-1, S - 1] >= dp[-1, S - 2] else S - 2
    if dp[-1, s] <= NEG / 2:
        return None
    path = []
    for t in range(T - 1, -1, -1):
        path.append(s)
        s -= int(bp[t, s])
    path.reverse()
    spans: list[list[int]] = [[-1, -1] for _ in range(L)]
    for t, s in enumerate(path):
        if s % 2 == 1:
            k = s // 2
            if spans[k][0] < 0:
                spans[k][0] = t
            spans[k][1] = t
    return [tuple(x) for x in spans]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="+")
    ap.add_argument("--model", default=REPO)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

    snap = Path(snapshot_download(args.model)) if not Path(args.model).is_dir() else Path(args.model)
    report: dict = {
        "model": args.model,
        "snapshot": str(snap),
        "device": args.device,
        "dtype": args.dtype,
        "torch": torch.__version__,
    }

    cfg = json.loads((snap / "config.json").read_text())
    voc = json.loads((snap / "ctc/ctc_vocab.json").read_text(encoding="utf-8"))
    ac = cfg["audio_config"]
    report["static"] = {
        "encoder": {
            k: ac[k]
            for k in (
                "d_model",
                "encoder_layers",
                "encoder_attention_heads",
                "n_window",
                "n_window_infer",
                "num_mel_bins",
                "output_dim",
            )
        },
        "decoder": {k: cfg["text_config"][k] for k in ("hidden_size", "num_hidden_layers", "vocab_size")},
        "ctc_meta": {k: v for k, v in voc.items() if k != "token_ids"},
        "ctc_identity_mapping": voc["token_ids"] == list(range(len(voc["token_ids"]))),
    }

    processor = AutoProcessor.from_pretrained(snap)
    tok = processor.tokenizer
    report["static"]["tokenizer_len"] = len(tok)
    frag = {}
    for w in FRAGMENT_WORDS:
        for variant in (w, " " + w):
            ids = tok.encode(variant, add_special_tokens=False)
            frag[repr(variant)] = [tok.convert_ids_to_tokens(i) for i in ids]
    report["fragmentation"] = frag

    dtype = getattr(torch, args.dtype)
    t0 = time.perf_counter()
    model = Qwen3ASRForConditionalGeneration.from_pretrained(snap, dtype=dtype).to(args.device).eval()
    report["load_seconds"] = round(time.perf_counter() - t0, 2)
    report["attn_implementation"] = model.config._attn_implementation
    ctc_mod = load_ctc_module(snap)
    ctc = ctc_mod.RumbleCTC.from_pretrained(str(snap), model, processor)

    # capture ln_post output during generate() to test reuse
    tower = ctc_mod._audio_tower(model)
    captured: list[torch.Tensor] = []
    hook = tower.ln_post.register_forward_hook(lambda m, i, o: captured.append(o.detach()))

    def sync():
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()

    clips = []
    for path in args.audio:
        wav = load_audio(path)
        dur = len(wav) / 16000
        r: dict = {"file": Path(path).name, "seconds": round(dur, 2)}

        # autoregressive transcript (language auto-detected)
        captured.clear()
        inputs = processor.apply_transcription_request(audio=wav).to(args.device, dtype=dtype)
        sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=512)
        sync()
        r["ar_seconds"] = round(time.perf_counter() - t0, 3)
        gen = out[0, inputs["input_ids"].shape[-1] :].tolist()
        parsed = processor.decode(gen, return_format="parsed")
        r["ar_language"] = parsed.get("language") if isinstance(parsed, dict) else None
        r["ar_text"] = processor.decode(gen, return_format="transcription_only")
        text_ids = gen[gen.index(ASR_TEXT_ID) + 1 :] if ASR_TEXT_ID in gen else gen
        text_ids = [i for i in text_ids if i < 99562]  # drop eos / specials
        gen_states = captured[0] if captured else None
        r["ln_post_calls_during_generate"] = len(captured)

        # CTC head
        sync()
        t0 = time.perf_counter()
        feats = ctc.log_mel(wav)
        states = ctc.ln_post_states(feats)
        logits = ctc.logits(states).float()
        sync()
        r["ctc_seconds"] = round(time.perf_counter() - t0, 3)
        T = logits.shape[0]
        r["mel_frames"] = int(feats.shape[1])
        r["ctc_frames"] = T
        r["ctc_frames_per_second"] = round(T / dur, 3)
        r["ctc_logits_shape"] = list(logits.shape)
        r["ln_post_shape"] = list(states.shape)
        logp = torch.log_softmax(logits, -1)
        probs = logp.exp()
        top = probs.max(-1)
        r["blank_frame_fraction"] = round((top.indices == ctc.blank).float().mean().item(), 3)
        r["mean_top_prob"] = round(top.values.mean().item(), 3)
        r["frames_top_prob_gt_0.9"] = round((top.values > 0.9).float().mean().item(), 3)
        ctc_ids = ctc.greedy_ids(logits)
        r["ctc_text"] = tok.decode(
            ctc_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        ).strip()
        r["ar_tokens"] = len(text_ids)
        r["ctc_tokens"] = len(ctc_ids)
        r["token_edit_distance_ctc_vs_ar"] = edit_distance(ctc_ids, text_ids)
        r["tokens_per_frame_ar"] = round(len(text_ids) / T, 3)

        # can the generate() encoder pass be reused?
        if gen_states is not None:
            g = gen_states.reshape(-1, gen_states.shape[-1]).float()
            r["generate_ln_post_shape"] = list(gen_states.shape)
            if g.shape[0] == T:
                r["generate_vs_ctc_states_max_abs_diff"] = round((g - states.float()).abs().max().item(), 5)
                gl = ctc.logits(g.to(ctc.dtype)).float()
                r["generate_states_ctc_argmax_agreement"] = round(
                    (gl.argmax(-1) == logits.argmax(-1)).float().mean().item(), 4
                )

        # rough forced alignment of decoder tokens on CTC log-probs
        lp = logp.cpu().numpy()
        t0 = time.perf_counter()
        spans = ctc_viterbi(lp, text_ids, ctc.blank) if text_ids and len(text_ids) <= T else None
        r["viterbi_seconds"] = round(time.perf_counter() - t0, 3)
        r["viterbi_ok"] = spans is not None
        if spans:
            words, cur = [], None
            for tid, (a, b) in zip(text_ids, spans):
                piece = tok.decode([tid], clean_up_tokenization_spaces=False)
                if cur is None or piece.startswith(" "):
                    if cur:
                        words.append(cur)
                    cur = {"word": piece.strip(), "start": a / 13, "end": (b + 1) / 13}
                else:
                    cur["word"] += piece
                    cur["end"] = (b + 1) / 13
            if cur:
                words.append(cur)
            r["aligned_words_first_15"] = [
                {"w": w["word"], "s": round(w["start"], 2), "e": round(w["end"], 2)} for w in words[:15]
            ]
            r["mean_frames_per_token_span"] = round(float(np.mean([b - a + 1 for a, b in spans])), 2)
        clips.append(r)
        print(json.dumps(r, ensure_ascii=False, indent=1), flush=True)

    hook.remove()
    report["clips"] = clips

    # transcription options on the first clip
    wav = load_audio(args.audio[0])
    opts = {}
    for name, kw in {
        "default": {},
        "digits=False": {"digits": False},
        "filler=True": {"filler": True},
        "verbatim=False": {"verbatim": False},
    }.items():
        try:
            inputs = processor.apply_transcription_request(audio=wav, language="English", **kw).to(
                args.device, dtype=dtype
            )
            prompt_text = tok.decode(inputs["input_ids"][0], skip_special_tokens=False)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=512)
            opts[name] = {
                "control_tokens_in_prompt": [
                    t
                    for t in (
                        "<digits>",
                        "<nodigits>",
                        "<filler>",
                        "<nofiller>",
                        "<verbatim>",
                        "<noverbatim>",
                    )
                    if t in prompt_text
                ],
                "text": processor.decode(
                    out[0, inputs["input_ids"].shape[-1] :], return_format="transcription_only"
                ),
            }
        except Exception as e:  # noqa: BLE001 - report, don't crash
            opts[name] = {"error": f"{type(e).__name__}: {e}"}
    report["options_first_clip"] = opts
    if args.device.startswith("cuda"):
        report["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)

    print(
        json.dumps(
            {k: report[k] for k in ("static", "fragmentation", "options_first_clip")},
            ensure_ascii=False,
            indent=1,
        )
    )
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
