"""Command line: `rumblex audio.mp4 --output_format srt json`, and `rumblex serve` for an OpenAI-compatible
transcription server (rumblex.server).

Flags follow WhisperX where the meaning is the same (--model, --device, --batch_size, --compute_type,
--language, --output_dir, --output_format, --max_line_width, --max_line_count, --highlight_words,
--no_align), plus Rumble-2-Flash's own options (--prompt, --no_digits, --filler, --no_verbatim).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from rumblex import __version__
from rumblex.writers import FORMATS, WriterOptions, write_result


def _positive(cast):
    def parse(s: str):
        try:
            v = cast(s)
        except ValueError:
            raise argparse.ArgumentTypeError(f"not a number: {s!r}") from None
        if v <= 0:
            raise argparse.ArgumentTypeError(f"must be positive, got {s}")
        return v

    return parse


def _add_model_args(p: argparse.ArgumentParser) -> None:
    m = p.add_argument_group("model")
    m.add_argument("--model", default="Reson8/Rumble-2-Flash",
                   help="Hugging Face repo id or local folder (default: %(default)s)")  # fmt: skip
    m.add_argument("--device", default=None, help="cuda, cuda:1, cpu (default: cuda if available)")
    m.add_argument("--compute_type", default=None, choices=["float32", "float16", "bfloat16"],
                   help="default: bfloat16 on Ampere+ GPUs, float32 otherwise")  # fmt: skip
    m.add_argument("--batch_size", type=_positive(int), default=8,
                   help="chunks decoded together (default: %(default)s)")  # fmt: skip
    m.add_argument(
        "--vllm_url", default=None,
        help="decode on a running vLLM server, e.g. http://localhost:8000/v1 (needs the [vllm] extra)",
    )  # fmt: skip


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rumblex",
        description="Transcribe audio/video with word timestamps.",
        epilog="rumblex serve --help: run an OpenAI-compatible transcription server instead.",
    )
    p.add_argument("audio", nargs="+", help="audio or video files (anything ffmpeg reads)")
    p.add_argument("--version", action="version", version=f"rumblex {__version__}")

    _add_model_args(p)

    t = p.add_argument_group("transcription")
    t.add_argument("--language", default=None, help="force a language (name or ISO code); default: detect")
    t.add_argument("--prompt", default=None, help="context or hotwords for the model, e.g. names and terms")
    t.add_argument("--no_digits", action="store_true", help="write numbers as words")
    t.add_argument("--filler", action="store_true", help="keep fillers like 'uh', 'um'")
    t.add_argument("--no_verbatim", action="store_true", help="collapse repeated words")
    t.add_argument("--chunk_size", type=_positive(float), default=20.0,
                   help="max seconds of audio per model input (default: %(default)s)")  # fmt: skip

    a = p.add_argument_group("alignment")
    a.add_argument("--no_align", action="store_true", help="skip word timestamps (segment times only)")
    a.add_argument("--text", default=None, metavar="FILE",
                   help="align this transcript (a text file) instead of transcribing; one audio file only")  # fmt: skip
    a.add_argument("--no_boundary_head", action="store_true",
                   help="word times as in 0.5.0, without the learned start and end correction")  # fmt: skip
    a.add_argument("--end_rule", default="next_start", help=argparse.SUPPRESS)

    d = p.add_argument_group("diarization (pyannote; needs the [diarize] extra and Hugging Face access)")
    d.add_argument("--diarize", action="store_true", help="label speakers")
    d.add_argument("--num_speakers", type=int, default=None, help="exact number of speakers, if known")
    d.add_argument("--min_speakers", type=int, default=None, help="lower bound on the number of speakers")
    d.add_argument("--max_speakers", type=int, default=None, help="upper bound on the number of speakers")
    d.add_argument(
        "--diarize_model", default="pyannote/speaker-diarization-community-1", help=argparse.SUPPRESS
    )

    o = p.add_argument_group("output")
    o.add_argument(
        "--output_dir", "-o", default=".", help="where to write the files (default: current folder)"
    )
    o.add_argument("--output_format", "-f", nargs="+", default=["all"], choices=[*FORMATS, "all"],
                   metavar="FORMAT", help=f"one or more of {', '.join(FORMATS)}, all (default: all)")  # fmt: skip
    o.add_argument("--max_line_width", type=_positive(int), default=None,
                   help="split subtitle cues at this many chars")  # fmt: skip
    o.add_argument("--max_line_count", type=_positive(int), default=2,
                   help="lines per subtitle cue with --max_line_width (default: %(default)s)")  # fmt: skip
    o.add_argument("--highlight_words", action="store_true", help="vtt: per-word timing tags")
    o.add_argument("--quiet", "-q", action="store_true", help="print errors only")
    return p


def _quiet_libraries() -> None:
    """Keep the terminal to rumblex's own output: no per-tensor "Loading weights" bar from transformers,
    no torch notice about triton (pulled in by pyannote). Errors still show."""
    import logging

    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)


def _load_pipeline(args):
    """(transcriber, aligner or None, diarizer or None). Separate so tests can swap in fakes."""
    from rumblex.aligner import Aligner, AlignOptions
    from rumblex.transcriber import RumbleTranscriber

    _quiet_libraries()
    dz = None
    if args.diarize:  # before the model, so a missing extra or gated model fails fast
        from rumblex.diarizer import Diarizer

        dz = Diarizer(args.diarize_model, device=args.device)
    if args.vllm_url:
        from rumblex.vllm_backend import VLLMTranscriber

        tr = VLLMTranscriber(
            args.vllm_url, args.model, device=args.device, dtype=args.compute_type, batch_size=args.batch_size
        )
    else:
        tr = RumbleTranscriber(
            args.model, device=args.device, dtype=args.compute_type, batch_size=args.batch_size
        )
    opts = AlignOptions(end_rule=args.end_rule, boundary_head=not args.no_boundary_head)
    al = None if args.no_align else Aligner.from_transcriber(tr, opts)
    return tr, al, dz


_EXTRAS = {"pyannote": "diarize", "openai": "vllm"}


def _load_error_hint(e: Exception) -> str:
    if isinstance(e, ImportError):
        extra = _EXTRAS.get((e.name or "").split(".")[0])
        return f'install the extra: pip install "rumblex[{extra}]"' if extra else ""
    try:
        from huggingface_hub.errors import GatedRepoError
    except ImportError:
        return ""
    if isinstance(e, GatedRepoError):
        return "accept the model's conditions on huggingface.co, then run `hf auth login` or set HF_TOKEN"
    return ""


def _check_args(args) -> str | None:
    """Problems that should stop the run before anything is loaded."""
    for a in args.audio:
        if not Path(a).exists():
            return f"not found: {a}"
        if not Path(a).is_file():
            return f"not a file: {a}"
    by_stem: dict[str, list[str]] = {}
    for a in args.audio:
        by_stem.setdefault(os.path.normcase(Path(a).stem), []).append(a)
    clashes = [paths for paths in by_stem.values() if len(paths) > 1]
    if clashes:
        listed = "; ".join(", ".join(paths) for paths in clashes)
        return f"inputs with the same name would overwrite each other's output in {args.output_dir}: {listed}"
    if args.text is not None:
        if len(args.audio) != 1:
            return "--text aligns one transcript to one audio file; give a single audio file"
        if not Path(args.text).is_file():
            return f"not found: {args.text}"
        if args.no_align:
            return "--text and --no_align can't be combined: aligning the text is all --text does"
        if args.vllm_url:
            return "--text doesn't use the decoder, so --vllm_url has no effect; leave it out"
    if args.language is not None:
        args.language = args.language.strip() or None
    if args.language is not None:
        from rumblex.transcriber import language_name

        try:
            language_name(args.language)
        except ValueError:
            from transformers.models.qwen3_asr.processing_qwen3_asr import LANGUAGE_CODE_TO_NAME

            codes = ", ".join(sorted(LANGUAGE_CODE_TO_NAME))
            return f"unsupported language {args.language!r}; use a language name or one of: {codes}"
    return None


def build_serve_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rumblex serve",
        description="OpenAI-compatible transcription server: POST /v1/audio/transcriptions "
        "(needs the [server] extra). One request runs on the model at a time.",
    )
    p.add_argument("--version", action="version", version=f"rumblex {__version__}")
    s = p.add_argument_group("server")
    s.add_argument("--host", default="127.0.0.1",
                   help="address to listen on; 0.0.0.0 for all interfaces (default: %(default)s)")  # fmt: skip
    s.add_argument("--port", type=_positive(int), default=9000,
                   help="default: %(default)s (vLLM's default is 8000)")  # fmt: skip
    s.add_argument("--api_key", default=os.environ.get("RUMBLEX_API_KEY"),
                   help="require this key as a bearer token (default: $RUMBLEX_API_KEY, else none)")  # fmt: skip
    _add_model_args(p)
    t = p.add_argument_group("transcription")
    t.add_argument("--chunk_size", type=_positive(float), default=20.0,
                   help="max seconds of audio per model input (default: %(default)s)")  # fmt: skip
    t.add_argument("--no_align", action="store_true", help="skip word timestamps (segment times only)")
    t.add_argument("--diarize", action="store_true",
                   help="label speakers; enables diarized_json (needs the [diarize] extra)")  # fmt: skip
    t.add_argument(
        "--diarize_model", default="pyannote/speaker-diarization-community-1", help=argparse.SUPPRESS
    )
    t.add_argument("--no_boundary_head", action="store_true",
                   help="word times as in 0.5.0, without the learned start and end correction")  # fmt: skip
    p.set_defaults(end_rule="next_start")
    return p


def serve_main(
    argv: Sequence[str], load_pipeline: Callable = _load_pipeline, run: Callable | None = None
) -> int:
    """`rumblex serve`. `run` (tests) replaces starting uvicorn; it gets the Pipeline and the server args."""
    args = build_serve_parser().parse_args(argv)
    if (
        args.vllm_url
        and args.vllm_url.rstrip("/").endswith(f":{args.port}/v1")
        and args.host in ("127.0.0.1", "localhost", "0.0.0.0")
    ):
        print(f"error: port {args.port} is the vLLM server's; pick another with --port", file=sys.stderr)
        return 2
    missing = [m for m in ("fastapi", "uvicorn", "multipart") if not _importable(m)]
    if missing and run is None:
        print(
            f'error: rumblex serve needs {", ".join(missing)}: pip install "rumblex[server]"', file=sys.stderr
        )
        return 2
    from rumblex.server import Pipeline, serve

    _quiet_libraries()
    print(f"loading {args.model}", file=sys.stderr)
    try:
        tr, al, dz = load_pipeline(args)
    except Exception as e:  # noqa: BLE001 - a message and a hint, not a traceback
        print(f"error: could not load the model: {type(e).__name__}: {e}", file=sys.stderr)
        hint = _load_error_hint(e)
        if hint:
            print(hint, file=sys.stderr)
        return 2
    pipeline = Pipeline(tr, al, dz, chunk_size=args.chunk_size)
    if run is not None:
        run(pipeline, args)
        return 0
    if args.host not in ("127.0.0.1", "localhost", "::1") and not args.api_key:
        print(f"warning: listening on {args.host} without --api_key; anyone who can reach it can use it",
              file=sys.stderr)  # fmt: skip
    serve(pipeline, host=args.host, port=args.port, model_name=args.model, api_key=args.api_key)
    return 0


def _importable(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


def main(argv: Sequence[str] | None = None, load_pipeline: Callable = _load_pipeline) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["serve"]:
        return serve_main(argv[1:], load_pipeline)
    args = build_parser().parse_args(argv)
    # Imported here, not at the top: they pull in torch, which would make --help and --version slow.
    from rich.console import Console
    from rich.markup import escape
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

    from rumblex.audio import load_audio
    from rumblex.diarizer import assign_speakers

    # soft_wrap: long lines stay whole when the output is piped or logged
    console = Console(stderr=True, quiet=args.quiet, soft_wrap=True)
    errors = Console(stderr=True, soft_wrap=True)  # --quiet hides progress, not errors
    problem = _check_args(args)
    if problem:
        errors.print(f"[red]error:[/red] {escape(problem)}", highlight=False)
        return 2
    if args.diarize and args.no_align:
        console.print("[yellow]warning:[/yellow] with --no_align, speakers are assigned per chunk, "
                      "without word timings")  # fmt: skip
    if not args.diarize and (args.num_speakers or args.min_speakers or args.max_speakers):
        console.print("[yellow]warning:[/yellow] speaker counts are ignored without --diarize")

    t0 = time.perf_counter()
    try:
        with console.status(f"loading {args.model}"):
            tr, al, dz = load_pipeline(args)
    except Exception as e:  # noqa: BLE001 - a load failure gets a message and a hint, not a traceback
        errors.print(
            f"[red]error:[/red] could not load the model: {type(e).__name__}: {escape(str(e))}",
            highlight=False,
        )
        hint = _load_error_hint(e)
        if hint:
            errors.print(hint, markup=False, highlight=False)
        return 2
    dtype = str(tr.dtype).removeprefix("torch.")
    console.print(f"model loaded in {time.perf_counter() - t0:.1f}s ({tr.device}, {dtype})")
    wopts = WriterOptions(args.max_line_width, args.max_line_count, args.highlight_words)

    failures = 0
    for path in args.audio:
        try:
            with Progress(TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(),
                          TimeElapsedColumn(), console=console, transient=True) as prog:  # fmt: skip
                task = prog.add_task(Path(path).name, total=None)
                timings: dict[str, float] = {}
                t = time.perf_counter()
                wav = load_audio(path)
                timings["decode_s"] = time.perf_counter() - t
                t = time.perf_counter()
                on_progress = lambda done, total, task=task: prog.update(task, completed=done, total=total)
                if args.text is not None:
                    from rumblex.textalign import align_text

                    text = Path(args.text).read_text(encoding="utf-8-sig")
                    result = align_text(tr, al, wav, text, max_chunk_s=args.chunk_size, language=args.language,
                                        progress=on_progress)  # fmt: skip
                    timings["align_s"] = time.perf_counter() - t
                else:
                    result = tr.transcribe(
                        wav,
                        max_chunk_s=args.chunk_size,
                        language=args.language,
                        prompt=args.prompt,
                        digits=not args.no_digits,
                        filler=args.filler,
                        verbatim=not args.no_verbatim,
                        keep_encoder_states=al is not None,
                        progress=on_progress,
                    )
                    timings["transcribe_s"] = time.perf_counter() - t
                    if al is not None:
                        t = time.perf_counter()
                        al.align(result, wav)
                        timings["align_s"] = time.perf_counter() - t
                if dz is not None:
                    prog.update(task, description=f"{Path(path).name} (speakers)")
                    t = time.perf_counter()
                    turns = dz(wav, num_speakers=args.num_speakers, min_speakers=args.min_speakers,
                               max_speakers=args.max_speakers)  # fmt: skip
                    assign_speakers(result, turns)
                    timings["diarize_s"] = time.perf_counter() - t
            dur = result.metadata.get("duration", 0.0)
            result.metadata.pop("chunk_transcripts", None)  # encoder states: large, not for output files
            result.metadata["timings"] = {k: round(v, 3) for k, v in timings.items()}
            outs = write_result(result, path, args.output_dir, args.output_format, wopts)
            total = sum(timings.values())
            speed = f", {_times(dur / total)} real time" if total > 0 and dur else ""
            spk = ""
            if "speakers" in result.metadata:
                spk = ", " + _count(len(result.metadata["speakers"]), "speaker")
            console.print(
                f"[green]{Path(path).name}[/green]: {dur:.1f}s audio, {result.language or '?'}{spk}, "
                f"{_count(len(result.segments), 'segment')} in {total:.1f}s{speed} -> "
                f"{', '.join(p.name for p in outs)}"
            )
        except Exception as e:  # noqa: BLE001 - one bad file shouldn't stop a batch
            failures += 1
            errors.print(f"[red]{escape(Path(path).name)}: {type(e).__name__}: {escape(str(e))}[/red]")
    return 1 if failures else 0


def _times(x: float) -> str:
    return f"{x:.1f}x" if x < 10 else f"{x:.0f}x"


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}{'' if n == 1 else 's'}"


if __name__ == "__main__":
    sys.exit(main())
