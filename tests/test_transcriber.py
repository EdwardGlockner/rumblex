"""Model-free tests for the transcriber's batching and output parsing.
The model itself is exercised on a GPU by scripts/check_transcriber.py."""

import pytest
import torch

from rumblex.transcriber import (
    ASR_TEXT_ID,
    EOS_ID,
    FIRST_SPECIAL_ID,
    ChunkTranscript,
    _TokenLogprobs,
    language_from_prefix,
    majority_language,
    make_batches,
    parse_generated,
    split_packed,
    text_positions,
)
from rumblex.vad import SpeechChunk

PAD = FIRST_SPECIAL_ID  # <|endoftext|>, the pad token
LANG_EN = [10, 11]  # stand-ins for the tokens of "language English"


def fake_decode(ids):
    return {(10, 11): "language English", (10, 12): "language None", (10, 13): "language Dutch"}[tuple(ids)]


def test_parse_detected_language():
    row = [*LANG_EN, ASR_TEXT_ID, 500, 501, 502, EOS_ID, PAD, PAD]
    assert parse_generated(row, fake_decode) == ([500, 501, 502], "English")


def test_parse_forced_language():
    row = [500, 501, EOS_ID, PAD]
    assert parse_generated(row, fake_decode, forced_language="Dutch") == ([500, 501], "Dutch")


def test_parse_empty_audio_answer():
    row = [10, 12, ASR_TEXT_ID, EOS_ID]
    assert parse_generated(row, fake_decode) == ([], None)


def test_parse_drops_specials_and_stops_at_eos():
    row = [*LANG_EN, ASR_TEXT_ID, 500, FIRST_SPECIAL_ID + 3, 501, EOS_ID, 777]
    assert parse_generated(row, fake_decode) == ([500, 501], "English")


def test_parse_truncated_without_eos():
    row = [*LANG_EN, ASR_TEXT_ID, 500, 500, 500]
    assert parse_generated(row, fake_decode) == ([500, 500, 500], "English")


@pytest.mark.parametrize(
    ("text", "lang"),
    [("language English", "English"), ("Dutch", "Dutch"), ("language None", None), ("  ", None)],
)
def test_language_from_prefix(text, lang):
    assert language_from_prefix(text) == lang


def test_split_packed():
    states = torch.arange(10.0).reshape(10, 1)
    a, b, c = split_packed(states, [3, 5, 2])
    assert a[:, 0].tolist() == [0, 1, 2] and b.shape[0] == 5 and c[:, 0].tolist() == [8, 9]
    with pytest.raises(ValueError):
        split_packed(states, [3, 3])


def test_make_batches_longest_first_and_complete():
    durs = [3.0, 19.0, 7.0, 12.0, 1.0]
    batches = make_batches(durs, 2)
    assert batches == [[1, 3], [2, 0], [4]]
    assert sorted(i for b in batches for i in b) == list(range(len(durs)))
    assert make_batches([], 4) == []


def _ct(dur: float, lang: str | None) -> ChunkTranscript:
    return ChunkTranscript(chunk=SpeechChunk(0, int(dur * 16000)), text="x", language=lang, token_ids=[])


def test_majority_language_weights_by_duration():
    items = [_ct(2, "English"), _ct(2, "English"), _ct(15, "Dutch"), _ct(5, None)]
    assert majority_language(items) == "Dutch"
    assert majority_language([_ct(3, None)]) is None


def test_trim_to_hop():
    import numpy as np

    from rumblex.transcriber import HOP, trim_to_hop

    assert len(trim_to_hop(np.zeros(HOP * 7 + 159))) == HOP * 7
    assert len(trim_to_hop(np.zeros(HOP * 3))) == HOP * 3


def test_batched_features_match_single():
    """Prepared clips get the same audio frames and identical mel features alone as inside a padded
    batch. Without prepare_clip and zero_feature_padding, the frame count could differ by one, the last
    mel frame differed, and the padding inside the last conv block differed."""
    import numpy as np

    pytest.importorskip("transformers")
    from huggingface_hub import try_to_load_from_cache

    if not isinstance(try_to_load_from_cache("Reson8/Rumble-2-Flash", "tokenizer_config.json"), str):
        pytest.skip("model files not cached")
    from transformers import AutoProcessor

    from rumblex.transcriber import prepare_clip, zero_feature_padding

    p = AutoProcessor.from_pretrained("Reson8/Rumble-2-Flash")
    pad = p.tokenizer.convert_tokens_to_ids(p.audio_token)
    rng = np.random.default_rng(0)
    clips = [
        prepare_clip((0.01 * rng.standard_normal(n)).astype(np.float32)) for n in (317696, 140928, 205376)
    ]
    single = [int((p.apply_transcription_request(audio=[c])["input_ids"] == pad).sum()) for c in clips]
    batched = (p.apply_transcription_request(audio=clips)["input_ids"] == pad).sum(-1).tolist()
    assert single == batched
    batch = p.apply_transcription_request(audio=clips)
    zero_feature_padding(batch)
    for i, c in enumerate(clips):
        one = p.apply_transcription_request(audio=[c])
        zero_feature_padding(one)
        n = int(one["input_features_mask"].sum())
        assert int(batch["input_features_mask"][i].sum()) == n
        # the convs also read the padding up to the end of the clip's last 100-frame block
        blk = one["input_features"].shape[-1]
        assert torch.allclose(one["input_features"][0], batch["input_features"][i, :, :blk], atol=1e-5)
        assert torch.allclose(one["input_features"][0, :, :n], batch["input_features"][i, :, :n], atol=1e-5)


def test_short_audio_skips_vad(monkeypatch):
    """Audio up to max_chunk_s is transcribed whole; longer audio goes through VAD."""
    import numpy as np

    import rumblex.transcriber as T

    seen = {}

    def fake_chunks(self, audio, chunks, **kw):
        seen["chunks"] = list(chunks)
        return [T.ChunkTranscript(c, "x", "English", [1]) for c in chunks]

    called = {"vad": 0}

    def fake_vad(audio, **kw):
        called["vad"] += 1
        return [T.SpeechChunk(0, len(audio))]

    monkeypatch.setattr(T.RumbleTranscriber, "transcribe_chunks", fake_chunks)
    monkeypatch.setattr(T, "vad_chunks", fake_vad)
    tr = object.__new__(T.RumbleTranscriber)  # no model needed
    tr.device = "cpu"

    class _M:
        name_or_path = "fake"

    tr.model = _M()
    tr.transcribe(np.zeros(16000 * 12, np.float32), max_chunk_s=20)
    assert called["vad"] == 0 and len(seen["chunks"]) == 1 and seen["chunks"][0].end == 16000 * 12
    tr.transcribe(np.zeros(16000 * 25, np.float32), max_chunk_s=20)
    assert called["vad"] == 1


def test_forced_language_comes_back_resolved(monkeypatch):
    """language="nl" must give result.language == "Dutch", the same name the segments carry."""
    import numpy as np

    import rumblex.transcriber as T

    def fake_chunks(self, audio, chunks, language=None, **kw):
        return [T.ChunkTranscript(c, "hallo", T.language_name(language), [1]) for c in chunks]

    monkeypatch.setattr(T.RumbleTranscriber, "transcribe_chunks", fake_chunks)
    tr = object.__new__(T.RumbleTranscriber)
    tr.device = "cpu"
    tr.model = type("M", (), {"name_or_path": "fake"})()
    res = tr.transcribe(np.zeros(16000 * 3, np.float32), language="nl")
    assert res.language == "Dutch" and res.segments[0].language == "Dutch"


def test_resolve_dtype():
    import torch

    from rumblex.transcriber import resolve_dtype

    assert resolve_dtype("BF16") == torch.bfloat16 and resolve_dtype("float16") == torch.float16
    assert resolve_dtype("fp32") == torch.float32 and resolve_dtype(torch.float16) == torch.float16
    assert resolve_dtype(None) is None
    with pytest.raises(ValueError, match="bfloat16"):
        resolve_dtype("int8")


def test_batch_size_must_be_positive():
    from rumblex.transcriber import RumbleTranscriber

    with pytest.raises(ValueError, match="batch_size"):
        RumbleTranscriber(batch_size=0)


def test_quiet_loading_restores_transformers_logging():
    from transformers.utils import logging as hf_logging

    from rumblex.transcriber import quiet_loading

    before = hf_logging.get_verbosity(), hf_logging.is_progress_bar_enabled()
    with quiet_loading():
        assert not hf_logging.is_progress_bar_enabled()
        assert hf_logging.get_verbosity() == hf_logging.ERROR
    assert (hf_logging.get_verbosity(), hf_logging.is_progress_bar_enabled()) == before


def test_text_positions_match_parse_generated():
    rows = [
        [*LANG_EN, ASR_TEXT_ID, 500, 501, 502, EOS_ID, PAD],
        [500, 501, EOS_ID],
        [*LANG_EN, ASR_TEXT_ID, 500, PAD + 70, 501, EOS_ID, 502],
        [*LANG_EN, ASR_TEXT_ID, 500, 500, 500],
        [10, 12, ASR_TEXT_ID, EOS_ID],
    ]
    for row in rows:
        ids, _ = parse_generated(row, fake_decode)
        assert [row[p] for p in text_positions(row)] == ids


def test_token_logprobs_recorder_keeps_the_greedy_token():
    rec = _TokenLogprobs()
    scores = torch.tensor([[0.0, 2.0, 0.0], [3.0, 0.0, 0.0]])
    assert rec(None, scores) is scores
    rec(None, torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]))
    toks, lps = rec.rows()
    assert toks == [[1, 2], [0, 0]]
    assert lps[0, 0] == pytest.approx(float(torch.log_softmax(scores[0], -1)[1]))
    assert lps[1, 1] == pytest.approx(float(torch.log(torch.tensor(1 / 3))))
