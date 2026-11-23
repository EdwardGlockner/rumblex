"""vLLM backend with a fake OpenAI client (no server here; the real one runs on the rented GPU)."""

from types import SimpleNamespace

import numpy as np
import torch

from rumblex.transcriber import ASR_TEXT_ID, EOS_ID
from rumblex.vad import SpeechChunk
from rumblex.vllm_backend import VLLMTranscriber, completion_logprobs, language_name, parse_completion

WORDS = {500: " Hello", 501: " world", 502: ".", 10: "language", 11: " English"}


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        inv = {v: k for k, v in WORDS.items()}
        return [inv.get(" " + w, 502) for w in text.replace(".", " .").split()]

    def decode(self, ids, **kw):
        return "".join(WORDS[i] for i in ids if i in WORDS)


def choice(token_ids=None, content=None, logprobs=None):
    lp = None if logprobs is None else SimpleNamespace(content=[SimpleNamespace(logprob=x) for x in logprobs])
    return SimpleNamespace(
        token_ids=token_ids, message=SimpleNamespace(content=content), model_extra={}, logprobs=lp
    )


def test_completion_logprobs_follow_text_tokens():
    row = [10, 11, ASR_TEXT_ID, 500, 501, EOS_ID]
    assert completion_logprobs(choice(row, logprobs=[-0.1, -0.2, -0.3, -0.4, -0.5, -0.6])) == [-0.4, -0.5]
    assert completion_logprobs(choice(row)) is None  # server sent no logprobs
    assert completion_logprobs(choice(row, logprobs=[-0.1])) is None  # don't line up
    assert completion_logprobs(choice(content="language English<asr_text> Hello")) is None


def test_parse_completion_token_ids_and_text():
    tok = FakeTokenizer()
    enc = lambda t: tok.encode(t)
    dec = lambda ids: tok.decode(ids)
    ids, lang = parse_completion(choice([10, 11, ASR_TEXT_ID, 500, 501, EOS_ID]), enc, dec, None)
    assert ids == [500, 501] and lang == "English"
    ids, lang = parse_completion(
        choice(content="language English<asr_text> Hello world<|im_end|>"), enc, dec, None
    )
    assert ids == [500, 501] and lang == "English"
    ids, lang = parse_completion(choice(content=" Hello world"), enc, dec, "Dutch")
    assert ids == [500, 501] and lang == "Dutch"
    assert parse_completion(choice(content="language None<asr_text>"), enc, dec, None) == ([], None)


def test_language_name():
    assert (
        language_name("nl") == "Dutch"
        and language_name("English") == "English"
        and language_name(None) is None
    )


class FakeClient:
    def __init__(self):
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.requests.append(kw)
        return SimpleNamespace(choices=[choice([500, 501, 502, EOS_ID], logprobs=[-0.1, -0.2, -0.3, -0.01])])


class FakeEncoder:
    device, dtype = torch.device("cpu"), torch.float32

    def states(self, clips):
        return [torch.zeros(len(c) // 1231, 896) for c in clips]


def test_transcribe_chunks_requests_and_states():
    client = FakeClient()
    tr = VLLMTranscriber(client=client, tokenizer=FakeTokenizer(), encoder=FakeEncoder(), concurrency=2)
    audio = np.zeros(16000 * 10, np.float32)
    chunks = [SpeechChunk(0, 16000 * 2), SpeechChunk(16000 * 3, 16000 * 9)]
    out = tr.transcribe_chunks(
        audio, chunks, language="nl", prompt="Kubernetes", digits=False, keep_encoder_states=True
    )
    assert [o.text for o in out] == ["Hello world.", "Hello world."] and [o.chunk for o in out] == chunks
    assert all(o.language == "Dutch" and o.token_ids == [500, 501, 502] for o in out)
    assert out[0].token_logprobs == [-0.1, -0.2, -0.3]
    assert out[1].encoder_states.shape[0] > out[0].encoder_states.shape[0]  # states follow chunk order
    r = client.requests[0]
    assert r["temperature"] == 0 and r["logprobs"] is True and r["extra_body"]["chat_template_kwargs"] == {
        "digits": False, "filler": False, "verbatim": True, "language": "Dutch"}  # fmt: skip
    assert r["messages"][0] == {"role": "system", "content": "Kubernetes"}
    assert r["messages"][1]["content"][0]["audio_url"]["url"].startswith("data:audio/wav;base64,")
    res = tr.transcribe(audio, chunks=chunks, language="English")
    assert res.metadata["model"] == "Reson8/Rumble-2-Flash" and len(res.segments) == 2


def test_device_does_not_load_the_encoder():
    tr = VLLMTranscriber(client=FakeClient(), tokenizer=FakeTokenizer(), device="cpu", dtype="float32")
    assert str(tr.device) == "cpu" and tr.dtype == torch.float32 and tr.model.name_or_path
    assert tr._encoder is None  # only built when encoder states are requested
