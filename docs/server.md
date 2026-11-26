# Server

`rumblex serve` answers OpenAI's transcription API, so the `openai` client and tools built on it can use
rumblex by changing the base URL (`pip install "rumblex[server]"`):

```bash
rumblex serve                                        # port 9000; add --diarize for speaker labels
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:9000/v1", api_key="unused")
result = client.audio.transcriptions.create(
    model="rumble-2-flash",
    file=open("meeting.mp3", "rb"),
    response_format="verbose_json",
    timestamp_granularities=["word"],
)
print(result.words[0].word, result.words[0].start, result.words[0].probability)
```

It supports the `json`, `text`, `srt`, `vtt` and `verbose_json` formats, and `diarized_json` with
`--diarize`. Words in `verbose_json` also have rumblex's `probability` and `score`. The server listens on
127.0.0.1 unless you pass `--host`; set `--api_key` (or `RUMBLEX_API_KEY`) before exposing it. Requests run
one at a time. `model` and `temperature` are accepted and ignored. Hotwords are not available through the
server yet.
