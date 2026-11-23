# Decoder log-probs through vLLM vs transformers, A40, 2026-11-23

Raw output of `scripts/runpod_logprobs.sh` (which runs `scripts/check_logprobs_vllm.py`) at commit
`b96b592`, on a rented Runpod pod (A40 48 GB, Secure Cloud, CA-MTL-1, driver 595.112.07). vLLM 0.34.0
serves the model with `--max-model-len 4096 --gpu-memory-utilization 0.5`; the transformers backend runs on
the same GPU. Both in bf16.

The same chunks (25 AMI and 25 Earnings-22 utterances, seed 0, chunked as `rumblex.transcribe` does, English
forced, digits off) go through both backends. For each chunk the check asks whether vLLM returned one
log-prob per text token, and where both backends produced the same tokens it compares the decoder log-prob
of every token and the word `probability` built from them.

| | chunks | vLLM log-probs missing | same tokens | token log-prob diff, mean / p99 / max | word probability diff, mean / p99 / max |
|---|---|---|---|---|---|
| AMI | 23 | 0 | 23 | 0.0077 / 0.071 / 0.229 | 0.0055 / 0.056 / 0.097 |
| Earnings-22 | 27 | 0 | 26 | 0.0022 / 0.053 / 0.106 | 0.0010 / 0.020 / 0.028 |

Every chunk got its log-probs from the server, lined up with its token ids. The differences are of the size
bf16 rounding gives between two implementations; one Earnings-22 chunk decoded to different tokens and is
left out of the comparison.

The script's last step (the whole pipeline through vLLM on the files in `samples/`) didn't run: `samples/`
is git-ignored, so the clone on the pod had none, and the script stopped before writing its JSON.
`logprobs_vllm.json` here holds the per-set lines it had already printed (`check_logprobs.log`, its log without the warning lines). The script
now writes the sets first and skips missing samples.

Package versions: `envs.txt`. Session log: `session.log`.
