| system | WER | start MAE | starts ≤ 100 ms | end MAE | ends ≤ 100 ms | long file speed | peak GPU |
|---|---|---|---|---|---|---|---|
| rumblex | 2.1% | 57.1 ms | 85.4% | 57.7 ms | 85.6% | 108.0x | 20428 MB |
| rumblex_vllm | 2.1% | 57.1 ms | 85.4% | 57.1 ms | 85.7% | 113.1x | 20315 MB |
| whisperx | 1.6% | 74.8 ms | 82.5% | 33.9 ms | 94.1% | 78.6x | 11356 MB |

- rumblex: {'model': 'Reson8/Rumble-2-Flash', 'device': 'cuda:0', 'compute_type': 'bfloat16', 'batch_size': 64, 'dtype': 'torch.bfloat16', 'digits': False, 'max_chunk_s': 30.0, 'language': 'English'} | NVIDIA GeForce RTX 4090, torch 2.14.1+cu126, commit 514bd41, 2026-11-20 | {'split': 'test_clean', 'n': 200, 'seed': 0, 'wer_normalizer': 'whisper-english'}
- rumblex_vllm: {'model': 'Reson8/Rumble-2-Flash', 'device': 'cuda:0', 'compute_type': 'bfloat16', 'batch_size': 8, 'dtype': 'torch.bfloat16', 'digits': False, 'max_chunk_s': 20.0, 'backend': 'vllm', 'base_url': 'http://localhost:8000/v1', 'concurrency': 32, 'language': 'English'} | NVIDIA GeForce RTX 4090, torch 2.14.1+cu126, commit 514bd41, 2026-11-20 | {'split': 'test_clean', 'n': 200, 'seed': 0, 'wer_normalizer': 'whisper-english'}
- whisperx: {'model': 'large-v3', 'device': 'cuda', 'compute_type': 'float16', 'batch_size': 16, 'whisperx': '3.8.6', 'language': 'English'} | NVIDIA GeForce RTX 4090, torch 2.8.0+cu128, commit 514bd41, 2026-11-20 | {'split': 'test_clean', 'n': 200, 'seed': 0, 'wer_normalizer': 'whisper-english'}
