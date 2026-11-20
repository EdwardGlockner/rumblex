## librispeech
| system | WER | start MAE | starts ≤ 100 ms | end MAE | ends ≤ 100 ms | long file speed | peak GPU |
|---|---|---|---|---|---|---|---|
| rumblex_librispeech | 2.2% | 57.6 ms | 85.3% | 57.3 ms | 85.9% | 466.9x | 9360 MB |
| rumblex_vllm_librispeech | 2.2% | 57.6 ms | 85.3% | 57.3 ms | 85.9% | 601.0x | 18753 MB |
| whisperx_librispeech | 1.6% | 74.8 ms | 82.5% | 33.9 ms | 94.1% | 92.4x | 11324 MB |

- rumblex_librispeech: {'model': 'Reson8/Rumble-2-Flash', 'device': 'cuda:0', 'compute_type': 'bfloat16', 'batch_size': 32, 'dtype': 'torch.bfloat16', 'digits': False, 'max_chunk_s': 20.0, 'language': 'English'} | NVIDIA GeForce RTX 4090, torch 2.14.1+cu126, commit a4b1461, 2026-11-20 | {'dataset': 'librispeech', 'split': 'test_clean', 'n': 200, 'seed': 0, 'wer_normalizer': 'whisper-english'}
- rumblex_vllm_librispeech: {'model': 'Reson8/Rumble-2-Flash', 'device': 'cuda:0', 'compute_type': 'bfloat16', 'batch_size': 8, 'dtype': 'torch.bfloat16', 'digits': False, 'max_chunk_s': 20.0, 'backend': 'vllm', 'base_url': 'http://localhost:8000/v1', 'concurrency': 32, 'language': 'English'} | NVIDIA GeForce RTX 4090, torch 2.14.1+cu126, commit a4b1461, 2026-11-20 | {'dataset': 'librispeech', 'split': 'test_clean', 'n': 200, 'seed': 0, 'wer_normalizer': 'whisper-english'}
- whisperx_librispeech: {'model': 'large-v3', 'device': 'cuda', 'compute_type': 'float16', 'batch_size': 16, 'whisperx': '3.8.6', 'language': 'English'} | NVIDIA GeForce RTX 4090, torch 2.8.0+cu128, commit a4b1461, 2026-11-20 | {'dataset': 'librispeech', 'split': 'test_clean', 'n': 200, 'seed': 0, 'wer_normalizer': 'whisper-english'}

## ami
| system | WER | start MAE | starts ≤ 100 ms | end MAE | ends ≤ 100 ms | long file speed | peak GPU |
|---|---|---|---|---|---|---|---|
| rumblex_ami | 10.9% | n/a | n/a | n/a | n/a | 329.0x | 8216 MB |
| whisperx_ami | 17.2% | n/a | n/a | n/a | n/a | 104.4x | 10088 MB |

- rumblex_ami: {'model': 'Reson8/Rumble-2-Flash', 'device': 'cuda:0', 'compute_type': 'bfloat16', 'batch_size': 32, 'dtype': 'torch.bfloat16', 'digits': False, 'max_chunk_s': 20.0, 'language': 'English'} | NVIDIA GeForce RTX 4090, torch 2.14.1+cu126, commit a4b1461, 2026-11-20 | {'dataset': 'ami', 'split': 'test', 'n': 500, 'seed': 0, 'wer_normalizer': 'whisper-english'}
- whisperx_ami: {'model': 'large-v3', 'device': 'cuda', 'compute_type': 'float16', 'batch_size': 16, 'whisperx': '3.8.6', 'language': 'English'} | NVIDIA GeForce RTX 4090, torch 2.8.0+cu128, commit a4b1461, 2026-11-20 | {'dataset': 'ami', 'split': 'test', 'n': 500, 'seed': 0, 'wer_normalizer': 'whisper-english'}

## earnings22
| system | WER | start MAE | starts ≤ 100 ms | end MAE | ends ≤ 100 ms | long file speed | peak GPU |
|---|---|---|---|---|---|---|---|
| rumblex_earnings22 | 9.3% | n/a | n/a | n/a | n/a | 430.3x | 9030 MB |
| whisperx_earnings22 | 10.6% | n/a | n/a | n/a | n/a | 102.0x | 10466 MB |

- rumblex_earnings22: {'model': 'Reson8/Rumble-2-Flash', 'device': 'cuda:0', 'compute_type': 'bfloat16', 'batch_size': 32, 'dtype': 'torch.bfloat16', 'digits': False, 'max_chunk_s': 20.0, 'language': 'English'} | NVIDIA GeForce RTX 4090, torch 2.14.1+cu126, commit a4b1461, 2026-11-20 | {'dataset': 'earnings22', 'split': 'test', 'n': 500, 'seed': 0, 'wer_normalizer': 'whisper-english'}
- whisperx_earnings22: {'model': 'large-v3', 'device': 'cuda', 'compute_type': 'float16', 'batch_size': 16, 'whisperx': '3.8.6', 'language': 'English'} | NVIDIA GeForce RTX 4090, torch 2.8.0+cu128, commit a4b1461, 2026-11-20 | {'dataset': 'earnings22', 'split': 'test', 'n': 500, 'seed': 0, 'wer_normalizer': 'whisper-english'}

## fleurs_nl
| system | WER | start MAE | starts ≤ 100 ms | end MAE | ends ≤ 100 ms | long file speed | peak GPU |
|---|---|---|---|---|---|---|---|
| rumblex_fleurs_nl | 9.2% | n/a | n/a | n/a | n/a | 379.9x | 9956 MB |
| whisperx_fleurs_nl | 7.0% | n/a | n/a | n/a | n/a | 78.7x | 12422 MB |

- rumblex_fleurs_nl: {'model': 'Reson8/Rumble-2-Flash', 'device': 'cuda:0', 'compute_type': 'bfloat16', 'batch_size': 32, 'dtype': 'torch.bfloat16', 'digits': True, 'max_chunk_s': 20.0, 'language': 'Dutch'} | NVIDIA GeForce RTX 4090, torch 2.14.1+cu126, commit a4b1461, 2026-11-20 | {'dataset': 'fleurs_nl', 'split': 'test', 'n': 364, 'seed': 0, 'wer_normalizer': 'whisper-basic'}
- whisperx_fleurs_nl: {'model': 'large-v3', 'device': 'cuda', 'compute_type': 'float16', 'batch_size': 16, 'whisperx': '3.8.6', 'language': 'Dutch'} | NVIDIA GeForce RTX 4090, torch 2.8.0+cu128, commit a4b1461, 2026-11-20 | {'dataset': 'fleurs_nl', 'split': 'test', 'n': 364, 'seed': 0, 'wer_normalizer': 'whisper-basic'}

