## librispeech
| system | WER | start MAE | starts ≤ 100 ms | end MAE | ends ≤ 100 ms | long file speed | peak GPU |
|---|---|---|---|---|---|---|---|
| rumblex_librispeech | 2.1% | 57.0 ms | 85.4% | 57.1 ms | 85.8% | 169.1x | 9348 MB |
| rumblex_vllm_librispeech | 2.1% | 57.1 ms | 85.4% | 57.1 ms | 85.7% | 210.1x | 17477 MB |
| whisperx_librispeech | 1.6% | 74.8 ms | 82.5% | 33.9 ms | 94.1% | 53.2x | 11260 MB |

- rumblex_librispeech: {'model': 'Reson8/Rumble-2-Flash', 'device': 'cuda:0', 'compute_type': 'bfloat16', 'batch_size': 32, 'dtype': 'torch.bfloat16', 'digits': False, 'max_chunk_s': 20.0, 'language': 'English'} | NVIDIA GeForce RTX 4090, torch 2.14.1+cu126, commit f085f0a, 2026-11-20 | {'dataset': 'librispeech', 'split': 'test_clean', 'n': 200, 'seed': 0, 'wer_normalizer': 'whisper-english'}
- rumblex_vllm_librispeech: {'model': 'Reson8/Rumble-2-Flash', 'device': 'cuda:0', 'compute_type': 'bfloat16', 'batch_size': 8, 'dtype': 'torch.bfloat16', 'digits': False, 'max_chunk_s': 20.0, 'backend': 'vllm', 'base_url': 'http://localhost:8000/v1', 'concurrency': 32, 'language': 'English'} | NVIDIA GeForce RTX 4090, torch 2.14.1+cu126, commit f085f0a, 2026-11-20 | {'dataset': 'librispeech', 'split': 'test_clean', 'n': 200, 'seed': 0, 'wer_normalizer': 'whisper-english'}
- whisperx_librispeech: {'model': 'large-v3', 'device': 'cuda', 'compute_type': 'float16', 'batch_size': 16, 'whisperx': '3.8.6', 'language': 'English'} | NVIDIA GeForce RTX 4090, torch 2.8.0+cu128, commit f085f0a, 2026-11-20 | {'dataset': 'librispeech', 'split': 'test_clean', 'n': 200, 'seed': 0, 'wer_normalizer': 'whisper-english'}

## ami
| system | WER | start MAE | starts ≤ 100 ms | end MAE | ends ≤ 100 ms | long file speed | peak GPU |
|---|---|---|---|---|---|---|---|
| rumblex_ami | 15.0% | n/a ms | n/a | n/a ms | n/a | 158.4x | 8212 MB |
| whisperx_ami | 17.2% | n/a ms | n/a | n/a ms | n/a | 58.7x | 10024 MB |

- rumblex_ami: {'model': 'Reson8/Rumble-2-Flash', 'device': 'cuda:0', 'compute_type': 'bfloat16', 'batch_size': 32, 'dtype': 'torch.bfloat16', 'digits': False, 'max_chunk_s': 20.0, 'language': 'English'} | NVIDIA GeForce RTX 4090, torch 2.14.1+cu126, commit f085f0a, 2026-11-20 | {'dataset': 'ami', 'split': 'test', 'n': 500, 'seed': 0, 'wer_normalizer': 'whisper-english'}
- whisperx_ami: {'model': 'large-v3', 'device': 'cuda', 'compute_type': 'float16', 'batch_size': 16, 'whisperx': '3.8.6', 'language': 'English'} | NVIDIA GeForce RTX 4090, torch 2.8.0+cu128, commit f085f0a, 2026-11-20 | {'dataset': 'ami', 'split': 'test', 'n': 500, 'seed': 0, 'wer_normalizer': 'whisper-english'}

## earnings22
| system | WER | start MAE | starts ≤ 100 ms | end MAE | ends ≤ 100 ms | long file speed | peak GPU |
|---|---|---|---|---|---|---|---|
| rumblex_earnings22 | 10.0% | n/a ms | n/a | n/a ms | n/a | 161.7x | 9030 MB |
| whisperx_earnings22 | 10.6% | n/a ms | n/a | n/a ms | n/a | 63.5x | 10434 MB |

- rumblex_earnings22: {'model': 'Reson8/Rumble-2-Flash', 'device': 'cuda:0', 'compute_type': 'bfloat16', 'batch_size': 32, 'dtype': 'torch.bfloat16', 'digits': False, 'max_chunk_s': 20.0, 'language': 'English'} | NVIDIA GeForce RTX 4090, torch 2.14.1+cu126, commit f085f0a, 2026-11-20 | {'dataset': 'earnings22', 'split': 'test', 'n': 500, 'seed': 0, 'wer_normalizer': 'whisper-english'}
- whisperx_earnings22: {'model': 'large-v3', 'device': 'cuda', 'compute_type': 'float16', 'batch_size': 16, 'whisperx': '3.8.6', 'language': 'English'} | NVIDIA GeForce RTX 4090, torch 2.8.0+cu128, commit f085f0a, 2026-11-20 | {'dataset': 'earnings22', 'split': 'test', 'n': 500, 'seed': 0, 'wer_normalizer': 'whisper-english'}

## fleurs_nl
| system | WER | start MAE | starts ≤ 100 ms | end MAE | ends ≤ 100 ms | long file speed | peak GPU |
|---|---|---|---|---|---|---|---|
| rumblex_fleurs_nl | 10.1% | n/a ms | n/a | n/a ms | n/a | 139.5x | 8990 MB |
| whisperx_fleurs_nl | 7.0% | n/a ms | n/a | n/a ms | n/a | 48.2x | 12518 MB |

- rumblex_fleurs_nl: {'model': 'Reson8/Rumble-2-Flash', 'device': 'cuda:0', 'compute_type': 'bfloat16', 'batch_size': 32, 'dtype': 'torch.bfloat16', 'digits': False, 'max_chunk_s': 20.0, 'language': 'Dutch'} | NVIDIA GeForce RTX 4090, torch 2.14.1+cu126, commit f085f0a, 2026-11-20 | {'dataset': 'fleurs_nl', 'split': 'test', 'n': 364, 'seed': 0, 'wer_normalizer': 'whisper-basic'}
- whisperx_fleurs_nl: {'model': 'large-v3', 'device': 'cuda', 'compute_type': 'float16', 'batch_size': 16, 'whisperx': '3.8.6', 'language': 'Dutch'} | NVIDIA GeForce RTX 4090, torch 2.8.0+cu128, commit f085f0a, 2026-11-20 | {'dataset': 'fleurs_nl', 'split': 'test', 'n': 364, 'seed': 0, 'wer_normalizer': 'whisper-basic'}

