## librispeech
| system | WER | start MAE | starts ≤ 100 ms | end MAE | ends ≤ 100 ms | long file speed | peak GPU |
|---|---|---|---|---|---|---|---|
| parakeet_bfloat16_librispeech | 1.8% | 67.3 ms | 77.9% | 74.8 ms | 74.4% | 82.6x | 3930 MB |
| parakeet_float32_librispeech | 1.8% | 67.4 ms | 77.8% | 74.8 ms | 74.4% | 80.3x | 7454 MB |
| parakeet_nemo_bfloat16_librispeech | 1.7% | 67.5 ms | 77.8% | 74.8 ms | 74.4% | 1364.0x | 7562 MB |
| parakeet_nemo_float32_librispeech | 1.8% | 67.3 ms | 77.8% | 74.7 ms | 74.4% | 930.9x | 11566 MB |
| rumblex_bfloat16_librispeech | 2.2% | 21.0 ms | 97.9% | 31.7 ms | 93.5% | 462.2x | 8964 MB |
| rumblex_float32_librispeech | 2.1% | 20.9 ms | 97.9% | 31.8 ms | 93.5% | 386.5x | 15680 MB |

- parakeet_bfloat16_librispeech: {'model': 'nvidia/parakeet-tdt-0.6b-v3', 'device': 'cuda:0', 'compute_type': 'bfloat16', 'batch_size': 32, 'dtype': 'torch.bfloat16', 'max_chunk_s': 20.0, 'transformers': '5.24.1', 'language': 'English'} | NVIDIA GeForce RTX 4090, torch 2.14.1+cu126, commit 0169240, 2026-11-27 | {'dataset': 'librispeech', 'split': 'test_clean', 'n': 200, 'seed': 0, 'wer_normalizer': 'whisper-english'}
- parakeet_float32_librispeech: {'model': 'nvidia/parakeet-tdt-0.6b-v3', 'device': 'cuda:0', 'compute_type': 'float32', 'batch_size': 32, 'dtype': 'torch.float32', 'max_chunk_s': 20.0, 'transformers': '5.24.1', 'language': 'English'} | NVIDIA GeForce RTX 4090, torch 2.14.1+cu126, commit 0169240, 2026-11-27 | {'dataset': 'librispeech', 'split': 'test_clean', 'n': 200, 'seed': 0, 'wer_normalizer': 'whisper-english'}
- parakeet_nemo_bfloat16_librispeech: {'model': 'nvidia/parakeet-tdt-0.6b-v3', 'device': 'cuda:0', 'compute_type': 'bfloat16', 'batch_size': 32, 'max_chunk_s': 20.0, 'nemo': '3.0.0', 'language': 'English'} | NVIDIA GeForce RTX 4090, torch 2.15.0+cu132, commit 0169240, 2026-11-27 | {'dataset': 'librispeech', 'split': 'test_clean', 'n': 200, 'seed': 0, 'wer_normalizer': 'whisper-english'}
- parakeet_nemo_float32_librispeech: {'model': 'nvidia/parakeet-tdt-0.6b-v3', 'device': 'cuda:0', 'compute_type': 'float32', 'batch_size': 32, 'max_chunk_s': 20.0, 'nemo': '3.0.0', 'language': 'English'} | NVIDIA GeForce RTX 4090, torch 2.15.0+cu132, commit 0169240, 2026-11-27 | {'dataset': 'librispeech', 'split': 'test_clean', 'n': 200, 'seed': 0, 'wer_normalizer': 'whisper-english'}
- rumblex_bfloat16_librispeech: {'model': 'Reson8/Rumble-2-Flash', 'device': 'cuda:0', 'compute_type': 'bfloat16', 'batch_size': 32, 'dtype': 'torch.bfloat16', 'digits': False, 'max_chunk_s': 20.0, 'language': 'English'} | NVIDIA GeForce RTX 4090, torch 2.14.1+cu126, commit 0169240, 2026-11-27 | {'dataset': 'librispeech', 'split': 'test_clean', 'n': 200, 'seed': 0, 'wer_normalizer': 'whisper-english'}
- rumblex_float32_librispeech: {'model': 'Reson8/Rumble-2-Flash', 'device': 'cuda:0', 'compute_type': 'float32', 'batch_size': 32, 'dtype': 'torch.float32', 'digits': False, 'max_chunk_s': 20.0, 'language': 'English'} | NVIDIA GeForce RTX 4090, torch 2.14.1+cu126, commit 0169240, 2026-11-27 | {'dataset': 'librispeech', 'split': 'test_clean', 'n': 200, 'seed': 0, 'wer_normalizer': 'whisper-english'}
