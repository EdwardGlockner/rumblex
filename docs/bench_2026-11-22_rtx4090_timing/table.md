## librispeech
| system | WER | start MAE | starts ≤ 100 ms | end MAE | ends ≤ 100 ms | long file speed | peak GPU |
|---|---|---|---|---|---|---|---|
| rumblex_librispeech | 2.2% | 52.6 ms | 88.4% | 57.7 ms | 85.4% | 357.0x | 9360 MB |
| whisperx_librispeech | 1.6% | 74.8 ms | 82.5% | 33.9 ms | 94.1% | 79.8x | 11516 MB |

- rumblex_librispeech: {'model': 'Reson8/Rumble-2-Flash', 'device': 'cuda:0', 'compute_type': 'bfloat16', 'batch_size': 32, 'dtype': 'torch.bfloat16', 'digits': False, 'max_chunk_s': 20.0, 'language': 'English'} | NVIDIA GeForce RTX 4090, torch 2.14.1+cu126, commit 5ae0e4d, 2026-11-22 | {'dataset': 'librispeech', 'split': 'test_clean', 'n': 200, 'seed': 0, 'wer_normalizer': 'whisper-english'}
- whisperx_librispeech: {'model': 'large-v3', 'device': 'cuda', 'compute_type': 'float16', 'batch_size': 16, 'whisperx': '3.8.6', 'language': 'English'} | NVIDIA GeForce RTX 4090, torch 2.8.0+cu128, commit 5ae0e4d, 2026-11-22 | {'dataset': 'librispeech', 'split': 'test_clean', 'n': 200, 'seed': 0, 'wer_normalizer': 'whisper-english'}
