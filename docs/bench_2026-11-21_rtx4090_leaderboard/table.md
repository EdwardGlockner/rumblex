| system | WER | start MAE | starts ≤ 100 ms | end MAE | ends ≤ 100 ms | long file speed | peak GPU |
|---|---|---|---|---|---|---|---|
| rumblex_fleurs_nl | 9.2% | n/a | n/a | n/a | n/a | 49.0x | 5576 MB |
| whisperx_fleurs_nl | 7.0% | n/a | n/a | n/a | n/a | 40.7x | 7718 MB |

- rumblex_fleurs_nl: {'model': 'Reson8/Rumble-2-Flash', 'device': 'cuda:0', 'compute_type': 'bfloat16', 'batch_size': 32, 'dtype': 'torch.bfloat16', 'digits': True, 'max_chunk_s': 20.0, 'language': 'Dutch'} | NVIDIA GeForce RTX 4090, torch 2.14.1+cu126, commit bde8a3d, 2026-11-21 | {'dataset': 'fleurs_nl', 'split': 'test', 'n': 364, 'seed': 0, 'wer_normalizer': 'whisper-basic'}
- whisperx_fleurs_nl: {'model': 'large-v3', 'device': 'cuda', 'compute_type': 'float16', 'batch_size': 16, 'whisperx': '3.8.6', 'language': 'Dutch'} | NVIDIA GeForce RTX 4090, torch 2.8.0+cu128, commit bde8a3d, 2026-11-21 | {'dataset': 'fleurs_nl', 'split': 'test', 'n': 364, 'seed': 0, 'wer_normalizer': 'whisper-basic'}
rumblex_fleurs_nl.json       rumblex       saved 9.18%  leaderboard 6.24%
whisperx_fleurs_nl.json      whisperx      saved 7.05%  leaderboard 4.43%
