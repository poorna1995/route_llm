# Offline corpus — unsupervised LLM routing

## Files
- `config.yaml` — research-driven roles & sizes
- `build.py` — builders + leakage checks
- `processed/` — `queries_*.jsonl`, `split_ids.json`, `manifest.json`

## Build
```bash
./run.sh --smoke
./run.sh
```

## Roles ↔ research use
| role | use |
|------|-----|
| fit | signal analysis + learn λ |
| calib | multi-task thresholds (escalate to 70B) |
| eval | paper metrics only |

## Model pool
| experiment | weak | strong |
|------------|------|--------|
| **primary** (main tables) | Llama-3.1-8B-Instruct | Llama-3.1-70B-Instruct |
| **ablation** (one row) | Qwen2.5-7B-Instruct | Llama-3.1-70B-Instruct |

Join later stages on `query_id` (+ `model_id`).
