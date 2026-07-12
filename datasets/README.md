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

Join later stages on `query_id` (+ `model_id`).
