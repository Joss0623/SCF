#!/usr/bin/env bash
set -euo pipefail

# Main SCF run: nine domains, four horizons, three seeds.
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
DOMAIN=${1:?Usage: bash scripts/run_scf.sh DOMAIN [GPU]}
GPU=${2:-0}
PYTHON=${PYTHON:-python}
SEEDS=${SEEDS:-2024,2025,2026}
DATA_ROOT=${DATA_ROOT:-$ROOT/data}
EMBED_ROOT=${EMBED_ROOT:-$ROOT/data/embeddings}
OUT_ROOT=${OUT_ROOT:-$ROOT/outputs}
source_tokens=4
case "$DOMAIN" in
  HealthUS)
    folder=Public_Health; filename=US_FLURATIO_Week.csv
    freq=w; seq_len=24; label_len=12; horizons=(12 24 36 48) ;;
  Energy)
    folder=Energy; filename=US_GasolinePrice_Week.csv
    freq=w; seq_len=24; label_len=12; horizons=(12 24 36 48) ;;
  Traffic)
    folder=Traffic; filename=US_VMT_Month.csv
    freq=m; seq_len=8; label_len=4; horizons=(6 8 10 12) ;;
  Climate)
    folder=Climate; filename=US_precipitation_month.csv
    freq=m; seq_len=8; label_len=4; horizons=(6 8 10 12) ;;
  Economy)
    folder=Economy; filename=US_TradeBalance_Month_processed.csv
    freq=m; seq_len=8; label_len=4; horizons=(6 8 10 12) ;;
  Environment)
    folder=Environment; filename=Environment_processed.csv
    freq=d; seq_len=96; label_len=48; horizons=(48 96 192 336) ;;
  Agriculture)
    folder=Algriculture; filename=US_RetailBroilerComposite_Month.csv
    freq=m; seq_len=8; label_len=4; horizons=(6 8 10 12) ;;
  Security)
    folder=Security; filename=US_FEMAGrant_Month.csv
    freq=m; seq_len=8; label_len=4; horizons=(6 8 10 12) ;;
  SocialGood)
    folder=SocialGood; filename=Unadj_UnemploymentRate_ALL_processed_clean.csv
    freq=m; seq_len=8; label_len=4; horizons=(6 8 10 12); source_tokens=2 ;;
  *) echo "Unsupported domain: $DOMAIN" >&2; exit 2 ;;
esac
if [[ -n "${HORIZONS:-}" ]]; then IFS=, read -ra horizons <<< "$HORIZONS"; fi
for pred_len in "${horizons[@]}"; do
  CUDA_VISIBLE_DEVICES="$GPU" CUBLAS_WORKSPACE_CONFIG=:4096:8 "$PYTHON" -m scf.train \
    --root_path "$DATA_ROOT/$folder" --data_path "$filename" --freq "$freq" \
    --seq_len "$seq_len" --label_len "$label_len" --pred_len "$pred_len" \
    --source_tokens "$source_tokens" --text_encoder precomputed \
    --text_embedding_path "$EMBED_ROOT/$DOMAIN/text_token_embeddings.npy" \
    --text_mask_path "$EMBED_ROOT/$DOMAIN/text_token_mask.npy" \
    --modes off,fixed,off_continue,prior_source_router_trend \
    --seeds "$SEEDS" --base_epochs 10 --continue_epochs 5 --patience 3 \
    --calibrate_gate_prior --skip_test_controls \
    --out_dir "$OUT_ROOT/${DOMAIN}_h${pred_len}"
done
