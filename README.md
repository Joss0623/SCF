# SCF: Source-Compositional Forecasting for Multimodal Time Series

SCF constructs Numerical, Prior, Report, and Search forecast candidates and combines them with availability-aware, sample–horizon factorized routing.

## Code

- `scf/model.py`: source-specific forecast construction and structured router. `SCF` aliases the original implementation class, preserving its initialization and parameter names.
- `scf/train.py`: training, validation selection, and evaluation entry point.
- `models/` and `layers/`: numerical backbones and their required layers.
- `data_provider/` and `utils/`: chronological splits, normalization, causal calendar priors, and model utilities.
- `scripts/run_scf.sh`: the single main launcher for all nine domains.

This release contains source code only. Datasets, text embeddings, checkpoints, experimental outputs, and plotting scripts are not included.

## Installation

The code was checked with Python 3.10 and the versions in `requirements.txt`. Install PyTorch for your hardware, then:

```bash
pip install -r requirements.txt
python -m scf.train --help
```

## Inputs

Autoformer uses CUDA in its upstream autocorrelation layer.

The numerical and text sources originate from [Time-MMD / MM-TSFlib](https://github.com/AdityaLab/MM-TSFlib). Supply the prepared CSV files and precomputed frozen text embeddings externally.

Each CSV uses the original Time-MMD fields: `date`, `OT`, `prior_history_avg`, `start_date`, `end_date`, and `Final_Search_4`. Historical priors are recomputed causally from observations before each forecast origin. Normalization is fitted on the training split. The default chronological train/validation/test split is 70/10/20.

For each domain, place `text_token_embeddings.npy` with shape `[timestamps, source slots, embedding dimension]` and `text_token_mask.npy` with shape `[timestamps, source slots]` under `$EMBED_ROOT/DOMAIN/`. Both arrays must follow the CSV row order. Report slots precede Search slots; the per-source slot count is four, or two for SocialGood. The training entry point reads the representations at the last observed timestamp of each window.

The launcher lists the CSV filenames and domain-specific forecast horizons. `DATA_ROOT` points to the parent of the original domain folders; `EMBED_ROOT` points to the parent of the domain embedding folders.

## Run

```bash
DATA_ROOT=/path/to/prepared/time-mmd \
EMBED_ROOT=/path/to/frozen/embeddings \
bash scripts/run_scf.sh HealthUS 0
```

Supported domains: `Agriculture`, `Climate`, `Economy`, `Energy`, `Environment`, `HealthUS`, `Security`, `SocialGood`, and `Traffic`.

The main launcher uses seeds 2024, 2025, and 2026, a ten-epoch expert initialization budget, a five-epoch fine-tuning/continuation budget, and validation early stopping with patience three. It runs the Numerical-only, fixed-fusion, numerical continuation, and SCF modes. The internal SCF mode is `prior_source_router_trend`. For a single horizon, set `HORIZONS=24`; `SEEDS`, `PYTHON`, and `OUT_ROOT` are also configurable. Newly generated outputs are written under `outputs/` and ignored by Git.

## Acknowledgments and license

The forecasting backbones and utility layers build on [MM-TSFlib](https://github.com/AdityaLab/MM-TSFlib) and the [Time-Series-Library](https://github.com/thuml/Time-Series-Library). Their MIT license and copyright notice are retained in `LICENSE`.
