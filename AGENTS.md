Yes: Start with the highest-signal sources first: README*, root manifests, workspace config, lockfiles.
Yes: Read repo-local OpenCode config (opencode.json) to understand wiring, entrypoints, and package boundaries before exploring code.
Yes: Inspect CI workflows and pre-commit/task runner config to learn exact command sequences (lint, typecheck, test) and their order.
Yes: Review existing instruction files (AGENTS.md, CLAUDE.md, .cursor/rules/, .cursorrules, .github/copilot-instructions.md) to align with established conventions.
Yes: Prefer executable sources of truth over prose; when docs conflict with scripts or configs, trust the executable source and prune outdated guidance.
Yes: If architecture is unclear after reading config/docs, inspect a small representative set of code files to identify real entrypoints, package boundaries, and execution flow.
Yes: Preserve repo conventions and boundaries; avoid introducing new flows or tools unless explicitly requested or necessary.
Yes: Use executable search tools (Glob and Grep via rg) and parallelize file reads to speed up context gathering.
Yes: When editing or adding guidance, keep changes minimal and targeted; prefer patches over large rewrites.
Yes: If a plan or decision is ambiguous, ask a clarifying question before implementing changes; use the designated question tool for a concise batch if needed.

## Statistical tests module (added 2026-05-16)

- `scripts/statistical_tests.py` — Reusable module with Mann-Kendall trend test, Sen's slope estimator, Pettitt change-point detection, and seasonal MK.
- `scripts/inject_stat_tests.py` — Injects statistical analysis cells into notebooks.
- To add stat tests to all notebooks: `python scripts/inject_stat_tests.py` from project root.
- Notebooks modified: Rain_seasons, Hourly_rain_analysis, Intensity_pluja, Evolucio_temperatures_nocturnes.

## Drought analysis module (added 2026-06-04)

- `scripts/drought_analysis.py` — Reusable module for drought analysis:
  - `thornthwaite_pet()`: Calculate PET using Thornthwaite (1948) method
  - `compute_drought_indices()`: Compute SPI and SPEI at multiple timescales
  - `detect_drought_events()`: Identify individual drought events
  - `drought_statistics()`: Compute annual drought statistics
  - `full_analysis()`: Run complete analysis for one station
- Dependencies: `spei` package (install with `pip install spei`)
- `Seccions/Sequera/Drought_analysis.ipynb` — Interactive notebook with detailed explanations

## Housekeeping fixes (2026-06-05)

- `.gitignore` now excludes `.venv/`, `.env/`, `*.csv`, `Raw Data/`, `Cleaned Data/`
- `scripts/inject_stat_tests.py`: removed hardcoded absolute path, uses `os.path.abspath(__file__)` instead; fixed notebook paths to match actual directory structure (`Seccions/Pluges/`, `Seccions/Temperatures/`)
- `requirements.txt` at project root with all dependencies (dashboard + scripts + notebooks)
- `dashboard/utils.py`: fixed `count_above_threshold` pandas deprecation warning
- `tests/` directory added with:
  - `test_utils.py` — 7 tests for summer/winter/night masks, season assignment, night classification, streaks
  - `test_statistical.py` — 9 tests for Mann-Kendall, Sen's slope, Pettitt, seasonal MK
  - Run with: `python -m pytest tests/` from project root

## ML downscaling: temperature curve prediction (2026-06-07)

- `scripts/prepare_ml_data.py` — Prepares feature matrices for ML:
  - Loads hourly temperature CSVs, extracts 48 target values per complete day
  - Loads daily data with ALL features (not just T_min/T_max/T_avg)
  - Computes engineered features: T_range, T_avg_lag1, sin_doy, cos_doy, month
  - Outputs two datasets:
    - `ml_data_common.npz` — 12 features available at both stations (T_avg, T_max, T_min, humidity_avg, rain_mm, pressure_avg, T_range, T_avg_lag1, sin_doy, cos_doy, month, is_sbd), ~17k days
    - `ml_data_full.npz` — 15 features (common + radiation_avg, avg_wind_kmh, wind_dir, max_wind_kmh), Sbd Nord only, ~6.3k days
- `scripts/generate_ml_notebook.py` — Generates `Seccions/ML/Prediccio_corba_temperatura.ipynb` (52 cells) with full ML pipeline: sinusoidal baseline, KNN, Random Forest, MLP, GridSearchCV, feature ablation

### Best MLP model results (from executed notebook):
| Features | Stations | Best params | RMSE | R² |
|---|---|---|---|---|
| Common (12) | Both | (256,128), lr=0.01, α=0.001 | 1.177°C | 0.9719 |
| Full (15) | Sbd Nord only | (128,64), lr=0.01, α=0.0001 | 1.162°C | 0.9698 |
| Common (12) | Sbd Nord only | (256,128), lr=0.01, α=0.001 | 1.145°C | 0.9708 |

### Synthetic hourly generation for Sbd Centre

- `scripts/generate_sbd_centre_hourly.py` — Main pipeline:
  - Trains MLP (256,128, lr=0.01, α=0.001) on Sbd Nord daily data with 12 Centre-compatible features
  - Predicts 30-min temperature curve for all 5,833 Sbd Centre days
  - Post-processes with np.clip(T_min, T_max) to enforce daily bounds
  - Output: `Cleaned Data/Sbd_Centre_hourly_synthetic.csv` (279,984 rows, columns: datetime, T_30min_C)
  - Validation metrics on Sbd Nord test set: RMSE 1.149°C, R² 0.9684
- `scripts/validate_synthetic.py` — Quality checks: completeness, timestep gaps, bounds violations, monthly means vs Nord

### Key conventions
- Sbd Centre has NO humidity, pressure, or radiation data — only the 12 common features are usable for transfer
- Sbd Nord hourly has `datetime_utc` column; Sbd Centre daily has an `Unnamed: 0` index column (ignore it with `index_col=0`)
- Wind data in Sbd Nord is missing for early years (pre-June 2009); impute with monthly median
- The ML output CSVs (`.npz` + `ml_metadata*.csv`) are intermediate artifacts and can be regenerated by `prepare_ml_data.py`
- `Cleaned Data/validation/` figures are also regenerable; only commit the final synthetic CSV
