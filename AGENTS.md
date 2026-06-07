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
