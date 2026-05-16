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
