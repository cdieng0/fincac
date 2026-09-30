# Changelog

Format based on [Keep a Changelog](https://keepachangelog.com/).

## [1.1.0] — 2026-09-30 — Code and reproducibility release

### Added
- The full pipeline code under `src/`, runnable from the repository root as
  `python src/<module>/<script>.py`: stratified sampling, linear probing, κ recomputation,
  ORPO diagnostics, and `fetch_gold_from_hf.py` to download the Gold Standard in the format the
  scripts read.
- Run artefacts in `data/results/`: probing results and Figure 2, ORPO run 1 configuration,
  training log and report, ORPO run 2 report and per-example predictions.
- `prompts/`: the exact zero-shot and 3-shot system prompts.
- `evaluate_orpo.py --from-predictions`: rebuild an evaluation report on CPU; FNR and FPR are
  reported under both conventions for malformed answers.
- Tests: archived results reproduced and checked against the paper, prompt files, annotation
  protocol, internal links.
- Documentation: pipeline, reproducibility, data and Gold selection, annotation protocol,
  prompts, Hugging Face, known discrepancies with the preprint (v1).
- `LICENSE` (Apache 2.0), `CITATION.cff`, `.env.example`, `requirements-gpu.txt`.

### Changed
- Repository layout: cards in `cards/`, notes in `docs/`, preprint in `paper/`.
- The sampler accepts the Hub corpus as well as the raw extraction.
- `build_orpo_pairs.py --help` and `--dry-run` work without an API key; `evaluate_orpo.py`
  runs without the baseline predictions file.
- README and cards: preprint link, Hugging Face links, corrected figures (see
  `docs/known_discrepancies.md`).

### Fixed
- `evaluate_orpo.py` reported a malformed-output rate of 0.0 when it was 30.7%.
- Broken imports and data paths after the move to `src/`.

### Removed
- The committed virtual environment (`.venv/`), empty template READMEs, a backup copy of the
  training script, and the abandoned press-scraping prototype.

### Security
- No API key in the code; keys are read from environment variables. Any key that appeared in
  earlier local versions must be treated as compromised and revoked.

## [1.0.0] — 2026-08 — Paper release

- Corpus of 313,898 paragraphs (313,608 published), Gold Standard of 140 paragraphs.
- Four-model benchmark in zero-shot and 3-shot; linear probing of Mistral-7B.
- ORPO run 1 (collapse) and run 2 (failure); dataset, adapter and demo on Hugging Face.
