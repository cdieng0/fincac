# Changelog — FINCAC40

Format basé sur [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased] — Audit reproductibilité (2026-09-04)

### Added
- Documentation reproductibilité complète (`docs/`, `README.md`)
- `INCIDENT_NOTE.md` — incident dérive de prompt ORPO
- `docs/large_files_report.md` — politique gros fichiers
- `docs/reproduction_matrix.md` — traçabilité papier → script
- `.env.example` — template variables d'environnement
- `tests/` — tests counts corpus, Gold, prompts
- `CITATION.cff`

### Changed
- `.gitignore` — exclusion scientifique (secrets, venv, gros CSV)
- `benchmark_temporal_drift.py` — clés API via variables d'environnement
- `build_orpo_pairs.py` — clé API via variable d'environnement

### Security
- **CRITIQUE :** suppression de clés API hardcodées dans le code source
- **Action requise :** rotation des clés exposées avant push public

### Known gaps documented
- Script linear probing absent du dépôt
- Checkpoints ORPO Run 1 non conservés localement
- `requirements.txt` incomplet pour pipeline complet (en cours)

## [1.0.0] — Release papier (2026-08)

### Added
- Corpus 313 898 paragraphes (`main_amf.py`)
- Gold Standard 140 paragraphes
- Benchmark 4 modèles zero-shot / 3-shot
- ORPO Run 1 (collapse) et Run 2 (échec partiel)
- Publication Hugging Face dataset + modèle ORPO
