# Reproducibility

What can be checked, at what cost, and where each number of the paper comes from.

## Setup

```bash
git clone https://github.com/cdieng0/fincac.git
cd fincac
python -m venv .venv
source .venv/bin/activate                  # Windows: .venv\Scripts\activate
pip install -r requirements.txt            # CPU stages, tests, API clients
pip install -r requirements-gpu.txt        # only for probing and ORPO (NVIDIA GPU)
python src/publishing/fetch_gold_from_hf.py
python -m pytest
```

API keys are read from the environment (see [`.env.example`](../.env.example)); the scripts do
not load a `.env` file themselves. Python 3.10 or later.

## Four levels of reproduction

| Level | What you do | Cost |
|---|---|---|
| 1. Check the archived artefacts | `python -m pytest` — recomputes the Run 2 report from its predictions and checks the probing and Run 1 artefacts against the paper | seconds, CPU |
| 2. Recompute from predictions | `evaluate_orpo.py --from-predictions`, `compute_metrics.py`, `compute_missing_kappa.py` | seconds, CPU |
| 3. Re-run model calls | benchmark, pair collection | API credits |
| 4. Re-run from scratch | extraction, probing, ORPO training | hours, GPU |

## Paper result → code → artefact

| Paper | Script | Archived artefact | Status |
|---|---|---|---|
| Table 1 — corpus volumes | `process_amf_extraction.py` | Hub corpus (313,608 rows) | Re-runnable; the AMF export changes over time |
| Table 2 — eras, Gold counts | `publish_dataset_to_hf.py` (`epoch`) | Hub `gold` config | Checked by `tests/test_gold_standard.py` |
| Stratified pool (14,974) | `build_annotation_sample_rigorous.py` | not archived | Re-runnable from the raw extraction or the Hub corpus |
| Gold selection (140) | not in this repository | Hub `gold` config | See [data.md](data.md#how-the-140-were-selected) |
| Table 3 — FNR by era, 4 models | `benchmark_temporal_drift.py`, `compute_metrics.py` | predictions not yet archived | Re-runnable with API keys |
| Table 4 — base policy κ | `compute_missing_kappa.py` | predictions not yet archived | Re-runnable with API keys |
| Section 4.2, Figure 2 — probing | `probe_temporal_signal.py` | [`data/results/probing/`](../data/results/probing/) | Checked by `tests/test_results.py`; re-runnable on GPU |
| Table 4 — ORPO Run 1 | `train_orpo_mistral7b.py`, `evaluate_orpo.py` | [`data/results/orpo_run1/`](../data/results/orpo_run1/) (configuration, log, report) | Checked; pairs and predictions not preserved |
| Table 4 — ORPO Run 2 | idem | [`data/results/orpo_run2/`](../data/results/orpo_run2/) + [model repo](https://huggingface.co/CID99/Mistral-7B-ORPO-CSRD) | Report recomputed from predictions by `tests/test_results.py` |

## Fixed parameters

| Parameter | Value |
|---|---|
| Random seed | 42 everywhere |
| Temperature | 0 |
| Bootstrap | B = 2000 (global), 500 (per era) |
| Few-shot | 3 examples taken from the Gold and removed from the test set ([prompts.md](prompts.md)) |
| ORPO | β = 0.1, LoRA r = 16 / α = 32, QLoRA NF4, effective batch 16 |

## Limits of reproduction

- **Proprietary models change.** Temperature 0 and a fixed seed do not guarantee identical
  answers from GPT-4o, Claude or Mistral-Large; the benchmark stores the model identifier each
  call resolved to.
- **Two sets of predictions are not yet archived here:** the four-model benchmark outputs
  (zero-shot and 3-shot) and Run 1's per-example ORPO predictions. The first can be regenerated
  with API keys; the second cannot.
- **Human annotation is a judgement.** A second annotator would not reproduce every label;
  no agreement measure exists yet.
- **Known differences between the preprint and this repository** are listed in
  [known_discrepancies.md](known_discrepancies.md).
