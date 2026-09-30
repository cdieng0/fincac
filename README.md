# FinCAC40 — Temporal Robustness of LLMs on French Regulatory Text

Code, data pointers and artefacts for the preprint
**[Regulatory Semantic Drift: When Preference-based Correction Fails Silently — FINCAC40, a Sixteen-year AMF Corpus (2010–2026) for Evaluating the Temporal Robustness of LLMs in ESG Classification](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7438503)**
(Cheikh Ibra Dieng, SSRN, 2026).

The repository covers the full pipeline: corpus extraction from AMF filings, stratified
sampling without filtering on the target variable, annotation under the ESRS taxonomy,
a four-LLM evaluation, linear probing of Mistral-7B, and a documented failure of preference
optimization.

[![Paper](https://img.shields.io/badge/Paper-SSRN%207438503-b31b1b)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7438503)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-FinCAC40-yellow)](https://huggingface.co/datasets/CID99/FinCAC40)
[![Model](https://img.shields.io/badge/🤗%20Model-Mistral--7B--ORPO--CSRD-yellow)](https://huggingface.co/CID99/Mistral-7B-ORPO-CSRD)
[![Demo](https://img.shields.io/badge/🤗%20Space-collapse%20demo-blue)](https://huggingface.co/spaces/CID99/FinCAC40-collapse-demo)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

| Resource | Where |
|---|---|
| Preprint | [SSRN 7438503](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7438503) |
| Corpus and Gold Standard | [huggingface.co/datasets/CID99/FinCAC40](https://huggingface.co/datasets/CID99/FinCAC40) |
| ORPO adapter and its training artefacts | [huggingface.co/CID99/Mistral-7B-ORPO-CSRD](https://huggingface.co/CID99/Mistral-7B-ORPO-CSRD) |
| Interactive demo | [huggingface.co/spaces/CID99/FinCAC40-collapse-demo](https://huggingface.co/spaces/CID99/FinCAC40-collapse-demo) |
| Code, protocol, run artefacts | this repository |

---

## What this project found

The CSRD (2023) standardised the vocabulary of sustainability disclosure in Europe. Older
filings describe the same issues in entirely different terms. We asked whether LLMs, trained
mostly on recent data, systematically miss sustainability content expressed in pre-CSRD
language.

**The behavioural result is heterogeneous.** Two frontier models support the hypothesis
(GPT-4o ΔFNR = +0.54, Mistral-Large +0.45 in zero-shot), while Mistral-7B contradicts it
directionally (−0.07). With 3 to 22 relevant paragraphs per temporal window, no statistical
test is conclusive — we report an association, not a causal effect.

**The most solid result is a negative one.** Attempting to correct the suspected bias with
ORPO produced a silent collapse:

| | Accuracy | Malformed JSON | `none` paragraphs given a CSRD label |
|---|---|---|---|
| Base model (3-shot) | **70.8%** | **0%** | not measured¹ |
| After ORPO — run 1 | 16.4% | not archived² | not archived² |
| After ORPO — run 2 | 35.7% | **30.7%** | **36.1%** (35 / 97)³ |

¹ The false-positive rate was added to the evaluation protocol after run 1.
² Run 1's per-example predictions were not preserved; its accuracy was recorded by the
evaluation run itself ([`data/results/orpo_run1/`](data/results/orpo_run1/)).
³ A further 28 `none` paragraphs got malformed output; the paper's FPR of 64.9% counts both.
Recomputed from the archived predictions ([`data/results/orpo_run2/`](data/results/orpo_run2/)).

Every standard signal said run 1 had succeeded — decreasing loss, preference accuracy reaching
100%, and a false-negative rate that reached its ideal value of zero. The FNR hit zero because
the model no longer answered `none`, which nulls the metric mechanically without correcting
anything. A practitioner monitoring the usual telemetry would have shipped
a model that had lost 54 accuracy points.

---

## Pipeline

```
AMF export, info-financiere.gouv.fr (524,589 entries)
        │  src/extraction/
        ▼
313,898 paragraphs from 23,144 documents, 2010–2026
        │  src/sampling/          era × document type × keyword-density quartile
        ▼
14,974-paragraph pool ──────────► src/orpo/  preference pairs (Gold excluded)
                                             → QLoRA + ORPO → collapse
Gold Standard: 140 ESRS-annotated paragraphs (selection: docs/data.md)
        ├─ src/evaluation/        4 LLMs × zero-shot / 3-shot → FNR by era
        ├─ src/probing/           linear probe, 33 layers of Mistral-7B
        └─ src/orpo/              evaluation of the ORPO adapters
                                          │
                                          ▼  src/publishing/
                            🤗 dataset · model · demo
```

Step-by-step documentation with the exact command for each stage:
**[`docs/PIPELINE.md`](docs/PIPELINE.md)**.

---

## Repository layout

| Path | Contents |
|---|---|
| `src/extraction/` | AMF corpus extraction, PDF paragraph reconstruction |
| `src/sampling/` | Three-axis stratified sampling (era × document type × CSRD-density quartile) |
| `src/annotation/` | ESRS taxonomy, Gold annotation workbook; LLM annotation and agreement tools (not used for the reported results) |
| `src/evaluation/` | Four-LLM benchmark, shared prompt, metrics with bootstrap intervals |
| `src/probing/` | Layer-wise linear probing of the temporal signal |
| `src/orpo/` | Preference-pair construction, rebalancing, QLoRA + ORPO training, evaluation, failure diagnostics |
| `src/publishing/` | Hugging Face publication, and retrieval of the Gold Standard from the Hub |
| `space/` | Gradio demo comparing the base and the collapsed model |
| `cards/` | Dataset and model cards published on Hugging Face |
| `data/results/` | Run artefacts: probing results and figure, ORPO run configurations, logs, reports and predictions |
| `prompts/` | The exact system prompts, zero-shot and 3-shot |
| `docs/` | [Pipeline](docs/PIPELINE.md), [reproducibility](docs/reproducibility.md), [data](docs/data.md), [annotation protocol](docs/annotation_protocol.md), [prompts](docs/prompts.md), [ORPO incident note](docs/INCIDENT_NOTE.md), [Hugging Face](docs/huggingface.md), [known discrepancies with the preprint](docs/known_discrepancies.md) |
| `paper/` | The preprint |
| `tests/` | Prompt consistency, archived results reproduced against the paper, Gold and corpus checks, documentation links |

The corpus (~88 MB) and the model weights live on Hugging Face, not in this repository.

---

## Quick start

```bash
git clone https://github.com/cdieng0/fincac.git
cd fincac
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt           # add requirements-gpu.txt for probing and ORPO

python src/publishing/fetch_gold_from_hf.py   # Gold Standard → data/
python -m pytest                              # no API key, no GPU
```

API keys are read from the environment, never from the code. The variables are listed in
[`.env.example`](.env.example):

```bash
export MISTRAL_API_KEY="..."         # PowerShell: $env:MISTRAL_API_KEY = "..."
export OPENAI_API_KEY="..."          # optional, GPT-4o in the benchmark
export ANTHROPIC_API_KEY="..."       # optional, Claude in the benchmark
```

The corpus is easier to obtain from the Hub than to re-extract:

```python
from datasets import load_dataset
corpus = load_dataset("CID99/FinCAC40", "corpus", split="train")   # 313,608 paragraphs
gold   = load_dataset("CID99/FinCAC40", "gold",   split="test")    # 140 paragraphs
```

---

## Reproducibility

Each paper result is traced to its script and archived artefact in
[`docs/reproducibility.md`](docs/reproducibility.md), and every command is in
[`docs/PIPELINE.md`](docs/PIPELINE.md). The test suite recomputes the published ORPO run 2
report from its per-example predictions and checks the probing and run 1 artefacts against the
paper; the differences found between the preprint and the released code and data are listed in
[`docs/known_discrepancies.md`](docs/known_discrepancies.md).

Every stage uses `seed = 42` and `temperature = 0`. Bootstrap confidence intervals use
B = 2000 resamples. The source AMF export is identified by its SHA-256 hash, recorded in the
extraction run metadata.

Fixing temperature and seed does **not** guarantee determinism on proprietary APIs: the served
model version and serving infrastructure can change without notice. The benchmark therefore
logs the resolved model identifier for every call.

The prompt used by the benchmark and by the whole ORPO chain is checked byte for byte by
[`tests/test_prompts.py`](tests/test_prompts.py) — the guard against the prompt drift described
in [`docs/INCIDENT_NOTE.md`](docs/INCIDENT_NOTE.md).

The Gold Standard is public, so it may enter the pretraining corpora of future models.
Evaluations run after this release should account for that contamination.

---

## Limitations

Stated plainly, and discussed at length in the paper:

- **Semantic equivalence between eras is not established.** The FNR difference across eras
  may partly reflect confounds — document length, type, issuer, topic frequency — rather than
  purely lexical drift.
- **The Gold Standard is small** (n = 140) and imbalanced: five ESRS categories are absent,
  three have a single example. It was selected by pre-computed semantic type rather than drawn
  proportionally from the corpus ([`docs/data.md`](docs/data.md#gold-standard)).
- **Single annotator**, with no inter-annotator agreement measure in this release.
- **Probing is correlational.** A linear probe shows decodability, not that the model uses that
  signal for its decisions.
- **Survivorship bias**: only current CAC 40 members are represented.

---

## Citation

```bibtex
@misc{dieng2026fincac40,
  title        = {Regulatory Semantic Drift: When Preference-based Correction Fails Silently.
                  {FINCAC40}, a Sixteen-year {AMF} Corpus (2010--2026) for Evaluating the
                  Temporal Robustness of {LLMs} in {ESG} Classification},
  author       = {Dieng, Cheikh Ibra},
  year         = {2026},
  howpublished = {SSRN preprint},
  url          = {https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7438503}
}
```

GitHub's "Cite this repository" button reads [`CITATION.cff`](CITATION.cff).

## License

Code released under the [Apache License 2.0](LICENSE). The corpus derives from AMF filings
published under the French **Licence Ouverte / Open Licence 2.0 (Etalab)** and is redistributed
under the same terms.
