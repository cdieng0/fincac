# FinCAC40 — Temporal Robustness of LLMs on French Regulatory Text

Research code accompanying the preprint *FinCAC40 : un corpus réglementaire français
(2010–2026) pour l'évaluation de la robustesse temporelle des LLM en classification de
durabilité*.

This repository contains the complete pipeline: corpus extraction from AMF filings,
unbiased stratified sampling, expert annotation under the ESRS taxonomy, multi-LLM
evaluation, linear probing, and a documented failure of preference optimization.

[![Dataset](https://img.shields.io/badge/🤗%20Dataset-FinCAC40-yellow)](https://huggingface.co/datasets/YOUR_USERNAME/FinCAC40)
[![Model](https://img.shields.io/badge/🤗%20Model-Mistral--7B--ORPO--CSRD-yellow)](https://huggingface.co/YOUR_USERNAME/Mistral-7B-ORPO-CSRD)
[![Demo](https://img.shields.io/badge/🤗%20Space-collapse%20demo-blue)](https://huggingface.co/spaces/YOUR_USERNAME/FinCAC40-collapse-demo)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

---

## What this project found

The CSRD (2023) standardised the vocabulary of sustainability disclosure in Europe. Older
filings describe the same issues in entirely different terms. We asked whether LLMs, trained
mostly on recent data, systematically miss sustainability content expressed in pre-CSRD
language.

**The behavioural result is heterogeneous.** Two frontier models support the hypothesis
(GPT-4o ΔFNR = +0.54, Mistral-Large +0.45), while Mistral-7B contradicts it directionally.
With 3 to 22 relevant paragraphs per temporal window, no statistical test is conclusive — we
report an association, not a causal effect.

**The most solid result is a negative one.** Attempting to correct the suspected bias with
ORPO produced a silent collapse:

| | Accuracy | Malformed output | Global FPR |
|---|---|---|---|
| Base model (3-shot) | **70.8%** | **0%** | — |
| After ORPO (run 1) | 16.4% | — | — |
| After ORPO (run 2) | 35.7% | **30.7%** | **64.9%** |

Every standard training signal said the run had succeeded — decreasing loss, 100% preference
accuracy, and a False Negative Rate that reached its ideal value of zero. The FNR hit zero
because the model stopped predicting `none` altogether, which nulls the metric mechanically
without correcting anything. A practitioner monitoring the usual telemetry would have shipped
a model that had lost 54 accuracy points.

---

## Pipeline

```
AMF portal (524,589 entries)
        │  src/extraction/
        ▼
313,898 paragraphs, 23,144 documents, 2010–2026
        │  src/sampling/          stratified, no filtering on the target variable
        ▼
Gold Standard: 140 paragraphs, ESRS-annotated
        │
        ├─ src/evaluation/        4 LLMs × 2 regimes → FNR by epoch
        ├─ src/probing/           linear probe, 33 layers of Mistral-7B
        └─ src/orpo/              preference pairs → QLoRA+ORPO → collapse
                                          │
                                          ▼  src/publishing/
                            🤗 dataset · model · demo
```

Full step-by-step documentation, with commands, inputs, outputs and cost per stage:
**[`docs/PIPELINE.md`](docs/PIPELINE.md)**

---

## Repository layout

| Path | Contents |
|---|---|
| `src/extraction/` | AMF corpus extraction, PDF paragraph reconstruction |
| `src/sampling/` | Three-axis stratified sampling (epoch × doc type × CSRD density) |
| `src/annotation/` | ESRS taxonomy, Gold export, LLM-as-judge annotation, Cohen's κ |
| `src/evaluation/` | Multi-LLM benchmark, metrics, bootstrap confidence intervals |
| `src/probing/` | Layer-wise linear probing of temporal signal |
| `src/orpo/` | Preference-pair construction, QLoRA+ORPO training, failure diagnosis |
| `src/publishing/` | Hugging Face dataset and model publication |
| `space/` | Gradio demo comparing base vs collapsed model |
| `data/gold/` | Gold Standard, 140 annotated paragraphs |
| `data/results/` | Evaluation metrics, reports, per-example predictions |
| `paper/` | LaTeX source of the preprint |
| `archive/` | Abandoned press-scraping source, kept for transparency |

Large artifacts — the 87 MB corpus and the model weights — live on Hugging Face, not here.

---

## Quick start

```bash
git clone https://github.com/YOUR_USERNAME/finsent.git
cd finsent
pip install -r requirements.txt
```

API keys are read from the environment. Never hardcode them:

```bash
export MISTRAL_API_KEY="..."
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
```

The dataset is easier to obtain from the Hub than to re-extract:

```python
from datasets import load_dataset
corpus = load_dataset("YOUR_USERNAME/FinCAC40", "corpus", split="train")
gold   = load_dataset("YOUR_USERNAME/FinCAC40", "gold",   split="test")
```

---

## Reproducibility

Every stage uses `seed = 42` and `temperature = 0`. Bootstrap confidence intervals use
B = 2000 resamples. The source AMF export is certified by its SHA256 hash, recorded in the
extraction run metadata.

Note that fixing temperature and seed does **not** guarantee determinism on proprietary APIs:
the served model version and serving infrastructure can change without notice. We therefore
log the resolved model identifier for every call and archive the raw responses — that is the
practical reproducibility guarantee, not a theoretical one.

The Gold Standard is now public, which means it may enter the pretraining corpora of future
models. Evaluations run after this release should account for that contamination.

---

## Limitations

Stated plainly, and discussed at length in the paper:

- **Semantic equivalence between epochs is not established.** The FNR difference across epochs
  may partly reflect confounds — document length, type, issuer, topic frequency — rather than
  purely lexical drift.
- **The Gold Standard is small** (n = 140) and imbalanced: five ESRS categories are absent,
  two have a single example.
- **Single annotator**, no inter-annotator agreement measure in this release.
- **Probing is correlational.** A linear probe shows decodability, not that the model uses that
  signal for its decisions.
- **Survivorship bias**: only current CAC 40 members are represented.

---

## Citation

```bibtex
@misc{dieng2026fincac40,
  title  = {FinCAC40 : un corpus réglementaire français (2010--2026) pour l'évaluation
            de la robustesse temporelle des LLM en classification de durabilité},
  author = {Dieng, Cheikh Ibra},
  year   = {2026},
  note   = {Preprint}
}
```

## License

Code released under Apache 2.0. The corpus derives from AMF filings published under the
French **Licence Ouverte / Open Licence 2.0 (Etalab)** and is redistributed under the same terms.
