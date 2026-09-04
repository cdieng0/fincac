---
license: etalab-2.0
language:
  - fr
tags:
  - finance
  - regulatory
  - esg
  - csrd
  - esrs
  - sustainability
  - temporal-robustness
  - french
task_categories:
  - text-classification
size_categories:
  - 100K<n<1M
pretty_name: "FinCAC40 — French AMF Regulatory Corpus, CAC 40 Issuers (2010–2026)"
configs:
  - config_name: corpus
    data_files:
      - split: train
        path: corpus/*.parquet
  - config_name: gold
    data_files:
      - split: test
        path: gold/*.parquet
---

# FinCAC40 — French AMF Regulatory Corpus, CAC 40 Issuers (2010–2026)

**FinCAC40** contains 313,898 paragraphs extracted from 23,144 documents filed with the
**Autorité des Marchés Financiers** (AMF, the French financial markets regulator) by CAC 40
issuers between 2010 and 2026, together with a **Gold Standard** of 140 paragraphs annotated
under the **ESRS** taxonomy (CSRD).

The corpus was built to study the **temporal robustness** of language models on French
regulatory text. It spans four successive regulatory regimes, from the absence of structured
ESG obligations (2010–2014) to the entry into force of the CSRD (2023–2026).

> **Note on language:** the corpus content is entirely in **French**. This card is in English
> for accessibility; the accompanying paper is written in French.

---

## ⚠️ Read before use

- **The Gold Standard (140 examples) is a preliminary evaluation set.** It is too small and too
  imbalanced (69.3% `none` class, five ESRS categories entirely absent) to serve as a training
  set or to validate a compliance system in production.
- **Public release means future contamination.** Now that the Gold Standard is public, it may
  appear in the pretraining corpora of future models. Any evaluation carried out after this
  release should account for that.
- **Survivorship bias.** The CAC 40 filter retains issuers that are *currently* index members,
  then collects their documents over 2010–2026. Companies that left the index are not
  represented.
- **This corpus is not regulatory advice.** Annotations reflect the judgment of a single
  annotator and are no substitute for a qualified auditor's opinion.
- **`paragraph_id` is not a primary key.** It is a SHA256 hash of the paragraph *content*, so
  two paragraphs with strictly identical text share the same identifier. Roughly **18% of the
  corpus** consists of repeated paragraphs — disclaimers, legal notices, standardised clauses
  recurring across documents and across years. This is a genuine property of regulatory
  filings, not an extraction defect. Use `row_id` as the unique identifier, and `paragraph_id`
  to detect or remove identical content:

  ```python
  # Corpus deduplicated by content
  unique = corpus.to_pandas().drop_duplicates(subset=["paragraph_id"])
  ```

  Depending on your use case, duplicates should be kept (the same text at two different dates
  is meaningful information for a temporal study) or removed (model training, where they
  over-weight boilerplate).

---

## Structure

### Config `corpus` — 313,898 paragraphs

| Field | Type | Description |
|---|---|---|
| `row_id` | int | Unique row identifier |
| `paragraph_id` | string | **SHA256 hash of the content** — not unique (see above) |
| `document_id` | string | Source AMF document identifier |
| `content` | string | Paragraph text (20–340 words) |
| `date_envoi` | date | Certified AMF filing date |
| `emetteur` | string | Issuing company (CAC 40) |
| `type_document` | string | Regulatory document type |
| `epoch` | string | Derived regulatory regime (see below) |

### Config `gold` — 140 annotated paragraphs

Same fields as above, plus:

| Field | Type | Description |
|---|---|---|
| `csrd_category` | string | ESRS category, or `none` (12 possible values) |
| `esrs_subcategory` | string | ESRS subcategory, where applicable |
| `chain_of_thought` | string | **Expert's written reasoning** behind the label (see below) |

#### Expert reasoning (`chain_of_thought`)

Each annotation carries the annotator's written justification, structured in three steps:
(1) nature of the disclosed information, (2) materiality analysis under the CSRD double
materiality framework, (3) assessment of market surprise. Reasonings average roughly 620
characters.

This field is released because expert-written justifications on regulatory text are scarce.
It supports uses the labels alone do not: auditing *why* a paragraph was classified a given
way, studying where human and model reasoning diverge, chain-of-thought distillation, or
retraining an annotator on the same protocol.

It is a **single annotator's** reasoning, with no inter-annotator agreement measure — treat it
as documented judgment, not ground truth.

### Regulatory regimes (`epoch`)

| Epoch | Dominant characteristic | n (Gold) |
|---|---|---|
| `2010-2014` | No systematic ESG obligation; informal vocabulary | 33 |
| `2015-2019` | Paris Agreement; French DPEF mandatory (2017) | 30 |
| `2020-2022` | EU Taxonomy, SFDR; "double materiality" emerging | 33 |
| `2023-2026` | CSRD in force, ESRS Set 1, standardised vocabulary | 42 |

### Gold Standard label distribution

`none` 97 · `E1` 18 · `ESRS2` 17 · `E5` 5 · `E2` 1 · `S1` 1 · `S4` 1
· **absent:** `E3`, `E4`, `S2`, `S3`, `G1`

---

## Construction

**Source.** Public portal [info-financiere.gouv.fr](https://www.info-financiere.gouv.fr),
full metadata export (524,589 entries).

**Filtering.** CAC 40 → French language → 2010–2026 period → valid URL, yielding 23,753
documents. Text extracted with `pdfplumber`; 97.44% of documents extracted successfully.

**Paragraph reconstruction.** Raw PDF extraction produces text fragmented line by line (3 to 8
words per line). A naive length filter rejected 95% of extracts. Paragraphs are therefore
reconstructed in two stages (boundary detection, then merging of wrapped lines), followed by
**structural, not semantic** filtering: 20–340 words, ≥ 2 sentences, alphabetic ratio ≥ 0.55,
unique-word ratio ≥ 0.30.

**Gold Standard sampling.** Proportional stratification along three axes — epoch, document
type, and CSRD keyword-density quartile — designed to **avoid any filtering on the variable of
interest**. A CSRD keyword filter would have mechanically biased the corpus toward recent
documents, where that vocabulary is standardised, and under-represented older documents
carrying the same content expressed differently.

---

## Usage

```python
from datasets import load_dataset

# Full corpus
corpus = load_dataset("YOUR_USERNAME/FinCAC40", "corpus", split="train")

# Annotated Gold Standard
gold = load_dataset("YOUR_USERNAME/FinCAC40", "gold", split="test")

# Filter by regulatory epoch
older = corpus.filter(lambda x: x["epoch"] == "2010-2014")
```

---

## License and compliance

Source AMF documents are published under the **Licence Ouverte / Open Licence 2.0 (Etalab)**,
which permits reuse and redistribution subject to attribution. The derived corpus is released
under the same licence.

The corpus contains only public regulatory documents from listed companies. It holds no
personal data within the meaning of the GDPR, other than the names of executives already
appearing in official filings.

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

---

## Contributing

The Gold Standard would benefit from extension, prioritising:

- the **absent ESRS categories** (`E3`, `E4`, `S2`, `S3`, `G1`)
- the **earlier epochs** (2010–2014, 2015–2019), where the number of genuinely CSRD-relevant
  paragraphs is lowest (3 and 4 respectively)
- an **inter-annotator agreement measure**, which this release lacks

The full annotation protocol is provided in the companion code repository.
