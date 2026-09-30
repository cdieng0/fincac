# Data: provenance, sampling and schemas

## Where each piece lives

| Data | Where | How to get it |
|---|---|---|
| Corpus, 313,608 paragraphs | [CID99/FinCAC40](https://huggingface.co/datasets/CID99/FinCAC40), config `corpus` | `load_dataset("CID99/FinCAC40", "corpus", split="train")` |
| Gold Standard, 140 paragraphs | same dataset, config `gold` | `python src/publishing/fetch_gold_from_hf.py` writes the workbook the scripts read |
| Run artefacts (probing, ORPO runs) | [`data/results/`](../data/results/) | in this repository |
| ORPO Run 2 pairs, training log, adapter | [CID99/Mistral-7B-ORPO-CSRD](https://huggingface.co/CID99/Mistral-7B-ORPO-CSRD/tree/main) | files `orpo_pairs.jsonl`, `training_log.json`, `adapter_model.safetensors` |
| AMF source export | [info-financiere.gouv.fr](https://www.info-financiere.gouv.fr) | `flux-amf-new-prod.csv`, placed at the repository root |

Everything a script writes goes to `data/`, which Git ignores except for `data/README.md` and
`data/results/` ([`data/README.md`](../data/README.md) lists the files).

---

## Corpus

### Provenance

| Stage | Entries | Script |
|---|---:|---|
| AMF metadata export (`flux-amf-new-prod.csv`, 454 MB, entries since 2006) | 524,589 | — |
| Issuers that are current CAC 40 members | 54,257 | `src/extraction/process_amf_extraction.py` |
| French-language documents | 30,263 | idem |
| Filed 2010–2026 | 23,753 | idem |
| Text extracted with `pdfplumber` (97.44%; failures are 404s and empty or scanned PDFs) | 23,144 | idem |
| **Paragraphs produced** | **313,898** | idem |
| **Published on the Hub** — the 290 paragraphs with empty content removed | **313,608** | `src/publishing/publish_dataset_to_hf.py` |

Raw PDF text comes out one line at a time (3 to 8 words per line), so paragraphs are rebuilt in
two steps — boundary detection (blank lines, upper-case headings), then merging of wrapped
lines up to strong punctuation — and filtered on structure only: 20–340 words, at least two
sentences, alphabetic ratio ≥ 0.55, unique-word ratio ≥ 0.30.

The extraction run records the SHA-256 of the export it read in its `run_meta_{TS}.json`. For
the release, that hash is
`adad19f619981c3e7ae38813b2a8d6523fef6522c99a845c294e9a6d0ccd72f1`. The AMF keeps updating
the export, so a file downloaded today will have a different hash and more entries; use the
Hub corpus when you need the release itself.

### Regulatory eras (`epoch`)

Assigned from the certified AMF filing date.

| `epoch` | Period | Regime |
|---|---|---|
| `2010-2014` | Jan 2010 – Dec 2014 | No systematic ESG obligation; voluntary initiatives (GRI) |
| `2015-2019` | Jan 2015 – Dec 2019 | Paris Agreement; French DPEF mandatory from 2017 |
| `2020-2022` | Jan 2020 – Dec 2022 | EU Taxonomy, SFDR; double materiality emerging |
| `2023-2026` | Jan 2023 – Mar 2026 | CSRD in force; ESRS Set 1 |

### Schema — config `corpus`

| Field | Type | Description |
|---|---|---|
| `row_id` | int | Unique row identifier |
| `paragraph_id` | string | SHA-256 of the content. **Not unique**: about 18% of rows repeat a paragraph that appears elsewhere (disclaimers, standard clauses) |
| `document_id` | string | Source AMF document |
| `content` | string | Paragraph text |
| `date_envoi` | date | Certified AMF filing date |
| `emetteur` | string | Issuer |
| `type_document` | string | AMF document type |
| `epoch` | string | Regulatory era |

The extraction script writes these columns under internal names (`article_id`,
`publish_date`, `author`, `doc_type`); the publication script renames them, and the sampler
accepts either set.

---

## Stratified pool

[`src/sampling/build_annotation_sample_rigorous.py`](../src/sampling/build_annotation_sample_rigorous.py)
turns the corpus into a pool of annotation candidates. The pool is the source of the 500-row
annotation workbook and of the ORPO preference pairs (Gold paragraphs excluded).

| Step | Paragraphs |
|---|---:|
| Corpus | 313,898 |
| Structural filters (length, sentence count, alphabetic and unique-word ratios, page numbers and cross-references) | 251,495 |
| Intra-document cap: at most six paragraphs per document | 97,620 |
| Proportional stratified draw, seed 42 | **14,974** |

- **Strata.** Era × document-type group × keyword-density quartile. Document types below 0.5%
  of the corpus are merged into `other_doc`; none is dropped. Each stratum receives a share
  proportional to its size, with a floor of five paragraphs.
- **Keyword density.** The number of CSRD keywords in the paragraph divided by its length in
  words. Quartiles are computed on the filtered corpus; the lowest, `q0_low`, holds every
  paragraph with no keyword at all. The density is a stratification axis, not a filter.
- **Intra-document cap.** In a document with more than six paragraphs, the six with the
  highest keyword density are kept. This keeps any single filing from dominating the pool, and
  favours keyword-dense paragraphs inside long documents.

The volumes above are those of the release run (`sampling_report_rigorous_20260531_225127.json`
on the author's machine, not archived here).

---

## Gold Standard

### Content

140 paragraphs annotated by one expert annotator under the protocol in
[`annotation_protocol.md`](annotation_protocol.md). There is no inter-annotator agreement
measure.

| Era | Paragraphs | CSRD-relevant |
|---|---:|---:|
| 2010–2014 | 33 | 3 |
| 2015–2019 | 31 | 5 |
| 2020–2022 | 33 | 12 |
| 2023–2026 | 43 | 23 |
| **Total** | **140** | **43** |

Labels: `none` 97 · `E1` 18 · `ESRS2` 17 · `E5` 5 · `E2` 1 · `S1` 1 · `S4` 1 — absent: `E3`, `E4`,
`S2`, `S3`, `G1`. The paragraphs come from 47 issuers; 41 are *Informations privilégiées*.

In 3-shot mode, the three few-shot examples are Gold paragraphs and are removed from the test
set, which leaves 137 paragraphs and 3 / 4 / 12 / 22 relevant ones per era
([`prompts.md`](prompts.md#few-shot-examples)).

### How the 140 were selected

The Gold is not a sub-sample of the 500-row workbook described in the annotation protocol:
the two share 16 paragraphs. Its 140 paragraphs are what remains, after cleaning, of a
150-paragraph annotation batch — rows numbered 1 to 150 in the annotation workbook, ten of
which were removed.

That batch was selected by semantic type, computed before annotation and recorded in the
workbook for every paragraph:

| Pre-computed type | Paragraphs | By era (2010–14 / 15–19 / 20–22 / 23–26) |
|---|---:|---|
| clear `none` | 55 | 24 / 13 / 9 / 9 |
| boundary between `ESRS2` and `none` | 29 | 6 / 8 / 8 / 7 |
| weak CSRD signal | 27 | 2 / 7 / 6 / 12 |
| evident CSRD | 29 | 1 / 3 / 10 / 15 |

Twenty of the 140 were also chosen for their extraction defects (OCR noise, tables).

Two consequences follow. First, the Gold is not a proportional draw from the corpus: its
composition by era and by label reflects this selection as well as the corpus, which matters
when reading per-era error rates built on 3 to 23 relevant paragraphs. Second, the script that
assembled the batch is not yet in this repository, so this step cannot be re-run from the
code here.

### Schema — config `gold`

The corpus fields, plus:

| Field | Type | Description |
|---|---|---|
| `csrd_category` | string | ESRS category, or `none` (12 values) |
| `esrs_subcategory` | string | ESRS subcategory, where applicable |
| `chain_of_thought` | string | The annotator's written justification, about 620 characters on average |

`document_id`, `date_envoi` and `type_document` are empty in this configuration: they were not
carried over from the annotation workbook. `fetch_gold_from_hf.py` renames `content` and
`epoch` to the workbook headers the scripts expect (`content — Extrait à annoter`, `Période`).

---

## Files written by the pipeline

### Benchmark predictions — `predictions_shot{0,3}_{TS}.csv`

Written by `src/evaluation/benchmark_temporal_drift.py`, one row per model and paragraph.

| Column | Content |
|---|---|
| `numero`, `paragraph_id`, `periode`, `societe`, `content` | Gold paragraph |
| `gold_label`, `gold_subcat` | Gold annotation |
| `model_key`, `model_id_resolved`, `n_shot` | Model as requested, model as served, mode |
| `pred_category`, `pred_esrs_subcategory`, `pred_chain_of_thought` | Parsed answer |
| `error` | API error message, if the call failed |

A response that cannot be parsed is recorded as `none` with an empty `pred_chain_of_thought`,
and one that names an unknown category is recorded as `none`. The raw text of the response is
not kept.

### ORPO preference pairs — `orpo_pairs_{TS}.jsonl`

Written by `src/orpo/build_orpo_pairs.py`, one JSON object per line.

| Field | Content |
|---|---|
| `pair_id`, `paragraph_id`, `epoch` | Identifiers |
| `pair_type` | `A_recency_fn`, `B_overprediction_fp` or `C_category_confusion` |
| `prompt` | User turn |
| `chosen`, `rejected` | JSON answers of the judge and of the base policy |
| `chosen_category`, `rejected_category` | Their categories |
| `chosen_model`, `rejected_model` | `mistral-large-latest`, `open-mistral-7b` |

Pool paragraphs that match a Gold paragraph by ID or by their first 100 characters are
excluded before any pair is built.

### ORPO evaluation — `raw_predictions_orpo_*.csv` and `eval_report_orpo.json`

Written by `src/orpo/evaluate_orpo.py`. The predictions file has one row per Gold paragraph:
`paragraph_id`, `content`, `epoch`, `csrd_category` (gold) and `pred_category`, which is empty
when the output could not be parsed. The report is described in
[`data/results/README.md`](../data/results/README.md).

---

## Licence

The AMF documents are published under the Licence Ouverte / Open Licence 2.0 (Etalab), which
allows reuse and redistribution with attribution; the corpus and the Gold Standard are
released under the same licence. The code is under the [Apache License 2.0](../LICENSE). The
corpus holds only public regulatory filings of listed companies and no personal data within
the meaning of the GDPR, other than executives' names already present in those filings.
