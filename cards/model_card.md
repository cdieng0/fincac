---
license: apache-2.0
base_model: mistralai/Mistral-7B-Instruct-v0.3
library_name: peft
language:
  - fr
tags:
  - orpo
  - preference-optimization
  - qlora
  - negative-result
  - failure-analysis
  - csrd
  - esrs
  - regulatory
  - alignment
datasets:
  - CID99/FinCAC40
pipeline_tag: text-generation
---

# Mistral-7B-ORPO-CSRD — a documented preference-optimization failure

> ## ⛔ This model is a research artifact, not a usable classifier.
>
> It performs **substantially worse than its own base model** on the task it was trained for.
> It is released so that the failure can be studied and reproduced — not to be deployed.
>
> **If you need CSRD classification, use `mistralai/Mistral-7B-Instruct-v0.3` directly.**
> It is twice as accurate and always returns valid JSON.

**Paper** · [SSRN 7438503](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7438503) —
**Code** · [github.com/cdieng0/fincac](https://github.com/cdieng0/fincac) —
**Dataset** · [CID99/FinCAC40](https://huggingface.co/datasets/CID99/FinCAC40) —
**Demo** · [FinCAC40-collapse-demo](https://huggingface.co/spaces/CID99/FinCAC40-collapse-demo)

---

## What this is

A QLoRA adapter for `mistralai/Mistral-7B-Instruct-v0.3`, trained with **ORPO**
(Odds Ratio Preference Optimization) to correct a suspected recency bias in CSRD/ESRS
sustainability classification of French regulatory filings.

**The correction failed.** This repository documents *how* it failed, with the training
artifacts, evaluation predictions, and diagnostic metrics needed to reproduce the analysis.

## Why release a failed model?

In run 1, three signals said the training had succeeded. All three were wrong:

| Signal observed in run 1 | What it suggested | Reality |
|---|---|---|
| `eval_loss` falling from 1.03 to 0.147 | Healthy convergence | The loss measures the preference objective, not the task |
| Preference accuracy **95%** at the first evaluation, **100%** at the end | Objective fully learned | The shortcut "always prefer a CSRD label" satisfies 95% of the pairs |
| **FNR = 0.000 in all four eras** | **Target bias eliminated** | Reached because the model no longer answered `none` |

Meanwhile, task accuracy fell from **70.8% to 16.4%**. The run's configuration, training log
and evaluation report are archived in the
[code repository](https://github.com/cdieng0/fincac/tree/main/data/results).

The third row is the interesting one. False Negative Rate is precisely the metric that
operationalises the hypothesis under test — and it reached its ideal value of zero. But it did
so by collapse: the model simply stopped predicting `none`, which drives FNR to zero
mechanically without correcting anything.

A practitioner following standard preference-tuning practice — monitor the loss, monitor
preference accuracy, check the target metric — would have concluded the bias was fixed and
shipped a model that had lost 54 accuracy points.

This is a concrete instance of Goodhart's law on a real regulatory task, with the data to
back it.

---

## Measured performance

Evaluated on the expert-annotated Gold Standard of
[FinCAC40](https://huggingface.co/datasets/CID99/FinCAC40), never seen during training: the 140
paragraphs for the ORPO runs, and the 137 left once the three few-shot examples are removed for
the 3-shot base model.

| | Accuracy | Macro-F1 | κ [95% CI] | Malformed output | `none` paragraphs given a CSRD label |
|---|---|---|---|---|---|
| **Base model** (3-shot) | **70.8%** | 18.7% | 0.193 [0.051, 0.352] | **0%** | not measured¹ |
| ORPO — Run 1 | 16.4% | 23.1% | not archived² | not archived² | not archived² |
| **ORPO — Run 2** (this release) | 35.7% | 35.9% | 0.238 [0.079, 0.387] | **30.7%** | **36.1%** (35 / 97)³ |

κ is Cohen's kappa on the binary CSRD vs. `none` decision, with bootstrap confidence intervals
(B = 2000, seed 42).
¹ The false-positive rate was added to the evaluation protocol after run 1.
² Run 1's per-example predictions were not preserved, so these values cannot be recomputed;
its accuracy and macro-F1 were recorded by the evaluation run itself.
³ Another 28 of the 97 `none` paragraphs received malformed output. The paper's global FPR of
64.9% (63 / 97) counts both.

**Read the confidence intervals.** κ moves from 0.193 to 0.238, but the intervals overlap
heavily. On n = 140 this apparent gain is indistinguishable from sampling noise, and we do not
claim it as an improvement.

### Two failure modes, diagnosed

**1. Refuge category.** `ESRS2` (general governance) simultaneously reaches 5.3% precision
(1 correct out of 19 predictions) and 5.9% recall (1 recovered out of 17 true instances). Both
near zero at once means the label carries no discriminative signal: the model uses it as a
default when uncertain, triggered notably by the generic word *"risque"* appearing in
standardised legal disclaimers.

**2. Capability regression.** The tuned model emits unparseable JSON in 30.7% of cases, against
0% for the base model. Fine-tuning did not merely fail to correct the target bias — it degraded
a formatting capability the base model had fully mastered.

---

## Why it failed

Four cumulative factors, not alternatives:

1. **Pair imbalance.** Run 1 trained on 300 "prefer a CSRD label over `none`" pairs against 15
   guardrail pairs (10 type B, 5 type C). The shortcut "always prefer a CSRD label" satisfies
   95% of that training signal.
2. **Fine taxonomy, scarce data.** 273–315 preference pairs cannot cover a 12-class output
   space, especially for categories with 1–5 examples in the Gold Standard.
3. **No quality hard negatives for rare classes.** Rejection sampling only produces pairs where
   judge and policy disagree, which mechanically under-samples already rare categories.
4. **Judge and policy from the same model family.** `mistral-large-latest` judged
   `open-mistral-7b`; a shared bias between them cannot be excluded.

We attribute the failure to this constrained-data regime, **not** to a limitation of ORPO as a
method.

---

## Training configuration

| | |
|---|---|
| Base model | `mistralai/Mistral-7B-Instruct-v0.3` |
| Method | ORPO ([Hong et al., 2024](https://arxiv.org/abs/2403.07691)) |
| Quantization | QLoRA 4-bit NF4, double quant, bfloat16 compute |
| LoRA | r = 16, α = 32, dropout 0.05, all linear projections |
| ORPO β | 0.1 |
| Preference pairs | 225 type-A / 30 type-B / 15 type-C (after rebalancing) |
| Training | 2 epochs, learning rate 2e-5, effective batch 16 (Run 1: 3 epochs, 5e-5) |
| Seed | 42 |

Preference pairs were built by **rejection sampling**: for each paragraph, a silver label from a
stronger judge was compared against the base policy's own 3-shot prediction. Disagreements
became `chosen`/`rejected` pairs, so every pair captures a genuine model error rather than a
fabricated counterexample.

---

## Intended and out-of-scope use

**Intended.** Research on preference-optimization failure modes; reproducing the collapse;
studying evaluation blind spots in alignment training; teaching material on Goodhart's law.

**Out of scope.** Any production classification. Any regulatory, compliance, audit, or
investment decision. Any use where a 30.7% malformed-output rate is not acceptable — which is
essentially all of them.

---

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_id = "mistralai/Mistral-7B-Instruct-v0.3"
tok = AutoTokenizer.from_pretrained(base_id)
model = AutoModelForCausalLM.from_pretrained(base_id, device_map="auto")
model = PeftModel.from_pretrained(model, "CID99/Mistral-7B-ORPO-CSRD")

# Expect over-prediction of CSRD categories and ~30% malformed JSON.
# For a working classifier, drop the PeftModel line and use the base model.
```

The exact system prompt used for training and for every evaluation reported above is
[`prompts/system_prompt_3shot.txt`](https://github.com/cdieng0/fincac/blob/main/prompts/system_prompt_3shot.txt)
in the code repository, wrapped as `<s>[INST] {prompt}\n\nExtrait : «{paragraph}» [/INST]`
and decoded greedily.

---

## Repository contents

- `adapter_model.safetensors`, `adapter_config.json` — the LoRA adapter
- `prompt_template.txt` — the system prompt used across all evaluations (see Usage)
- `training_log.json` — full training telemetry (the metrics that looked healthy)
- `orpo_pairs.jsonl` — the preference pairs used for training
- `eval_predictions.csv` — per-example predictions on the 140 Gold paragraphs
- `eval_report.json` — all metrics reported above. Its `parse_failure_rate` field reads 0.0:
  the script version that wrote it tested for the string `PARSE_ERROR`, while unparseable
  outputs were stored as missing values. The correct rate, 43 / 140 = 30.7%, is given by
  `orpo_confusion.predicted_category_distribution`; the corrected script is
  [`src/orpo/evaluate_orpo.py`](https://github.com/cdieng0/fincac/blob/main/src/orpo/evaluate_orpo.py).

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

## Related

- 📄 Paper: [SSRN 7438503](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7438503)
- 💻 Code and run artefacts: [github.com/cdieng0/fincac](https://github.com/cdieng0/fincac)
- 📊 Dataset: [FinCAC40](https://huggingface.co/datasets/CID99/FinCAC40)
- 🔬 Demo: [collapse comparison Space](https://huggingface.co/spaces/CID99/FinCAC40-collapse-demo)
