# Hugging Face resources

| Resource | Contents |
|---|---|
| [CID99/FinCAC40](https://huggingface.co/datasets/CID99/FinCAC40) | Corpus (config `corpus`, 313,608 rows, ~88 MB) and Gold Standard (config `gold`, 140 rows). Etalab 2.0 |
| [CID99/Mistral-7B-ORPO-CSRD](https://huggingface.co/CID99/Mistral-7B-ORPO-CSRD) | The Run 2 LoRA adapter, its tokenizer, preference pairs (`orpo_pairs.jsonl`), training log, evaluation report and per-example predictions. Apache 2.0. A research artefact, not a usable classifier |
| [CID99/FinCAC40-collapse-demo](https://huggingface.co/spaces/CID99/FinCAC40-collapse-demo) | Gradio demo comparing the base model with the collapsed one; source in [`space/`](../space/) |

## What lives where

| GitHub (this repository) | Hugging Face |
|---|---|
| Code, tests, prompts, protocol, documentation | Corpus and Gold Standard |
| Small run artefacts ([`data/results/`](../data/results/)) | Adapter weights and ORPO training files |
| Source of the dataset and model cards ([`cards/`](../cards/)) | The published cards |

Large files never go into Git: the corpus, the AMF export (454 MB), model weights and
checkpoints are excluded by [`.gitignore`](../.gitignore).

## Updating the Hub

The cards on the Hub are copies. After editing a file in `cards/`, upload it again — either
paste it into the card editor on the Hub, or:

```bash
huggingface-cli login
huggingface-cli upload CID99/FinCAC40 cards/dataset_card.md README.md --repo-type dataset
huggingface-cli upload CID99/Mistral-7B-ORPO-CSRD cards/model_card.md README.md
huggingface-cli upload CID99/Mistral-7B-ORPO-CSRD prompts/system_prompt_3shot.txt prompt_template.txt
```

The last line replaces the model repository's `prompt_template.txt`, a condensed version, with
the exact prompt used in training and evaluation. Full republication of the data or the model
is described in [PIPELINE.md](PIPELINE.md#10-publication).
