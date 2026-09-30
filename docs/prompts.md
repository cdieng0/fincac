# Prompts

Every model call in the project — the four-LLM benchmark, the ORPO pair collection, ORPO
training and evaluation — uses one system prompt, defined once in
[`src/evaluation/shared_prompts.py`](../src/evaluation/shared_prompts.py). Why it is defined
once is explained in [INCIDENT_NOTE.md](INCIDENT_NOTE.md#1-prompt-drift-between-scripts).

## The exact prompts

| File | Built by | Used for |
|---|---|---|
| [`prompts/system_prompt_0shot.txt`](../prompts/system_prompt_0shot.txt) | `build_system_prompt(0)` | Zero-shot benchmark |
| [`prompts/system_prompt_3shot.txt`](../prompts/system_prompt_3shot.txt) | `build_system_prompt(3)` | 3-shot benchmark, ORPO pairs, training, evaluation |

[`tests/test_prompts.py`](../tests/test_prompts.py) fails if these files, the benchmark's own
copy or the ORPO scripts diverge from `shared_prompts.py`, or if the ORPO base policy stops
being queried in 3-shot.

The prompt is in French, like the paragraphs. It is assembled from three blocks:

1. `TAXONOMY_BLOCK` — the twelve categories with their definitions and subcategories.
2. The few-shot examples, in 3-shot mode only.
3. `TASK_BLOCK` — the auditor role, the instructions, and the required JSON output:
   `csrd_category`, `esrs_subcategory`, `chain_of_thought`.

## Few-shot examples

`build_system_prompt(3)` takes the first three entries of `FEW_SHOT_EXAMPLES`. They are Gold
paragraphs, so the 3-shot benchmark removes them from its test set (137 paragraphs left).

| Gold # | Issuer | Era | Label |
|---|---|---|---|
| 101 | STMicroelectronics | 2023–2026 | `none` |
| 23 | Total | 2015–2019 | `E1` |
| 57 | TotalEnergies | 2023–2026 | `E2` |

The preprint describes a different trio; see
[known_discrepancies.md](known_discrepancies.md). The example answers use the key `reasoning`
where the task block asks for `chain_of_thought`; the parsers accept both, and no metric reads
that field.

The live mode of the demo in [`space/`](../space/) uses a shorter prompt of its own (the
condensed summary also published as `prompt_template.txt` on the model repository); the
demo's offline mode replays the archived evaluation predictions.

## How each call is made

| Where | Format |
|---|---|
| Benchmark, pair collection (APIs) | system message = the prompt; user message = `Extrait : «{paragraph}»`; temperature 0, seed 42, JSON mode for Mistral and OpenAI, 256 output tokens |
| ORPO training and evaluation (local weights) | `<s>[INST] {prompt}\n\nExtrait : «{paragraph}» [/INST]`, written by hand rather than with `apply_chat_template`; greedy decoding |

## Changing a prompt

A change of prompt is a change of experimental condition. Edit `shared_prompts.py` only,
regenerate the two files in `prompts/`, and record the change in the
[changelog](../CHANGELOG.md):

```bash
python -c "from src.evaluation.shared_prompts import build_system_prompt as b; \
[open(f'prompts/system_prompt_{n}shot.txt', 'w', encoding='utf-8', newline='\n').write(b(n)) for n in (0, 3)]"
python -m pytest tests/test_prompts.py
```
