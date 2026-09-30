# Known discrepancies with the preprint (v1)

Checking the released code and data against the
[preprint](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7438503) found the differences
below. Each is verifiable from this repository or from the Hub, and is to be corrected in the
next version of the paper.

| # | Preprint says | Code and data show |
|---|---|---|
| 1 | 313,898 paragraphs are released (Section 3.6). | 313,898 were extracted; the Hub corpus holds 313,608, the 290 empty paragraphs having been removed at publication. |
| 2 | Gold counts per era 33 / 30 / 33 / 42 (Table 2). | The published Gold has 33 / 31 / 33 / 43 (140 paragraphs). |
| 3 | All four models are evaluated on the 137 paragraphs outside the few-shot pool, with 3 / 4 / 12 / 22 relevant ones per era (Section 4.1, Table 3). | The benchmark removes the few-shot examples only in 3-shot. The zero-shot columns are on all 140 paragraphs, with 3 / 5 / 12 / 23 relevant ones — e.g. Mistral-7B's 0.400 in 2015–2019 is 2 / 5. |
| 4 | The 3-shot examples are `none` (Danone, 2010–2014), `E1` (STMicroelectronics, 2023–2026) and `ESRS2` (STMicroelectronics, 2015–2019) (Section 3.4.3). | `build_system_prompt(3)` uses Gold #101 `none` (STMicroelectronics, 2023–2026), #23 `E1` (Total, 2015–2019) and #57 `E2` (TotalEnergies, 2023–2026). The 3-shot predictions file identifies the trio actually used; it is not yet archived here. |
| 5 | Run 2 has a global false-positive rate of 64.9%, almost constant across eras, and an FNR of 0.0–0.043 (Section 4.3, Table 4). | 64.9% (63 / 97) counts every non-`none` answer, including 28 malformed ones. A CSRD label was given to 35 / 97 = 36.1% of `none` paragraphs, lower in 2010–2014 than later (0.23 / 0.38 / 0.48 / 0.40), while malformed answers fall (0.47 / 0.27 / 0.19 / 0.15). With malformed answers counted as `none`, as for κ, the FNR is 0.33 / 0.40 / 0.42 / 0.35. |
| 6 | The Gold is presented alongside a stratified protocol that avoids filtering on the variable of interest (Contribution 3). | That protocol builds the 14,974-paragraph pool. The 140 Gold paragraphs were selected separately, by pre-computed semantic type; see [data.md](data.md#how-the-140-were-selected). |
| 7 | — | The pool's intra-document cap (six paragraphs per document, highest keyword density first) is not described in the paper. |

Item 5 is recomputed from the archived Run 2 predictions by `tests/test_results.py`. Item 3
follows from the reported values: the zero-shot columns contain fractions of 5 and 23, which
cannot arise from 4 and 22 relevant paragraphs.
