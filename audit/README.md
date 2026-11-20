# Member 1: Token audit

## Real data and provenance

The audit input is the original Meta FLORES-200 `devtest` split. The archive is
obtained from the official FLORES repository's published download link:

- Dataset archive:
  <https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz>
- Official repository and dataset README:
  <https://github.com/facebookresearch/flores>
- License: CC-BY-SA 4.0
- Paper citation: NLLB Team et al., *No Language Left Behind: Scaling
  Human-Centered Machine Translation*, 2022.

The official README says language files in `devtest/` share the same order.
The script additionally verifies all four target files contain exactly 1,012
non-empty lines before tokenizing. The archive is ignored by Git because the
raw dataset must not be copied into the code repository.

Downloaded files are expected at:

```text
audit/flores200_dataset/devtest/eng_Latn.devtest
audit/flores200_dataset/devtest/kan_Knda.devtest
audit/flores200_dataset/devtest/hin_Deva.devtest
audit/flores200_dataset/devtest/tam_Taml.devtest
```

## Tokenizers and prices

The audit counts raw sentence text with:

- `cl100k`: OpenAI `cl100k_base` encoding.
- `o200k`: OpenAI `o200k_base` encoding.
- `qwen`: the tokenizer from `Qwen/Qwen2.5-7B-Instruct`, pinned to revision
  `a09a35458c702b33eeacc393d103063234e8bc28`.
- `sarvam`: the tokenizer from `sarvamai/sarvam-1`, pinned to revision
  `e9607337286ddf496d4a2562b194e489dcf3feea`.

Hugging Face model weights are not downloaded; only tokenizer metadata is
loaded. The Qwen and Sarvam results are local token counts, not API charges.

`pricing.json` records the dated conversion quote and model-to-price mapping.
An encoding does not have a price by itself; price belongs to a model. The
cl100k benchmark uses GPT-4 Turbo's published historical input price, but
OpenAI marks GPT-4 Turbo as shut down. The o200k benchmark uses GPT-4o's
published standard input price. These are comparison estimates, not a quote
from the team's KodeKloud endpoint or a claim that the shut-down model can be
called. For current billed cost, replace the benchmark model and price with the
actual provider/model price. The CSV's `cost_inr` is blank for unpriced local
tokenizers.

## Run the audit

From the repository root:

```bash
python -m pip install -r requirements.txt
python -m audit.count_tokens
```

The first run downloads only the two tokenizer assets from Hugging Face if
they are not cached. To prohibit network access:

```bash
python -m audit.count_tokens --offline
```

It writes `data/audit/token_counts.csv` in the shared contract shape:

```text
id,lang,tokenizer,tokens,cost_inr
```

IDs are zero-based FLORES line IDs. Output order is deterministic. Re-running
with the same tokenizer revisions, input files, and pricing snapshot writes the
same CSV. The console reports average tokens per sentence and ratios against
English for each tokenizer; English is 1.00 by construction.

## Run the offline Streamlit demo

```bash
streamlit run audit/app.py
```

The interface reads only local files and cached tokenizer assets. When other
members' experiment results, transform module, router predictions, or measured
failure cases are absent, it explicitly reports those panels as unavailable;
it never fills them with made-up demo rows.

