# Member A Experiment Log

## 2026-11-20 — FLORES-200 tokenizer audit

- **Input:** Official FLORES-200 `devtest` files from the Meta/Facebook
  archive. The script verified 1,012 non-empty lines in each of English,
  Kannada, Hindi, and Tamil before counting. FLORES documents a shared
  sentence order across language files.
- **Command:** `python -m audit.count_tokens`
- **Output:** `data/audit/token_counts.csv`, 16,192 rows; schema
  `id,lang,tokenizer,tokens,cost_inr`.
- **Tokenizers:** `cl100k_base`, `o200k_base`, Qwen 2.5 7B tokenizer at
  `a09a35458c702b33eeacc393d103063234e8bc28`, and Sarvam-1 tokenizer at
  `e9607337286ddf496d4a2562b194e489dcf3feea`.
- **Average tokens per sentence:**

  | Tokenizer | English | Kannada | Hindi | Tamil |
  |---|---:|---:|---:|---:|
  | cl100k | 26.86 | 237.96 | 127.83 | 205.31 |
  | o200k | 26.55 | 52.35 | 41.79 | 52.57 |
  | Qwen | 27.29 | 188.94 | 120.53 | 166.83 |
  | Sarvam | 31.05 | 37.81 | 35.53 | 35.96 |

- **Indic/English ratios:**

  | Tokenizer | Kannada | Hindi | Tamil |
  |---|---:|---:|---:|
  | cl100k | 8.86 | 4.76 | 7.64 |
  | o200k | 1.97 | 1.57 | 1.98 |
  | Qwen | 6.92 | 4.42 | 6.11 |
  | Sarvam | 1.22 | 1.14 | 1.16 |

- **Interpretation:** The booklet's 12x Kannada claim was not reproduced by
  this measurement: cl100k measured 8.86x on this FLORES sample. The ratio
  varies substantially by tokenizer; it is not a universal language constant.
- **Price assumptions:** The pricing snapshot uses GPT-4 Turbo at $10 / 1M
  input tokens (historical; model is shut down) for the cl100k reference and
  GPT-4o at $2.50 / 1M input tokens for the o200k reference. The recorded FX
  quote is 1 USD = 93.372746 INR, updated 2026-11-20T00:02:32Z. These are
  OpenAI reference estimates, not KodeKloud billed rates. Qwen and Sarvam
  costs remain blank because these tokenizer counts do not establish an API
  price.
- **Repeatability:** A second run produced the same CSV SHA-256:
  `fee035ca1a4ccc13a7bc620ef12584289f9d1631358a74f006d6887644546682`.
- **Checks:** Unit tests cover line-count/blank-line validation, cost
  calculation, unpriced tokenizers, and the CSV contract. The Streamlit app
  was smoke-tested with Kannada input using cached tokenizer assets.
- **Not completed:** Manual comparison of five Kannada sentences in the
  OpenAI Tokenizer Playground and the independent teammate spot-check still
  need a human reviewer. The measured strategy/accuracy chart and verified
  failure-case examples also remain unavailable until Members B/C provide
  real outputs.
