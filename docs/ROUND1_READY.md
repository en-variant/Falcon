# Round 1 — 20-Minute Evaluation Readiness

## One-sentence pitch

The Token Tax measures whether rewriting Kannada, Hindi, and Tamil text as native script, romanized text, or English pivot text reduces real model token consumption without hiding quality or infrastructure costs.

## Verified live demo

1. Open `http://localhost:8501`.
2. On Overview, show the measured language/tokenizer chart and the observed 8.86× cl100k Kannada ratio.
3. Open **Convert text** and enter `ನಾಳೆ ಬೆಂಗಳೂರಿನಲ್ಲಿ ಮಳೆ ಬರುತ್ತದೆಯೇ?`.
4. Show the automatic ISO romanization and the live token comparison chart.
5. To demonstrate English pivot, click **Translate to English** only if you have time for the first NLLB model download (about 2.4 GB); it may be slow on CPU. Otherwise demonstrate automatic romanization and token comparisons.
6. Point out that tokenizer counts come from pinned Qwen/Sarvam tokenizers and official OpenAI encodings, not fabricated values.

Start with:

```bash
.venv/bin/python -m audit.count_tokens --offline
.venv/bin/python -m streamlit run audit/app.py --server.port 8501
```

Romanization is automatic in the Convert tab. NLLB translation needs internet
on first use to fetch weights and may be slow on CPU; do not start its first
download during a short presentation.

## Measured Member A findings

FLORES-200 devtest is real: 1,012 aligned sentences in English, Kannada, Hindi, and Tamil.

| Tokenizer | Kannada / English | Hindi / English | Tamil / English |
|---|---:|---:|---:|
| cl100k | 8.86× | 4.76× | 7.64× |
| o200k | 1.97× | 1.57× | 1.98× |
| Qwen | 6.92× | 4.42× | 6.11× |
| Sarvam | 1.22× | 1.14× | 1.16× |

The 12× booklet claim is not reproduced for `cl100k`; report the measured 8.86× and avoid presenting a universal token-tax number.

## Honest evidence available now

- Official FLORES-200 four-language input files
- 16,192 real token-count rows
- Pinned Qwen and Sarvam tokenizer revisions
- Member B romanization implementation and round-trip notes
- Member D grouped synthetic pipeline and FastAPI/n8n infrastructure
- Streamlit demo with automatic romanization, measured charts, and explicit unavailable states
- 19 passing repository tests

## Do not claim in Round 1

- No measured model accuracy because no `data/eval/results.csv` is present.
- No completed pivot or real Member C quality run.
- No trained-real-data router comparison or oracle result.
- No real billed-token API run.
- No human-verified failure-case set.
- No claim that the 12× claim is confirmed.
- No claim that romanization preserves meaning: only round-trip string behavior was measured.

## What to say if asked about missing evidence

“We have verified the token side and the serving pipeline. Strategy quality and billed-cost experiments are the remaining measured inputs. The demo deliberately does not invent them.”

## Required final checklist before judges

- Keep Wi-Fi available only for the app startup; the audit demo runs from cached files.
- Start the Streamlit server before the judge arrives.
- Test the Kannada/English inspector once.
- Mention Member B ISO romanization and that round-trip exact-match varied by language.
- If asked for API cost, distinguish OpenAI reference prices and FX estimates from real KodeKloud billed usage.
- Do not show synthetic router results as team findings.

## 90-second presentation sequence

1. Problem: Indic scripts may consume substantially more model tokens.
2. Evidence: aligned FLORES sentences isolate language as the variable.
3. Result: the measured Kannada/cl100k ratio is 8.86×, but it varies sharply by tokenizer.
4. Strategy: native, romanized, and English pivot tradeoff.
5. Product: offline Streamlit inspector plus OpenAI-compatible optimizer proxy.
6. Failure honesty: quality and billed-cost measurements are pending, so the demo labels missing evidence instead of fabricating it.
