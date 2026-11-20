# Member D: Router and Optimizer Server

## Input data contract

The evaluator's `data/eval/results.csv` must contain:

```text
id,lang,strategy,correct,tokens
```

The router also needs the original input text and passage grouping, which are not
in that results schema. Keep those in `data/eval/examples.csv`:

```text
id,passage_id,lang,text
```

Each `(id, lang)` must occur once in `examples.csv` and exactly three times in
`results.csv`, once per `native`, `romanized`, and `pivot` strategy. The agreed
router output has no language column, so the script writes prediction IDs as
`<lang>:<id>` to keep IDs unambiguous when different languages reuse an ID.

## Synthetic pipeline check

Install dependencies from the repository root, then run:

```bash
python -m pip install -r requirements.txt
python -m router.train_eval --synthetic
```

This writes 300 synthetic strategy rows under `data/eval/synthetic_*.csv`, held-out
recommendations to `data/router/predictions.csv`, and a local deployment model
to `router/model.joblib`. Synthetic inputs are for pipeline checks only; do not
present their results as experiment findings.

## Train on experiment results

After Member C has produced `results.csv`, prepare `examples.csv` with the
matching IDs, language codes, passage IDs, and original text. Run:

```bash
python -m router.train_eval \
  --results data/eval/results.csv \
  --examples data/eval/examples.csv
```

The script splits by `passage_id`, asserts there is no group overlap, trains a
`DecisionTreeClassifier(max_depth=4)`, and reports held-out accuracy and mean
tokens for the three fixed strategies, router, and oracle. It writes
`data/router/predictions.csv` and a model fitted on all current examples to
`router/model.joblib`. Report the held-out metrics; the all-data model is only
for serving.

## Configure Member B's transforms and run the proxy

Configure a Python module on `PYTHONPATH` exposing these functions:

```python
def romanize(text: str, lang: str) -> str: ...
def pivot(text: str, lang: str) -> str: ...
```

Then set the transform module and provider credentials, and start the server:

```bash
export TOKEN_TAX_TRANSFORM_MODULE=transform.runtime
export OPENAI_API_KEY=your-provider-key
export OPENAI_BASE_URL=https://api.openai.com/v1
uvicorn router.server:app --host 127.0.0.1 --port 8000
```

The proxy accepts standard non-streaming OpenAI chat-completions requests and
an optional `language` (`en`, `kn`, `hi`, or `ta`) field. It transforms user
messages only; system/developer messages are preserved. Native strategy is a
byte-for-byte content passthrough. If a selected non-native transform is
unavailable, the request fails explicitly with HTTP 503 rather than silently
forwarding untransformed text.

The response's `token_tax` block includes the selected strategy, cl100k
text-only estimates for the original and transformed user text, and the actual
upstream `usage.prompt_tokens` when the provider returns it. The estimate is
not the billed count; use the upstream usage field and provider dashboard for
the billed-token comparison. Streaming is explicitly rejected.

## Verify billed usage

Before any calls, set the provider dashboard's hard spend limit (for example,
$2), check the provider's current prices, and keep `--limit` between 50 and 100.
With the proxy running and `data/eval/examples.csv` prepared, set `OPENAI_MODEL`
and `OPENAI_API_KEY`, then run (the default language is Kannada):

```bash
python -m router.verify_billed_tokens --limit 50
```

The script selects 50 Kannada prompts and sends each once directly and once
through the proxy (100 API requests total), records
the provider-reported `usage.prompt_tokens` in
`data/router/billed_token_verification.csv`, and flushes each completed pair so
partial results survive a later request failure. This makes paid requests;
never run it before setting the cap. Review the saved file and the provider
dashboard rather than treating local tokenizer estimates as billed usage.
