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
`results.csv`, once per `native`, `romanized`, and `pivot` strategy. IDs must be
globally unique across languages because the agreed prediction output contains
only `id,chosen_strategy`. Language should not be encoded into the ID.

## Synthetic pipeline check

Install dependencies from the repository root, then run:

```bash
python -m pip install -r requirements.txt
python -m router.train_eval --synthetic
```

This writes clearly labeled synthetic inputs under `data/eval/synthetic_*.csv`,
out-of-fold recommendations to `data/router/synthetic_predictions.csv`, and a
test-only model to `router/synthetic_model.joblib`. The production server
refuses this model. Synthetic inputs are for pipeline checks only; do not
present their results as experiment findings or use them as a deployment model.

## Train on experiment results

After Member C has produced `results.csv`, prepare `examples.csv` with the
matching IDs, language codes, passage IDs, and original text. Run:

```bash
python -m router.train_eval \
  --confirm-real-data \
  --results data/eval/results.csv \
  --examples data/eval/examples.csv
```

The script performs five-fold grouped cross-validation by `passage_id`, asserts
zero passage overlap in each fold, and reports out-of-fold accuracy and mean
tokens with passage-clustered 95% bootstrap intervals for the three fixed
strategies, router, and oracle. The out-of-fold router recommendations go to
`data/router/predictions.csv`. A separate `DecisionTreeClassifier(max_depth=4)`
trained on all real rows is saved to `router/model.joblib` for serving; do not
use its training accuracy as a measured result. The model bundle records data
kind, training counts, timestamp, and input-data SHA-256.

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
export OPENAI_BASE_URL=https://api.ai.kodekloud.com/v1
export OPENAI_MODEL=your-enabled-model-id
export TOKEN_TAX_PROXY_API_KEY=choose-a-separate-client-secret
export TOKEN_TAX_DAILY_REQUEST_LIMIT=100
uvicorn router.server:app --host 127.0.0.1 --port 8000
```

The proxy requires a client bearer token separate from the provider key, refuses
models not marked as trained on real experiment data, and enforces a persistent
UTC-day request-count cap in SQLite. Set `TOKEN_TAX_USAGE_DB` to change the
counter file. This request cap is not a dollar cap: still configure the
provider's hard spend limit, and use one server instance or a shared database
volume because separate database files have separate limits.

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

Set `TOKEN_TAX_PROXY_API_KEY` in the shell that runs the verification script as
well as the server. Keep that key private.

The script selects 50 Kannada prompts and sends each once directly and once
through the proxy (100 API requests total), records
the provider-reported `usage.prompt_tokens` in
`data/router/billed_token_verification.csv`, and flushes each completed pair so
partial results survive a later request failure. This makes paid requests;
never run it before setting the cap. Review the saved file and the provider
dashboard rather than treating local tokenizer estimates as billed usage.

## Optional n8n workflow

Import [`n8n/token-tax-proxy.json`](../n8n/token-tax-proxy.json) into n8n. The
workflow accepts an OpenAI-style JSON request at the `token-tax/optimize`
webhook, forwards its body to the local FastAPI proxy, and returns the proxy
response (including the `token_tax` metadata).

The HTTP Request node defaults to
`http://host.docker.internal:8000/v1/chat/completions`, suitable for common
Docker Desktop setups. Change that URL to the address reachable from the n8n
process. For Linux Docker, configure host-gateway access or use a shared Docker
network and the API container's service name. n8n Cloud cannot reach a laptop's
localhost; it needs a properly secured, reachable API endpoint.

Start the FastAPI server and activate the imported workflow. For a local test,
use the webhook's test URL and a request like:

```json
{
  "model": "your-provider-model",
  "messages": [
    {
      "role": "user",
      "content": "ನಮಸ್ಕಾರ"
    }
  ],
  "language": "kn",
  "stream": false
}
```

The workflow forwards the caller's `Authorization` header to FastAPI; send
`Authorization: Bearer <TOKEN_TAX_PROXY_API_KEY>` to the n8n webhook. The
server rejects missing or invalid credentials and applies its daily request
limit. Do not expose the webhook publicly without adding n8n-side authentication
and network protections. The workflow does not replace the Python router or
Member B's transformations; it only orchestrates a request through them.
