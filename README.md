# Falcon: The Token Tax

Falcon is an interactive tokenizer and text-conversion research application for
measuring how scripts and language representations affect token usage.

The main application is a Streamlit web app that lets you:

- Count the same text with multiple local or benchmark tokenizers.
- Compare native Indic text, ISO Romanized text, and an English translation.
- Explore measured FLORES-200 token counts by language and tokenizer.
- View a reference cost estimate for tokenizers mapped to published model prices.
- Inspect measured strategy-quality data when an evaluation result file exists.

The project currently supports English, Kannada, Hindi, and Tamil in its data
pipeline. The live conversion studio supports Kannada, Hindi, and Tamil as
source languages.

## Main application

Start the tokenizer app from the repository root:

```bash
python -m pip install -r requirements.txt
python -m streamlit run audit/app.py
```

Then open the local URL printed by Streamlit, normally
<http://localhost:8501>.

### Application tabs

#### Overview

Displays the measured project summary, including the 1,012 aligned FLORES-200
sentences in each of the four audited languages. It also shows average token
counts and the relative token burden compared with English.

#### Convert text

This is the live conversion and tokenizer studio.

1. Choose Kannada, Hindi, or Tamil.
2. Enter or edit text in the source-language box.
3. Select one of three conversion modes:
   - **Native:** keeps the original text unchanged.
   - **Romanized:** converts the Indic script to Latin characters with ISO
     transliteration while preserving the source wording.
   - **English pivot:** translates the meaning into English with
     `facebook/nllb-200-distilled-600M`.
4. The app counts the native and converted representations with the locally
   available tokenizers and charts the result.

Romanization runs locally and updates immediately. English pivot translation
runs in a background worker and can take time on CPU. The first pivot run
downloads the NLLB model weights, which require additional disk space and
network access.

#### Token data

Compares measured average token counts for English and one selected Indic
language. It also calculates the Indic/English token ratio for every tokenizer
in the audit dataset.

#### Quality results

Displays accuracy-versus-token-cost results only when
`data/eval/results.csv` is present and valid. Until real evaluation results are
supplied, the app explicitly reports that measured quality data is unavailable.

## Tokenizers

The audit uses these tokenizer implementations:

| Name | Implementation | Data type |
|---|---|---|
| `cl100k` | OpenAI `cl100k_base` | Reference encoding |
| `o200k` | OpenAI `o200k_base` | Reference encoding |
| `qwen` | `Qwen/Qwen2.5-7B-Instruct` | Local Hugging Face tokenizer |
| `sarvam` | `sarvamai/sarvam-1` | Local Hugging Face tokenizer |

The Qwen and Sarvam model revisions are pinned in `audit/count_tokens.py`.
Hugging Face loads tokenizer metadata rather than downloading model weights for
the four-tokenizer audit.

## Convert text strategies

### Native

Returns the source text unchanged. This is the baseline representation.

### Romanized

Uses `indic-transliteration` to change the writing system to Latin script.
Romanization preserves the source wording; it is not translation.

Example behavior:

```text
Kannada: ನಮಸ್ಕಾರ
ISO:       namaskaara
```

### English pivot

Uses Meta's NLLB-200 distilled 600M model to translate the source text into
English. NLLB requires substantially more setup and runtime resources than
Romanization.

CPU-only usage is supported, but the first model download and inference can be
slow. A CUDA-compatible GPU is used automatically when PyTorch detects one.

## Measured token audit

The data audit is based on the official Meta FLORES-200 `devtest` split:

- English: `eng_Latn`
- Kannada: `kan_Knda`
- Hindi: `hin_Deva`
- Tamil: `tam_Taml`

The audit expects 1,012 aligned, non-empty lines per language. It verifies the
line counts and alignment before counting tokens.

### Download the FLORES archive

The detailed audit instructions and provenance are in
[audit/README.md](audit/README.md). The FLORES dataset is not committed to this
repository.

Place the downloaded files under:

```text
audit/flores200_dataset/devtest/
├── eng_Latn.devtest
├── kan_Knda.devtest
├── hin_Deva.devtest
└── tam_Taml.devtest
```

### Generate token counts

The normal online run:

```bash
python -m audit.count_tokens
```

To require cached Hugging Face tokenizer assets and disable network downloads:

```bash
python -m audit.count_tokens --offline
```

The audit writes:

```text
data/audit/token_counts.csv
```

The CSV contract is:

```text
id,lang,tokenizer,tokens,cost_inr
```

The `id` field is the zero-based FLORES sentence position. Output ordering is
deterministic, and unpriced local tokenizers have an empty `cost_inr` field.

## Cost estimates

Cost information in `audit/pricing.json` is a reference estimate, not a statement
of actual billed usage.

- `cl100k` maps to GPT-4 Turbo for a historical benchmark. That model is shut
  down and may not be callable.
- `o200k` maps to GPT-4o's published standard input price.
- Qwen and Sarvam are local tokenizers and have no API price in this audit.
- The USD-to-INR rate is stored as a dated snapshot and is not a live quote.

When deploying or billing against a real provider, replace the benchmark entry
with the actual model and provider pricing.

## Installation

### Windows

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run audit/app.py
```

If PowerShell blocks activation, use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run audit/app.py
```

### Python requirement

Use a supported Python 3 environment. The project dependencies include PyTorch
only for the optional NLLB pivot translation workflow; the main tokenizer count
and Romanization interfaces do not require model inference.

For a CPU-only PyTorch installation, install the appropriate wheel for your
platform first, then install the optional pivot dependencies:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r transform/requirements.txt
```

For GPU systems, install the PyTorch build matching your CUDA environment before
installing `transform/requirements.txt`.

## Optional demo command

On Unix-like systems, `run_round1_demo.sh` runs the offline audit and starts the
Streamlit app using `.venv`:

```bash
./run_round1_demo.sh
```

The script assumes a virtual environment named `.venv` and uses its Python
interpreter.

## Project structure

```text
audit/
  app.py                  # Main Streamlit tokenizer application
  count_tokens.py         # FLORES audit and CSV generation
  pricing.json            # Dated reference pricing and exchange-rate snapshot
  README.md               # Dataset and audit provenance

transform/
  romanize.py             # Native text to ISO Romanization
  pivot.py                # NLLB English pivot translation
  requirements.txt        # Optional translation dependencies

router/
  features.py             # Router feature extraction
  server.py               # FastAPI OpenAI-compatible proxy
  train_eval.py           # Router training and evaluation
  transforms.py           # Native, Romanized, and pivot strategies

tests/
  test_audit.py           # Audit loader, count, price, and CSV tests
  test_conversion.py      # Romanization tests
  test_server.py          # Proxy and server tests
  test_router_features.py # Router feature tests

data/
  audit/                  # Generated token counts
  eval/                   # Optional measured evaluation results
```

## Testing

Run the test suite:

```bash
python -m pytest
```

To run only focused audit tests:

```bash
python -m pytest tests/test_audit.py tests/test_conversion.py
```

## Optional API proxy

The `router/server.py` module provides an OpenAI-compatible FastAPI proxy that
can apply native, Romanized, or English-pivot strategies before forwarding a
request. It requires configuration and a valid router model bundle.

The proxy is separate from the Streamlit tokenizer app. For local app usage,
running `audit/app.py` is sufficient.

## Data and privacy

- Token counts run locally once tokenizer assets are available.
- Cached Hugging Face assets remain in the local Hugging Face cache.
- Romanization is local.
- NLLB pivot translation may download model weights and performs inference
  locally.
- The FLORES source archive and raw generated project data should not contain
  private user text.

## Troubleshooting

### Streamlit cannot find a tokenizer

Run the audit once while online:

```bash
python -m audit.count_tokens
```

Then use offline mode in the app or shell command.

### Offline app token counting fails

The live inspector requires the Qwen and Sarvam tokenizer assets to already be
cached locally. Cache them by running the normal audit once with network access,
then retry with `--offline`.

### English pivot translation is slow or fails

- Confirm `transform/requirements.txt` dependencies are installed.
- Allow enough disk space for the NLLB model download.
- On CPU, reduce expectations for generation speed.
- If the model is not downloaded, confirm network access and retry.

### The audit says the FLORES files are missing

Download the official FLORES-200 archive and extract the four `devtest` files
into `audit/flores200_dataset/devtest/`, as described above.

## Dataset and model sources

- FLORES-200: <https://github.com/facebookresearch/flores>
- FLORES archive: <https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz>
- Qwen2.5-7B-Instruct: <https://huggingface.co/Qwen/Qwen2.5-7B-Instruct>
- Sarvam-1: <https://huggingface.co/sarvamai/sarvam-1>
- NLLB-200 distilled 600M:
  <https://huggingface.co/facebook/nllb-200-distilled-600M>

FLORES-200 is released under CC-BY-SA 4.0. Verify model licenses and provider
terms separately before production use.