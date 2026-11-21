# Token Tax

Token Tax is a local research dashboard for examining how common tokenizers handle Indian-language text. It includes tokenizer benchmarks, script transliteration comparisons, a standalone SentencePiece experiment, and an interactive native-language sentence optimizer.

The project reports measured tokenizer counts where available. Token counts are not proof of semantic equivalence, translation quality, answer quality, production cost, or support by a particular language model.

## Contents

- [Capabilities](#capabilities)
- [Technology stack](#technology-stack)
- [Requirements](#requirements)
- [Initialize the project on Windows](#initialize-the-project-on-windows)
- [Run the application](#run-the-application)
- [Dashboard guide](#dashboard-guide)
- [HTTP API](#http-api)
- [Experiments and output files](#experiments-and-output-files)
- [Interpretation and limitations](#interpretation-and-limitations)
- [Tests](#tests)
- [Troubleshooting](#troubleshooting)

## Capabilities

- Inspect local token counts and token pieces for user-provided text.
- Compare fertility measurements across `cl100k_base`, `o200k_base`, and `p50k_base`.
- Run Indic-script transliteration and compare measured input-string token counts.
- Train a separate multilingual SentencePiece BPE tokenizer and evaluate held-out or custom text.
- Clean conservative whitespace and Unicode formatting issues in user-provided text without translating or transliterating.
- Generate a separate native-query noise experiment with unmodified controls and synthetic formatting variants.

## Technology stack

| Component | Purpose |
| --- | --- |
| Python 3.10+ | Application logic, experiments, and command-line entry point |
| Flask | Local web server and JSON API |
| `tiktoken` | Local `cl100k_base`, `o200k_base`, and `p50k_base` token counts |
| `indic-transliteration` | Indic-script conversion in the transliteration experiment |
| SentencePiece | Training and evaluating a standalone multilingual BPE tokenizer |
| HTML, CSS, JavaScript | Dashboard UI, interaction, and visualizations |
| Chart.js 4 | Dashboard charts, loaded from jsDelivr |
| JSON | Source corpus and saved experiment results |
| Python standard library | Unicode processing, hashing, temporary files, and tests |

The dashboard and optimizer do not send user text to an AI model. The page loads Chart.js from a CDN for charts; the Python experiments and API calculations run locally.

## Requirements

- Windows, macOS, or Linux
- Python 3.10 or later
- `pip`
- Internet access during dependency installation
- Internet access in the browser for the Chart.js CDN charts; other dashboard functions remain local

Dependencies are listed in [requirements.txt](requirements.txt).

## Initialize the project on Windows

Open PowerShell in the project directory:

```powershell
Set-Location "C:\path\to\token_tax"
```

Create and activate a virtual environment:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If `py -3` selects an unavailable interpreter, specify an installed version, for example `py -3.12`. Then install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Check that Python and the dependencies are available:

```powershell
python --version
python -m pip show flask tiktoken sentencepiece indic-transliteration
```

To leave the virtual environment, run `deactivate`.

## Run the application

Start the dashboard:

```powershell
python run_all.py dashboard
```

Open <http://127.0.0.1:5000/> in a browser. Keep the terminal open while using the app; press **Ctrl+C** there to stop the server. The dashboard binds to localhost and is intended for local research, not direct public deployment.

The dashboard calculates benchmark, transliteration, and SentencePiece data when it first loads. This may take a short time. It does not call an AI provider.

### Run an experiment from the command line

Run one component:

```powershell
python run_all.py bench
python run_all.py pipeline
python run_all.py vocab
python run_all.py native-noise
```

Run the standard benchmark, transliteration, and vocabulary experiments in sequence:

```powershell
python run_all.py all
```

`native-noise` is a separate command and is not included in `all`. To see available commands:

```powershell
python run_all.py --help
```

## Dashboard guide

### Live inspector

Enter text to see its locally measured token counts, approximate Unicode word-unit count, script detection, and token pieces. Script detection does not identify a language when multiple languages use the same script. A transliteration result is shown only when the installed library supports that script.

### Fertility benchmark

Displays measurements from the curated shared corpus using three `tiktoken` encodings. Word counts are Unicode letter/number runs, not language-specific morphological analysis. Fertility and English comparisons are therefore approximations.

### Transliteration

Converts supported Indic-script text to ITRANS and compares input-string token counts using `cl100k_base`. Transliteration changes script; it is not translation. Urdu is unsupported by the configured script mapping. A lower transliterated count does not establish semantic preservation or reduced production model cost.

### SentencePiece

The project trains a standalone SentencePiece BPE tokenizer using the training split of the shared corpus and evaluates held-out questions. The dashboard also accepts custom text for local tokenization and reconstruction checks. SentencePiece pieces are not GPT or Gemini tokens, and the tokenizer is not merged into a model vocabulary.

### Sentence optimizer

Select a language and tokenizer, enter text, and choose **Optimize**. The optimizer:

- Applies NFC normalization and conservative whitespace cleanup.
- Detects selected invisible formatting characters and preserves orthographic joiners.
- Shows adjacent repeated-word candidates but retains them unless a user explicitly confirms they are accidental.
- Measures before-and-after token counts using the selected local `tiktoken` encoding.
- Rejects a proposed result if its selected-encoding token count increases.

String and whitespace checks do not prove semantic equivalence. Review any suggested duplicate removal before confirming it. The original input remains available in the results.

## HTTP API

The Flask dashboard exposes these routes:

| Method and route | Purpose |
| --- | --- |
| `GET /api/data` | Return the benchmark, transliteration, and vocabulary data used by the dashboard |
| `POST /api/inspect` | Measure tokenization, word units, and script/transliteration availability for text |
| `POST /api/vocab/inspect` | Evaluate custom text with the trained SentencePiece model and compare against cl100k |
| `POST /api/native-optimize` | Optimize user text and return measured token counts and reported changes |

Example request for the optimizer in PowerShell:

```powershell
$body = @{
  text = "  ಕನ್ನಡ   ವಾಕ್ಯ  "
  language = "kn"
  encoding = "cl100k_base"
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "http://127.0.0.1:5000/api/native-optimize" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

Supported optimizer language codes are `kn` (Kannada), `ta` (Tamil), `te` (Telugu), `ml` (Malayalam), `hi` (Hindi), `mr` (Marathi), `bn` (Bengali), `gu` (Gujarati), `pa` (Punjabi), `or` (Odia), `ur` (Urdu), and `en` (English). Supported encodings are `cl100k_base`, `o200k_base`, and `p50k_base`.

The optimizer response includes the original and optimized strings, original and optimized token counts, token savings and percentage, repeated-word candidates and removals, whitespace changes, and optimization status. It also indicates that semantic equivalence has not been proven.

If the response includes repeated-word candidates, review them first. To confirm candidates, send the same original input again and include the reviewed candidate IDs in `confirmed_duplicate_ids`. Unknown IDs are rejected. The optimizer input limit is 10,000 characters; the custom SentencePiece and general inspection endpoints have their own input limits.

## Experiments and output files

The shared source corpus is [indian_language_questions.json](indian_language_questions.json). It contains curated draft prompts in 11 Indian languages and is used by the benchmark, transliteration, SentencePiece, and native-noise experiments.

| File | Role |
| --- | --- |
| `1_fertility_benchmarker.py` | Token counts, approximate word counts, fertility, and English comparisons |
| `2_transliteration_pipeline.py` | Indic-script conversion, token counts, and round-trip diagnostics |
| `3_vocab_expansion_demo.py` | SentencePiece training, held-out evaluation, and custom-text evaluation |
| `native_language_optimizer.py` | User-input sentence cleanup and token measurement |
| `native_noise_experiment.py` | Synthetic noise dataset generation and technique measurements |
| `4_dashboard.py` | Flask dashboard and JSON API |
| `run_all.py` | Command-line entry point |
| `tests/` | Unit and dashboard/API tests |

Generated result files:

- `benchmark_results.json` — benchmark measurements.
- `pipeline_results.json` — transliteration counts and round-trip diagnostics.
- `vocab_expansion_results.json` — SentencePiece training and held-out results.
- `native_noise_dataset.json` and `native_noise_results.json` — clean controls, synthetic variants, and local measurements.

The dashboard computes current local measurements when it starts rather than reading all saved result files. Commands that run experiments write their result files alongside the scripts.

## Interpretation and limitations

- The 110 multilingual prompts are curated drafts, not a naturally sampled corpus; native-speaker review has not been established.
- The native-noise variants are deterministic synthetic perturbations, not collected user inputs.
- Unicode, whitespace, and token-count comparisons cannot establish that meaning or answer quality is preserved.
- Transliteration round trips and character similarity are diagnostics, not semantic accuracy.
- Token counts are not API cost estimates. Cost depends on the selected model, provider prices, complete prompt, and generated response.
- SentencePiece is a separate tokenizer; piece-count comparisons do not represent GPT/Gemini token savings.
- Results depend on the input corpus, library versions, and tokenizer encodings. Re-run experiments to reproduce measurements in the current environment.

## Tests

From the project directory, run:

```powershell
python -m unittest discover -s tests
```

The suite covers tokenizer experiments, optimizer safety and measurements, the native-noise experiment, and dashboard/API behavior.

## Troubleshooting

### `ModuleNotFoundError`

Activate the virtual environment and reinstall project dependencies:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### The dashboard port is already in use

Stop the other local server or change the `port` argument in `4_dashboard.py`'s `main()` function, then open the corresponding localhost URL.

### Charts are missing

Chart.js is loaded from jsDelivr. Check browser internet access and the browser console. Token inspection and API processing do not depend on the charts.
