# Token Tax

Token Tax is a local research application for examining how tokenizer encodings,
Indic-script transliteration, and native-language text formatting affect measured
token counts.

The application includes a Flask dashboard, command-line experiments, and a
native-language sentence optimizer. It runs locally and does not send dashboard
text to an AI provider.

## Main application

Start the Token Tax dashboard from the repository root:

```powershell
python -m pip install -r token_tax/requirements.txt
python token_tax/run_all.py dashboard
```

Open <http://127.0.0.1:5000/> in a browser.

Keep the terminal open while using the dashboard. Press `Ctrl+C` to stop the
server.

The dashboard provides:

- A live token inspector for custom text.
- A fertility benchmark for Indian languages and English.
- An Indic-script to ITRANS transliteration comparison.
- A separately trained SentencePiece BPE tokenizer.
- A native-language sentence optimizer.

## Dashboard features

### Live inspector

Enter text in the live inspector to measure:

- `cl100k_base` token count.
- `o200k_base` token count.
- `p50k_base` token count.
- Approximate Unicode word-unit count.
- Detected Unicode script.
- Individual token pieces and byte fragments.
- Indic-script transliteration to ITRANS when supported.

Script detection reports the Unicode script, not the language. For example,
Devanagari text could be Hindi or Marathi.

### Fertility benchmark

The benchmark uses a curated set of 10 aligned English and Indian-language
prompts. It reports:

- Approximate Unicode word count.
- Character count.
- Tokens per tokenizer.
- Tokens per word, or fertility.
- Measured multiplier compared with English.

Supported encodings are:

| Name | Encoding |
|---|---|
| GPT-4 benchmark | `cl100k_base` |
| GPT-4o benchmark | `o200k_base` |
| GPT-3 benchmark | `p50k_base` |

These are local tokenizer measurements, not provider cost or API usage estimates.

### Transliteration pipeline

The transliteration feature converts supported Indic text to ITRANS and compares
token counts using `cl100k_base`.

It does not translate meaning. Urdu is unsupported by the installed script
mapping. Round-trip character similarity and exact round trips are diagnostics;
they do not prove semantic equivalence.

### SentencePiece

The dashboard trains a separate SentencePiece BPE tokenizer from 8 questions per
language in the source corpus. The remaining 2 questions per language are used
as held-out samples.

The separate tokenizer reports:

- Vocabulary size.
- Piece count.
- `cl100k_base` token count for comparison.
- Exact round-trip reconstruction.
- Percentage count change.

SentencePiece pieces are not GPT or Gemini tokens. This tokenizer is not merged
into a model vocabulary and does not establish answer quality or production
token savings.

### Native-language sentence optimizer

The optimizer provides conservative Unicode and whitespace cleanup for
Kannada, Tamil, Telugu, Malayalam, Hindi, Marathi, Bengali, Gujarati, Punjabi,
Odia, Urdu, and English.

Supported selected encodings are:

- `cl100k_base`
- `o200k_base`
- `p50k_base`

The optimizer:

- Applies NFC normalization.
- Cleans selected whitespace and formatting issues.
- Preserves orthographic ZWJ and ZWNJ characters.
- Finds adjacent repeated-word candidates.
- Keeps repeated words unless the user confirms removal.
- Rejects a result when the selected encoding increases its token count.

Whitespace changes and token-count reductions do not prove that meaning is
preserved. Review proposed changes before using them.

## Installation

### Windows

Open PowerShell in the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r token_tax/requirements.txt
```

If PowerShell does not allow activation, use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r token_tax/requirements.txt
```

Python 3.10 or later is recommended.

## Run the dashboard

Activate the virtual environment, then run:

```powershell
python token_tax/run_all.py dashboard
```

On the first dashboard load, the application loads the benchmark,
transliteration, and SentencePiece data. SentencePiece training may take a short
time.

Open <http://127.0.0.1:5000/>.

The dashboard requires internet access for its CDN-hosted Chart.js library.
Tokenizer inspection and local experiments run on the machine.

## Command-line experiments

Run an individual experiment:

```powershell
python token_tax/run_all.py bench
python token_tax/run_all.py pipeline
python token_tax/run_all.py vocab
python token_tax/run_all.py native-noise
```

Run the standard benchmark, transliteration, and vocabulary experiments:

```powershell
python token_tax/run_all.py all
```

The `native-noise` experiment is separate and is not included in `all`.

List commands:

```powershell
python token_tax/run_all.py --help
```

## HTTP API

The Flask dashboard exposes these routes:

| Method and route | Purpose |
|---|---|
| `GET /api/data` | Return benchmark, transliteration, and vocabulary data |
| `POST /api/inspect` | Measure tokenization, word units, script, and transliteration |
| `POST /api/vocab/inspect` | Evaluate custom text with the trained SentencePiece tokenizer |
| `POST /api/native-optimize` | Clean text and report measured token changes |

Example native-optimizer request:

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

Supported optimizer language codes:

```text
kn  Kannada
ta  Tamil
te  Telugu
ml  Malayalam
hi  Hindi
mr  Marathi
bn  Bengali
gu  Gujarati
pa  Punjabi
or  Odia
ur  Urdu
en  English
```

The optimizer limit is 10,000 characters. Custom SentencePiece and general
inspection endpoints have separate input limits.

If repeated-word candidates are returned, review them before confirming removal.
Send the original input again and include the reviewed candidate IDs in
`confirmed_duplicate_ids`.

## Project structure

```text
token_tax/
  run_all.py                       # Command-line entry point
  1_fertility_benchmarker.py       # Token fertility benchmark
  2_transliteration_pipeline.py    # Indic transliteration and round-trip checks
  3_vocab_expansion_demo.py        # Separate SentencePiece tokenizer experiment
  4_dashboard.py                   # Flask dashboard and JSON API
  native_language_optimizer.py     # Native text cleanup and token measurement
  native_noise_experiment.py       # Synthetic native-noise experiment
  indian_language_questions.json   # Shared source corpus
  benchmark_results.json           # Saved benchmark measurements
  pipeline_results.json            # Saved transliteration measurements
  vocab_expansion_results.json     # Saved SentencePiece measurements
  native_noise_dataset.json        # Generated clean and noisy variants
  native_noise_results.json        # Saved native-noise measurements
  requirements.txt                 # Python dependencies
  README.md                        # Detailed Token Tax documentation
  tests/                           # Unit and dashboard/API tests
```

## Source corpus

The source corpus is
[token_tax/indian_language_questions.json](token_tax/indian_language_questions.json).

It contains 10 curated prompts per language across:

- English
- Hindi
- Tamil
- Kannada
- Telugu
- Bengali
- Malayalam
- Gujarati
- Punjabi
- Odia
- Urdu
- Marathi

The corpus is a curated draft dataset, not a naturally sampled or
native-speaker-reviewed corpus.

## Generated result files

Running the standard experiments creates or updates:

- [token_tax/benchmark_results.json](token_tax/benchmark_results.json)
- [token_tax/pipeline_results.json](token_tax/pipeline_results.json)
- [token_tax/vocab_expansion_results.json](token_tax/vocab_expansion_results.json)

Running the native-noise experiment creates or updates:

- [token_tax/native_noise_dataset.json](token_tax/native_noise_dataset.json)
- [token_tax/native_noise_results.json](token_tax/native_noise_results.json)

The dashboard calculates current local measurements when it loads. It does not
depend on all saved JSON result files.

## Tests

Run all Token Tax tests from the repository root:

```powershell
python -m unittest discover -s token_tax/tests
```

The tests cover tokenizer experiments, optimizer safety and measurements,
native-noise generation, and dashboard API behavior.

## Dependencies

The Token Tax dependencies are:

```text
tiktoken
sentencepiece
indic-transliteration
flask
```

Install them with:

```powershell
python -m pip install -r token_tax/requirements.txt
```

## Interpretation and limitations

- Word counts use Unicode letter/number runs, not morphological segmentation.
- Synthetic noise is deterministic and is not collected from real user input.
- Transliteration round trips do not establish semantic accuracy.
- Token counts do not establish API cost or model quality.
- SentencePiece results describe a separate experimental tokenizer.
- Results depend on the corpus, library versions, and tokenizer encodings.
- The dashboard should not be publicly deployed without authentication,
  production hardening, and additional input validation.

## Troubleshooting

### `ModuleNotFoundError`

Activate the virtual environment and install the Token Tax dependencies:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r token_tax/requirements.txt
```

### Port 5000 is already in use

Stop the other local Flask server or change the port in the `main()` function of
[token_tax/4_dashboard.py](token_tax/4_dashboard.py).

### Dashboard charts are missing

Chart.js is loaded from jsDelivr. Check browser internet access and the browser
console. Live token inspection and API processing do not depend on Chart.js.
