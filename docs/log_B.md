# Member B Experiment Log

## 2026-11-20

- Role: Transformer
- Romanization script created in `transform/romanize.py`.
- Schemes implemented for Kannada, Hindi, and Tamil: ITRANS, ISO, and HK.
- FLORES and Belebele outputs follow `data/transformed/`.
- Pivot implementation in `transform/pivot.py` uses `facebook/nllb-200-distilled-600M`.
- Priority batch command: `python transform/pivot.py --belebele-limit 100`
- Full run command: `python transform/pivot.py`
- Output validation command: `python transform/check_outputs.py`

### Completed locally

- Downloaded Belebele Kannada, Hindi, and Tamil: 900 records per language.
- Generated native and ISO-Romanized files; all six files passed the 900-row and
  non-empty-field checks.
- Compared ITRANS, ISO, and HK round trips on 50 passages per language. Exact
  sentence match rates: Kannada 0%/46%/46%, Hindi 0%/12%/12%, Tamil 0%/46%/46%.
  ISO was used for the generated Romanized data. These are round-trip checks,
  not semantic quality scores.
- FLORES download is blocked because its Hugging Face repository requires
  authenticated access (HTTP 401).
- NLLB pivot inference was attempted on CPU but was too slow to finish the
  priority batch. CUDA is unavailable in this environment.

### Still pending

- Human 20-row spot check per strategy and numbers/names spot check
- Completed pivot dataset and average pivot seconds per sentence
- GPU VRAM measurement on a CUDA-enabled machine
