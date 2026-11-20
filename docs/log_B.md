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

- Downloaded the official FLORES-200 devtest split from the public NLLB archive
  (CC BY-SA 4.0; cite the NLLB Team, 2022). Kannada, Hindi, and Tamil each
  contain 1,012 aligned rows.
- Downloaded Belebele Kannada, Hindi, and Tamil: 900 records per language.
- Generated native and ISO-Romanized files for both datasets. All 12 output
  files passed row-count and non-empty-field checks.
- Added Tamil-specific ISO 15919 transliteration after the initial spot check
  exposed poor Tamil output from the Sanskrit transliterator.
- Measured normalized character round-trip match on all 1,012 FLORES sentences.
  ITRANS/ISO/HK: Kannada 98.92%/99.77%/99.81%, Hindi 98.13%/98.75%/98.76%,
  Tamil 96.66%/99.70%/99.79%. ISO is used for generated Romanized files.
- Measured normalized character round-trip match on all 900 Belebele passages.
  ITRANS/ISO/HK: Kannada 98.89%/99.73%/99.78%, Hindi 98.07%/98.68%/98.69%,
  Tamil 96.64%/99.64%/99.74%.
- Human-reviewed 20 evenly spaced FLORES Romanization samples per language.
  Number tokens were preserved across FLORES and 4,921 Belebele numeric fields
  after normalizing Indic digits.
- NLLB model `facebook/nllb-200-distilled-600M` loaded and an eight-passage
  CPU batch took 26.24 seconds. CUDA is unavailable.

### Still pending

- Human review of pivot translations and final pivot batch timing
- GPU VRAM measurement on a CUDA-enabled machine
