"""
Component 4: Dashboard — Flask web server + rich HTML UI
Includes Live Token Inspector, Fertility Benchmarker, Transliteration Pipeline, and Vocab Expansion.
"""
from flask import Flask, render_template_string, jsonify, request
import sys
import unicodedata
from pathlib import Path
import tiktoken

sys.path.insert(0, str(Path(__file__).parent))

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    HAS_INDIC = True
except ImportError:
    HAS_INDIC = False

import native_language_optimizer as native_optimizer

app = Flask(__name__)
_cache = None
_vocab_component = None

# Preload encoders
enc_cl100k = tiktoken.get_encoding("cl100k_base")
enc_o200k = tiktoken.get_encoding("o200k_base")
enc_p50k = tiktoken.get_encoding("p50k_base")


def detect_script(text: str) -> tuple[str, object]:
    """Detect Indian language script based on Unicode codepoints."""
    for char in text:
        cp = ord(char)
        if 0x0C80 <= cp <= 0x0CFF:
            return "Kannada script (language not inferred)", sanscript.KANNADA if HAS_INDIC else "KANNADA"
        elif 0x0B80 <= cp <= 0x0BFF:
            return "Tamil script (language not inferred)", sanscript.TAMIL if HAS_INDIC else "TAMIL"
        elif 0x0C00 <= cp <= 0x0C7F:
            return "Telugu script (language not inferred)", sanscript.TELUGU if HAS_INDIC else "TELUGU"
        elif 0x0900 <= cp <= 0x097F:
            return "Devanagari script (language not inferred)", sanscript.DEVANAGARI if HAS_INDIC else "DEVANAGARI"
        elif 0x0980 <= cp <= 0x09FF:
            return "Bengali/Assamese script (language not inferred)", sanscript.BENGALI if HAS_INDIC else "BENGALI"
        elif 0x0D00 <= cp <= 0x0D7F:
            return "Malayalam script (language not inferred)", sanscript.MALAYALAM if HAS_INDIC else "MALAYALAM"
        elif 0x0A80 <= cp <= 0x0AFF:
            return "Gujarati script (language not inferred)", sanscript.GUJARATI if HAS_INDIC else "GUJARATI"
        elif 0x0A00 <= cp <= 0x0A7F:
            return "Gurmukhi script (language not inferred)", sanscript.GURMUKHI if HAS_INDIC else "GURMUKHI"
        elif 0x0B00 <= cp <= 0x0B7F:
            return "Odia script (language not inferred)", sanscript.ORIYA if HAS_INDIC else "ORIYA"
        elif 0x0600 <= cp <= 0x06FF:
            return "Arabic script (language not inferred; transliteration unavailable)", None
    return "English / Other", None

def get_tokens_detail(text: str, enc):
    if not text:
        return [], 0
    token_ids = enc.encode(text, disallowed_special=())
    pieces = []
    for tid in token_ids:
        raw_b = enc.decode_single_token_bytes(tid)
        try:
            display_str = raw_b.decode('utf-8')
            is_valid_utf8 = True
        except UnicodeDecodeError:
            display_str = repr(raw_b)[2:-1]  # hex escaped string like \xe0\xb2
            is_valid_utf8 = False
        pieces.append({
            "id": tid,
            "text": display_str,
            "is_valid_utf8": is_valid_utf8,
            "raw_hex": raw_b.hex()
        })
    return pieces, len(token_ids)

def get_data():
    global _cache, _vocab_component
    if _cache is not None:
        return _cache
    import importlib.util

    def load(name, file):
        spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load component {file}.")
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    bench = load("bench", "1_fertility_benchmarker.py")
    pipe  = load("pipe",  "2_transliteration_pipeline.py")
    vocab = load("vocab", "3_vocab_expansion_demo.py")
    _vocab_component = vocab
    _cache = {
        "benchmark": bench.run_benchmark(),
        "pipeline": pipe.run_pipeline(),
        "vocab": vocab.run_vocab_expansion(),
    }
    return _cache

HTML = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Token Tax | Research Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    :root {
      color-scheme: dark;
      --page: #08090a;
      --surface: #141618;
      --surface-muted: #1b1e21;
      --surface-subtle: #25292c;
      --border: #2b2f33;
      --border-strong: #41464b;
      --text: #f5f6f7;
      --text-muted: #c1c5c9;
      --text-subtle: #969ca1;
      --accent: #d8dbde;
      --accent-hover: #ffffff;
      --success: #49bd76;
      --danger: #f16a6a;
      --warning: #d4ad71;
      --shadow: 0 12px 32px rgba(0, 0, 0, .28);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      min-height: 100vh;
      background: radial-gradient(ellipse at 14% 0%, #25282b 0%, #141618 42%, #08090a 100%) fixed;
      color: var(--text);
      font-family: Inter, "Segoe UI", Roboto, "Noto Sans", Arial, sans-serif;
      font-size: 15px;
      line-height: 1.55;
      text-rendering: optimizeLegibility;
    }
    header {
      padding: 24px max(28px, calc((100% - 1320px) / 2));
      background: linear-gradient(110deg, #151719 0%, #1c1f22 52%, #121416 100%);
      border-bottom: 1px solid var(--border);
    }
    h1 { font-size: 1.55rem; font-weight: 700; letter-spacing: -0.035em; line-height: 1.25; }
    h1 span { color: #c1c5c9; font-weight: 500; }
    .subtitle { color: var(--text-muted); font-size: 0.88rem; margin-top: 6px; }
    .container { max-width: 1320px; margin: 0 auto; padding: 28px 32px 48px; }
    .tabs {
      display: flex;
      gap: 6px;
      margin-bottom: 22px;
      padding: 5px;
      flex-wrap: wrap;
      background: linear-gradient(180deg, #1d2023 0%, #121416 100%);
      border: 1px solid var(--border);
      border-radius: 10px;
    }
    .tab {
      padding: 8px 15px;
      border: 1px solid transparent;
      border-radius: 7px;
      background: transparent;
      color: var(--text-muted);
      cursor: pointer;
      font: inherit;
      font-size: 0.86rem;
      font-weight: 600;
      transition: background .15s ease, color .15s ease, box-shadow .15s ease;
    }
    .tab:hover { background: #25292c; color: var(--text); }
    .tab.active {
      background: linear-gradient(145deg, #3b4044 0%, #25292c 100%);
      border-color: #53595e;
      box-shadow: var(--shadow);
      color: #ffffff;
    }
    .panel { display: none; }
    .panel.active { display: block; }
    .card {
      margin-bottom: 18px;
      padding: 22px;
      background: linear-gradient(145deg, #1b1e21 0%, #121416 72%, #0e1011 100%);
      border: 1px solid #2e3236;
      border-radius: 12px;
      box-shadow: var(--shadow);
    }
    .card h3 {
      margin-bottom: 12px;
      color: #e1e4e6;
      font-size: 0.84rem;
      font-weight: 700;
      letter-spacing: .045em;
      text-transform: uppercase;
    }
    .grid2 { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }
    .grid4 { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 18px; }
    .stat-box {
      min-width: 0;
      padding: 15px;
      background: linear-gradient(145deg, #202326 0%, #17191b 100%);
      border: 1px solid #303438;
      border-radius: 9px;
    }
    .stat-label { color: var(--text-muted); font-size: .72rem; font-weight: 700; letter-spacing: .055em; text-transform: uppercase; }
    .stat-val {
      margin-top: 5px;
      color: var(--text);
      font-family: "IBM Plex Mono", "SFMono-Regular", Consolas, monospace;
      font-size: 1.7rem;
      font-weight: 650;
      font-variant-numeric: tabular-nums;
    }
    .stat-sub { margin-top: 4px; color: var(--text-subtle); font-size: .76rem; }
    textarea {
      width: 100%;
      min-height: 110px;
      padding: 13px 14px;
      border: 1px solid #41464b;
      border-radius: 8px;
      outline: none;
      background: #0e1012;
      color: #f5f6f7;
      font: inherit;
      line-height: 1.6;
      resize: vertical;
    }
    textarea::placeholder { color: #858b90; }
    textarea:focus, select:focus {
      border-color: #4cbd77;
      box-shadow: 0 0 0 3px rgba(73, 189, 118, .16);
    }
    label { color: var(--text-muted); font-size: .84rem; font-weight: 600; }
    .presets { display: flex; align-items: center; gap: 8px; margin: 10px 0 16px; flex-wrap: wrap; }
    .preset-btn {
      padding: 6px 11px;
      border: 1px solid #41464b;
      border-radius: 7px;
      background: linear-gradient(180deg, #2a2e31 0%, #1c1f22 100%);
      color: #f1f2f3;
      cursor: pointer;
      font: inherit;
      font-size: .78rem;
      font-weight: 600;
      transition: background .15s ease, border-color .15s ease;
    }
    .preset-btn:hover { background: #353a3e; border-color: #62686d; }
    .preset-btn.primary-action { background: linear-gradient(135deg, #2e8751, #236640); border-color: #45a96a; color: #fff; }
    .preset-btn.primary-action:hover { background: linear-gradient(135deg, #359a5e, #2b774a); }
    .token-chips {
      display: flex;
      max-height: 280px;
      gap: 6px;
      overflow-y: auto;
      flex-wrap: wrap;
      padding: 14px;
      background: linear-gradient(180deg, #0d0f11 0%, #121416 100%);
      border: 1px solid #303438;
      border-radius: 9px;
    }
    .token-chip {
      display: inline-flex;
      align-items: center;
      padding: 4px 8px;
      border: 1px solid #3a3f43;
      border-radius: 5px;
      font-family: "IBM Plex Mono", "SFMono-Regular", Consolas, monospace;
      font-size: .78rem;
      font-weight: 600;
      cursor: default;
    }
    .chip-utf8 { background: #292d30; color: #f1f3f4; border-color: #41464a; }
    .chip-byte { background: #3b2022; color: #ffb1b1; border-color: #75383c; }
    .chip-latin { background: #232629; color: #d1d4d7; border-color: #393d41; }
    .bar-row { display: flex; align-items: center; gap: 10px; margin-bottom: 9px; }
    .bar-label { width: 90px; color: var(--text-muted); font-size: .78rem; text-align: right; }
    .bar-track { flex: 1; height: 8px; overflow: hidden; border-radius: 4px; background: #2c3034; }
    .bar-fill { height: 100%; border-radius: 4px; transition: width .4s ease; }
    .bar-val { width: 48px; color: var(--text); font-family: "IBM Plex Mono", Consolas, monospace; font-size: .75rem; font-weight: 700; }
    .pill-green, .pill-red, .pill-orange {
      display: inline-block;
      padding: 2px 8px;
      border: 1px solid #41464b;
      border-radius: 12px;
      font-size: .72rem;
      font-weight: 700;
    }
    .pill-green { background: rgba(49, 151, 87, .17); border-color: rgba(73, 189, 118, .38); color: #73d895; }
    .pill-red { background: rgba(190, 62, 62, .18); border-color: rgba(241, 106, 106, .38); color: #ff9090; }
    .pill-orange { background: rgba(180, 145, 88, .14); border-color: rgba(212, 173, 113, .32); color: #e0c18c; }
    table { width: 100%; border-collapse: collapse; font-size: .82rem; }
    th { padding: 9px 12px; border-bottom: 1px solid #3d4246; color: #c5c9cc; font-size: .74rem; font-weight: 700; text-align: left; text-transform: uppercase; }
    td { padding: 9px 12px; border-bottom: 1px solid #292d30; color: #e3e5e7; }
    tr:hover td { background: #202326; }
    .tok-box {
      margin-top: 8px;
      padding: 12px;
      border: 1px solid #303438;
      border-radius: 8px;
      background: linear-gradient(180deg, #0e1012 0%, #151719 100%);
      color: #f0f2f3;
      font-family: "IBM Plex Mono", "SFMono-Regular", Consolas, monospace;
      font-size: .79rem;
      line-height: 1.65;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    .savings-big { color: var(--success); font-size: 2.35rem; font-weight: 700; font-variant-numeric: tabular-nums; }
    .lang-select {
      padding: 7px 11px;
      border: 1px solid #41464b;
      border-radius: 7px;
      background: #17191b;
      color: #f1f2f3;
      cursor: pointer;
      font: inherit;
      font-size: .83rem;
    }
    .segmented { display: inline-flex; gap: 3px; padding: 3px; border: 1px solid #34383c; border-radius: 8px; background: #17191b; }
    .seg-btn { padding: 5px 12px; border: 0; border-radius: 5px; background: transparent; color: var(--text-muted); cursor: pointer; font: inherit; font-size: .76rem; font-weight: 600; }
    .seg-btn.active { background: #41464b; color: #fff; }
    .positive { color: var(--success) !important; }
    .negative { color: var(--danger) !important; }
    .status-success { color: #73d895 !important; }
    .status-warning { color: #e0c18c !important; }
    .status-danger { color: #ff9090 !important; }
    .status-neutral { color: var(--text-muted) !important; }
    button:focus-visible, select:focus-visible, textarea:focus-visible { outline: 2px solid #52bd78; outline-offset: 2px; }
    @media (max-width: 900px) {
      .grid4 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .container { padding: 22px 20px 36px; }
    }
    @media (max-width: 640px) {
      header { padding: 20px; }
      h1 { font-size: 1.35rem; }
      h1 span { display: block; margin-top: 2px; font-size: 1rem; }
      .container { padding: 18px 14px 28px; }
      .tabs { gap: 4px; }
      .tab { flex: 1 1 auto; padding: 8px 10px; }
      .grid2, .grid4 { grid-template-columns: 1fr; }
      .card { padding: 17px; }
      .bar-label { width: 72px; }
      .segmented { flex-wrap: wrap; }
    }
  </style>
</head>
<body>
<header>
  <h1>Token Tax <span>Research dashboard</span></h1>
  <div class="subtitle">Tokenizer benchmarks, script transliteration, SentencePiece, and native-language text cleanup</div>
</header>
<div class="container">
  <div class="tabs">
    <button class="tab active" onclick="show('live',this)">Live inspector</button>
    <button class="tab" onclick="show('fertility',this)">Fertility benchmark</button>
    <button class="tab" onclick="show('pipeline',this)">Transliteration</button>
    <button class="tab" onclick="show('vocab',this)">SentencePiece</button>
    <button class="tab" onclick="show('native',this)">Sentence optimizer</button>
  </div>

  <!-- PANEL 0: LIVE INSPECTOR -->
  <div id="p-live" class="panel active">
    <div class="card">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
        <h3>Input Text (Any Indian Language or English)</h3>
        <span id="detected-script" class="pill-orange">Detecting script…</span>
      </div>
      <textarea id="live-input" placeholder="Type or paste any text in Kannada, Tamil, Hindi, Telugu, English, etc. to inspect live token consumption..." oninput="handleInput()">ವಿದ್ಯಾರ್ಥಿ ತರಕಾರಿ ಕೊಳ್ಳಲು ಮಾರುಕಟ್ಟೆಗೆ ಹೋಗಿ ಮನೆಗೆ ಮರಳಿದನು.</textarea>
      
      <div class="presets">
        <span style="font-size:0.75rem; color:var(--text-muted); font-weight:600;">Presets:</span>
        <button class="preset-btn" onclick="setPreset('ವಿದ್ಯಾರ್ಥಿ ತರಕಾರಿ ಕೊಳ್ಳಲು ಮಾರುಕಟ್ಟೆಗೆ ಹೋಗಿ ಮನೆಗೆ ಮರಳಿದನು.')">Kannada</button>
        <button class="preset-btn" onclick="setPreset('மாணவன் காய்கறிகள் வாங்க சந்தைக்கு சென்று வீடு திரும்பினான்.')">Tamil</button>
        <button class="preset-btn" onclick="setPreset('విద్యార్థి కూరగాయలు కొనడానికి మార్కెట్‌కు వెళ్ళి ఇంటికి తిరిగి వచ్చాడు.')">Telugu</button>
        <button class="preset-btn" onclick="setPreset('छात्र सब्जियां खरीदने बाजार गया और घर वापस आया।')">Hindi</button>
        <button class="preset-btn" onclick="setPreset('The student went to the market to buy vegetables and came back home.')">English</button>
      </div>
    </div>

    <!-- Live Stat Cards -->
    <div class="grid4">
      <div class="stat-box">
        <div class="stat-label">cl100k (GPT-4)</div>
        <div class="stat-val" id="val-cl100k">—</div>
        <div class="stat-sub" id="sub-cl100k">Fertility: - tok/word</div>
      </div>
      <div class="stat-box">
        <div class="stat-label">o200k (GPT-4o)</div>
        <div class="stat-val" id="val-o200k">—</div>
        <div class="stat-sub" id="sub-o200k">Reduction: -</div>
      </div>
      <div class="stat-box">
        <div class="stat-label">Transliterated (Latin)</div>
        <div class="stat-val" id="val-latin">—</div>
        <div class="stat-sub" id="sub-latin">Savings: -%</div>
      </div>
      <div class="stat-box">
        <div class="stat-label">Words / Characters</div>
        <div class="stat-val" id="val-words">—</div>
        <div class="stat-sub" id="sub-words">- characters</div>
      </div>
    </div>

    <!-- Cost and Token Breakdown -->
    <div class="grid2">
      <div class="card">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
          <h3>Visual Token Breakdown (What the model sees)</h3>
          <div class="segmented">
            <button class="seg-btn active" id="btn-seg-cl100k" onclick="switchBreakdown('cl100k')">cl100k (GPT-4)</button>
            <button class="seg-btn" id="btn-seg-o200k" onclick="switchBreakdown('o200k')">o200k (GPT-4o)</button>
          </div>
        </div>
        <div style="font-size:0.75rem; color:var(--text-muted); margin-bottom:8px;">
          <span style="color:var(--success)">■ Full characters / words</span> &nbsp;|&nbsp;
          <span style="color:var(--danger)">■ Fragmented byte slices (the token tax)</span>
        </div>
        <div id="token-chips-container" class="token-chips"></div>
        <div id="token-summary-text" style="font-size:0.75rem; color:var(--text-muted); margin-top:8px;"></div>
      </div>

      <div class="card">
        <h3>Measured Counts & Cost Notes</h3>
        <div id="cost-analysis" style="font-size:0.82rem; line-height:1.8; color:var(--text-muted);"></div>
        
        <h3 style="margin-top:16px;">Transliterated String (Latin)</h3>
        <div id="latin-output" class="tok-box"></div>
      </div>
    </div>
  </div>

  <!-- PANEL 1: FERTILITY -->
  <div id="p-fertility" class="panel">
    <div class="grid2">
      <div class="card">
        <h3>Token Multiplier vs English — cl100k (GPT-4)</h3>
        <div id="bars-cl100k"></div>
      </div>
      <div class="card">
        <h3>Multiplier Comparison — All Tokenizers</h3>
        <canvas id="chart-compare" height="220"></canvas>
      </div>
    </div>
    <div class="card" style="margin-top:20px">
      <h3>Full Fertility Table</h3>
      <table>
        <thead><tr><th>Language</th><th>Words</th>
          <th>cl100k Tokens</th><th>cl100k Fertility</th><th>cl100k ×Eng</th>
          <th>o200k Tokens</th><th>o200k ×Eng</th>
        </tr></thead>
        <tbody id="fertility-table"></tbody>
      </table>
    </div>
  </div>

  <!-- PANEL 2: PIPELINE -->
  <div id="p-pipeline" class="panel">
    <div class="grid2" style="margin-bottom:20px">
      <div class="card" style="text-align:center">
        <h3>Measured Token Count Change</h3>
        <div class="savings-big" id="avg-savings">—</div>
        <div style="color:var(--text-muted);font-size:0.8rem;margin-top:4px">supported samples only (cl100k)</div>
      </div>
      <div class="card">
        <h3>Select Language</h3>
        <select class="lang-select" id="pipeline-lang" onchange="renderPipeline()">
        </select>
        <div id="pipeline-detail" style="margin-top:14px"></div>
      </div>
    </div>
    <div class="card">
      <h3>Pipeline Savings per Language</h3>
      <canvas id="chart-savings" height="180"></canvas>
    </div>
  </div>

  <!-- PANEL 3: VOCAB -->
  <div id="p-vocab" class="panel">
    <div class="card">
      <h3>Evaluate custom text with the trained SentencePiece tokenizer</h3>
      <p style="color:var(--text-muted);line-height:1.6;font-size:0.82rem;margin-bottom:12px">
        Input is tokenized locally by this separately trained tokenizer. No text is sent to an AI service.
        Piece counts are not GPT/Gemini token counts and do not establish equivalent model behavior.
      </p>
      <textarea id="vocab-custom-input" maxlength="100000" placeholder="Enter text in any supported language or script..."></textarea>
      <div class="presets">
        <button class="preset-btn primary-action" onclick="evaluateVocabCustom()">Evaluate text</button>
        <span id="vocab-custom-status" style="font-size:0.8rem;color:var(--text-muted)"></span>
      </div>
      <pre id="vocab-custom-result" class="tok-box" aria-live="polite"></pre>
    </div>
    <div class="grid2">
      <div class="card">
        <h3>Held-out token counts — SentencePiece vs cl100k</h3>
        <canvas id="chart-vocab" height="220"></canvas>
      </div>
      <div class="card">
        <h3>Sample Tokenization (SentencePiece)</h3>
        <select class="lang-select" id="vocab-lang" onchange="renderVocabDetail()"></select>
        <div id="vocab-detail" style="margin-top:12px"></div>
      </div>
    </div>
    <div class="card" style="margin-top:20px">
      <h3>Vocabulary Experiment — Held-out Results</h3>
      <table>
        <thead><tr><th>Language</th><th>Held-out cl100k Tokens</th><th>SentencePiece Pieces</th><th>Count Change</th><th>Exact Round Trips</th></tr></thead>
        <tbody id="vocab-table"></tbody>
      </table>
    </div>
  </div>

  <!-- PANEL 4: NATIVE LANGUAGE OPTIMIZATION -->
  <div id="p-native" class="panel">
    <div class="card">
      <h3>Native-language sentence optimizer</h3>
      <p style="color:var(--text-muted);line-height:1.6;font-size:0.82rem;margin-bottom:12px">
        Local NFC, whitespace, and safe edge-format cleanup only. Repeated words are suggestions:
        they are kept unless you explicitly confirm each one is accidental. No translation or
        transliteration is performed, and text comparison does not prove semantic equivalence.
      </p>
      <label for="native-optimizer-input">Sentence or short text</label>
      <textarea id="native-optimizer-input" maxlength="10000" placeholder="Type or paste text in Kannada, Tamil, Telugu, Malayalam, Hindi, Marathi, Bengali, Gujarati, Punjabi, Odia, Urdu, or English..."></textarea>
      <div class="presets">
        <label for="native-optimizer-language">Language</label>
        <select id="native-optimizer-language" class="lang-select">
          <option value="kn">Kannada</option>
          <option value="ta">Tamil</option>
          <option value="te">Telugu</option>
          <option value="ml">Malayalam</option>
          <option value="hi">Hindi</option>
          <option value="mr">Marathi</option>
          <option value="bn">Bengali</option>
          <option value="gu">Gujarati</option>
          <option value="pa">Punjabi</option>
          <option value="or">Odia</option>
          <option value="ur">Urdu</option>
          <option value="en">English</option>
        </select>
        <label for="native-optimizer-encoding">Tokenizer</label>
        <select id="native-optimizer-encoding" class="lang-select">
          <option value="cl100k_base">cl100k_base</option>
          <option value="o200k_base">o200k_base</option>
          <option value="p50k_base">p50k_base</option>
        </select>
        <button class="preset-btn primary-action" id="native-optimize-button" onclick="optimizeNativeInput()">Optimize</button>
      </div>
      <div id="native-optimizer-status" class="status-neutral" role="status" style="font-size:0.84rem;margin-bottom:12px"></div>
      <div class="grid2">
        <div><h3>Original (removed text highlighted below)</h3><pre id="native-original-output" class="tok-box"></pre></div>
        <div><h3>Optimized</h3><pre id="native-optimized-output" class="tok-box"></pre></div>
      </div>
      <div class="grid4" style="margin-top:12px">
        <div class="stat-box"><div class="stat-label">Original tokens</div><div class="stat-val" id="native-original-tokens">—</div></div>
        <div class="stat-box"><div class="stat-label">Optimized tokens</div><div class="stat-val" id="native-optimized-tokens">—</div></div>
        <div class="stat-box"><div class="stat-label">Tokens saved</div><div class="stat-val" id="native-tokens-saved">—</div></div>
        <div class="stat-box"><div class="stat-label">Savings</div><div class="stat-val" id="native-savings-percent">—</div></div>
      </div>
      <div id="native-duplicate-candidates" style="margin:12px 0"></div>
      <div id="native-optimizer-changes" style="font-size:0.82rem;line-height:1.7"></div>
    </div>

  </div>
</div>

<script>
const DATA = {{ data | tojson }};
let currentInspect = null;
let currentBreakdown = 'cl100k';
let debounceTimer = null;
let nativeOptimizerCandidateSource = null;

function show(panel, btn) {
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
  document.getElementById('p-' + panel).classList.add('active');
  btn.classList.add('active');
}

function setPreset(txt) {
  document.getElementById('live-input').value = txt;
  inspectLive();
}

function handleInput() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(inspectLive, 200);
}

async function inspectLive() {
  const text = document.getElementById('live-input').value.trim();
  if (!text) {
    document.getElementById('val-cl100k').textContent = '0';
    document.getElementById('val-o200k').textContent = '0';
    document.getElementById('val-latin').textContent = '0';
    document.getElementById('val-words').textContent = '0';
    document.getElementById('token-chips-container').innerHTML = '<span style="color:var(--text-muted)">Type something above...</span>';
    return;
  }

  try {
    const res = await fetch('/api/inspect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });
    const d = await res.json();
    currentInspect = d;
    renderInspect();
  } catch (err) {
    console.error(err);
  }
}

function switchBreakdown(mode) {
  currentBreakdown = mode;
  document.getElementById('btn-seg-cl100k').classList.toggle('active', mode === 'cl100k');
  document.getElementById('btn-seg-o200k').classList.toggle('active', mode === 'o200k');
  renderChips();
}

function renderInspect() {
  if (!currentInspect) return;
  const d = currentInspect;

  document.getElementById('detected-script').textContent = d.detected_script;
  document.getElementById('val-cl100k').textContent = d.cl100k.count;
  document.getElementById('sub-cl100k').textContent = `Fertility: ${d.cl100k.fertility} tok/word`;

  document.getElementById('val-o200k').textContent = d.o200k.count;
  const diffO2 = d.cl100k.count - d.o200k.count;
  const pctO2 = d.cl100k.count > 0 ? ((diffO2 / d.cl100k.count) * 100).toFixed(1) : 0;
  const o200kSummary = document.getElementById('sub-o200k');
  o200kSummary.textContent = diffO2 >= 0 ? `-${pctO2}% vs cl100k` : `+${Math.abs(pctO2)}%`;
  o200kSummary.classList.toggle('positive', diffO2 > 0);
  o200kSummary.classList.toggle('negative', diffO2 < 0);

  document.getElementById('val-latin').textContent = d.latin.count;
  document.getElementById('sub-latin').textContent = `Savings: ${d.latin.savings_pct}%`;

  document.getElementById('val-words').textContent = d.word_count;
  document.getElementById('sub-words').textContent = `${d.char_count} chars · approximate Unicode word units`;

  renderChips();

  document.getElementById('cost-analysis').textContent =
    'Only input token counts are measured here. API cost depends on the selected model, provider prices, prompt, and answer length; no price is assumed.';
  document.getElementById('latin-output').textContent =
    d.latin.available ? d.latin.text : d.latin.reason;
  document.getElementById('val-latin').textContent =
    d.latin.available ? d.latin.count : 'N/A';
  document.getElementById('sub-latin').textContent =
    d.latin.available ? `Measured count change: ${d.latin.savings_pct}%` : 'No measured transliteration';
  document.getElementById('sub-latin').classList.toggle(
    'positive', d.latin.available && d.latin.savings_pct > 0
  );
  document.getElementById('sub-latin').classList.toggle(
    'negative', d.latin.available && d.latin.savings_pct < 0
  );
}

function renderChips() {
  if (!currentInspect) return;
  const target = currentBreakdown === 'cl100k' ? currentInspect.cl100k : currentInspect.o200k;
  const container = document.getElementById('token-chips-container');
  container.innerHTML = '';

  target.pieces.forEach((p, idx) => {
    const chip = document.createElement('span');
    chip.className = `token-chip ${p.is_valid_utf8 ? 'chip-utf8' : 'chip-byte'}`;
    chip.title = `Token #${idx+1} | ID: ${p.id} | Hex: ${p.raw_hex}`;
    chip.textContent = p.text === ' ' ? '␣' : p.text;
    container.appendChild(chip);
  });

  const byteFragments = target.pieces.filter(p => !p.is_valid_utf8).length;
  document.getElementById('token-summary-text').innerHTML = `
    Showing <b>${target.pieces.length} tokens</b> in <b>${currentBreakdown}</b>.
    ${byteFragments > 0 ? `<span style="color:var(--danger)">Warning: ${byteFragments} tokens are fragmented raw byte slices</span>` : `<span style="color:var(--success)">All tokens are valid UTF-8 characters</span>`}
  `;
}

// ---- FERTILITY ----
function renderFertility() {
  const bench = DATA.benchmark;
  const langs = Object.keys(bench);
  const tokName = "cl100k (GPT-4)";

  // Bars
  const barsDiv = document.getElementById('bars-cl100k');
  barsDiv.innerHTML = '';
  const maxMult = Math.max(...langs.map(l => bench[l].tokenizers[tokName]?.multiplier_vs_english || 1));

  langs.forEach((lang, i) => {
    const mult = bench[lang].tokenizers[tokName]?.multiplier_vs_english || 1;
    const pct = (mult / maxMult * 100).toFixed(1);
    const color = '#858a8f';
    barsDiv.innerHTML += `
      <div class="bar-row">
        <div class="bar-label">${lang}</div>
        <div class="bar-track"><div class="bar-fill" style="width:${pct}%;background:${color}"></div></div>
        <div class="bar-val" style="color:${color}">${mult}×</div>
      </div>`;
  });

  // Compare chart
  const tok2Name = "o200k (GPT-4o)";
  const labels = langs;
  const cl100kData = langs.map(l => bench[l].tokenizers[tokName]?.multiplier_vs_english || 0);
  const o200kData  = langs.map(l => bench[l].tokenizers[tok2Name]?.multiplier_vs_english || 0);

  new Chart(document.getElementById('chart-compare'), {
    type: 'bar',
    data: { labels, datasets: [
      { label: 'cl100k', data: cl100kData, backgroundColor: '#83898ecc' },
      { label: 'o200k',  data: o200kData,  backgroundColor: '#c4c8cbcc' },
    ]},
    options: { responsive: true, plugins: { legend: { labels: { color:'#e4e6e8' } } },
      scales: {
        x: { ticks: { color:'#c1c5c9' }, grid: { color:'#383c40' } },
        y: { ticks: { color:'#c1c5c9' }, grid: { color:'#383c40' }, title: { display:true, text:'× English', color:'#c1c5c9' } }
      }
    }
  });

  // Table
  const tbody = document.getElementById('fertility-table');
  langs.forEach(lang => {
    const d = bench[lang];
    const cl = d.tokenizers[tokName] || {};
    const o2 = d.tokenizers[tok2Name] || {};
    const mult = cl.multiplier_vs_english || 1;
    const badge = mult >= 8 ? `<span class="pill-red">${mult}×</span>` :
                  mult >= 4 ? `<span class="pill-orange">${mult}×</span>` :
                              `<span class="pill-green">${mult}×</span>`;
    tbody.innerHTML += `<tr>
      <td><b>${lang}</b></td>
      <td>${d.word_count}</td>
      <td>${cl.token_count || '—'}</td>
      <td>${cl.fertility || '—'}</td>
      <td>${badge}</td>
      <td>${o2.token_count || '—'}</td>
      <td>${o2.multiplier_vs_english || '—'}×</td>
    </tr>`;
  });
}

// ---- PIPELINE ----
function renderPipelineInit() {
  const pipe = DATA.pipeline;
  const langs = Object.keys(pipe);
  const sel = document.getElementById('pipeline-lang');
  langs.forEach(l => { const o = document.createElement('option'); o.value=l; o.textContent=l; sel.appendChild(o); });

  const measured = langs.filter(l => pipe[l].status === 'measured');
  const originalTotal = measured.reduce((sum, l) => sum + pipe[l].original_tokens, 0);
  const transliteratedTotal = measured.reduce((sum, l) => sum + pipe[l].transliterated_tokens, 0);
  const change = originalTotal
    ? ((1 - transliteratedTotal / originalTotal) * 100).toFixed(1)
    : 'N/A';
  const averageSavings = document.getElementById('avg-savings');
  averageSavings.textContent = change + '%';
  averageSavings.classList.toggle('positive', Number(change) > 0);
  averageSavings.classList.toggle('negative', Number(change) < 0);

  new Chart(document.getElementById('chart-savings'), {
    type: 'bar',
    data: { labels: langs, datasets: [{
      label: 'Measured token count reduction (%)',
      data: langs.map(l => pipe[l].status === 'measured' ? pipe[l].savings_percent : null),
      backgroundColor: langs.map(language => {
        const saving = pipe[language].savings_percent;
        if (pipe[language].status !== 'measured') return '#555a5f';
        return saving > 0 ? '#3c9b60' : saving < 0 ? '#d95454' : '#858a8f';
      }),
    }]},
    options: { responsive: true, plugins: { legend: { display:false } },
      scales: {
        x: { ticks:{color:'#c1c5c9'}, grid:{color:'#383c40'} },
        y: { ticks:{color:'#c1c5c9'}, grid:{color:'#383c40'},
             title:{display:true,text:'Measured reduction (%)',color:'#c1c5c9'} }
      }
    }
  });
  renderPipeline();
}

function renderPipeline() {
  const pipe = DATA.pipeline;
  const lang = document.getElementById('pipeline-lang').value;
  const d = pipe[lang];
  if (!d) return;
  if (d.status !== 'measured') {
    document.getElementById('pipeline-detail').textContent = d.reason;
    return;
  }
  const sample = d.samples[0];
  document.getElementById('pipeline-detail').innerHTML = `
    <div style="font-size:0.78rem;color:var(--text-muted);line-height:1.9">
      <div>${d.sample_count} questions; ${d.original_tokens} original tokens → ${d.transliterated_tokens} ITRANS tokens
        (${d.savings_percent}% measured change).</div>
      <div>Script transliteration is not translation; exact round trips: ${d.exact_round_trips}/${d.sample_count}.
        Mean normalized code-point edit similarity: ${d.mean_round_trip_character_similarity_percent}% (not semantic accuracy).</div>
      <div><b>Original (${sample.original_tokens} tokens)</b>:<br><span style="color:var(--text)">${sample.original}</span></div>
      <div class="tok-box">ITRANS: ${sample.transliterated}\nTokens: ${sample.transliterated_tokens} (${sample.savings_percent}% measured change)</div>
      <div style="margin-top:8px">Round trip: ${sample.round_trip}<br>Exact: ${sample.round_trip_exact}
        · Code-point edit similarity: ${sample.round_trip_character_similarity_percent}%</div>
    </div>`;
}

// ---- VOCAB ----
function renderVocab() {
  const vocab = DATA.vocab.languages;
  const langs = Object.keys(vocab);

  const sel = document.getElementById('vocab-lang');
  langs.forEach(l => { const o=document.createElement('option'); o.value=l; o.textContent=l; sel.appendChild(o); });

  new Chart(document.getElementById('chart-vocab'), {
    type: 'bar',
    data: { labels: langs, datasets: [
      { label: 'Held-out cl100k tokens', data: langs.map(l=>vocab[l].heldout_cl100k_tokens), backgroundColor:'#858a8fcc' },
      { label: 'Held-out SentencePiece pieces', data: langs.map(l=>vocab[l].heldout_sentencepiece_pieces), backgroundColor:'#c4c8cbcc' },
    ]},
    options: { responsive: true, plugins:{legend:{labels:{color:'#e4e6e8'}}},
      scales:{x:{ticks:{color:'#c1c5c9'},grid:{color:'#383c40'}}, y:{ticks:{color:'#c1c5c9'},grid:{color:'#383c40'}}}
    }
  });

  const tbody = document.getElementById('vocab-table');
  langs.forEach(l => {
    const d = vocab[l];
    const change = d.piece_count_change_percent;
    tbody.innerHTML += `<tr>
      <td><b>${l}</b></td>
      <td>${d.heldout_cl100k_tokens}</td>
      <td>${d.heldout_sentencepiece_pieces}</td>
      <td>${change}%</td>
      <td>${d.exact_round_trips}/${d.heldout_sample_count}</td>
    </tr>`;
  });
  renderVocabDetail();
}

function renderVocabDetail() {
  const vocab = DATA.vocab.languages;
  const lang = document.getElementById('vocab-lang').value;
  const d = vocab[lang];
  if (!d) return;
  const sample = d.samples[0];
  document.getElementById('vocab-detail').innerHTML = `
    <div style="font-size:0.78rem;color:var(--text-muted)">
      <div style="margin-bottom:6px">${sample.text}</div>
      <div class="tok-box">SentencePiece: ${JSON.stringify(sample.pieces)}</div>
      <div style="margin-top:8px">Held-out count change: <b>${d.piece_count_change_percent}%</b>
        (${d.heldout_cl100k_tokens} cl100k tokens → ${d.heldout_sentencepiece_pieces} SentencePiece pieces).
        Exact reconstruction: ${d.exact_round_trips}/${d.heldout_sample_count}.</div>
    </div>`;
}

async function evaluateVocabCustom() {
  const text = document.getElementById('vocab-custom-input').value;
  const status = document.getElementById('vocab-custom-status');
  const output = document.getElementById('vocab-custom-result');
  status.textContent = 'Evaluating locally...';
  output.textContent = '';
  try {
    const response = await fetch('/api/vocab/inspect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Evaluation failed.');
    status.textContent = '';
    output.textContent = JSON.stringify({
      sentencepiece_piece_count: result.sentencepiece_piece_count,
      sentencepiece_pieces: result.sentencepiece_pieces,
      cl100k_token_count: result.cl100k_token_count,
      count_change_percent_vs_cl100k_not_production_savings:
        result.piece_count_change_percent_vs_cl100k,
      exact_reconstruction: result.round_trip_exact,
      tokenizer: result.tokenizer,
      answer_quality_evaluated: result.semantic_or_answer_quality_evaluated
    }, null, 2);
  } catch (error) {
    status.textContent = error.message;
  }
}

function addText(container, tagName, text, className) {
  const element = document.createElement(tagName);
  element.textContent = text;
  if (className) element.className = className;
  container.appendChild(element);
  return element;
}

function renderNativeOptimizerCandidates(candidates, sourceText, confirmedIds) {
  const container = document.getElementById('native-duplicate-candidates');
  container.replaceChildren();
  nativeOptimizerCandidateSource = sourceText;
  if (!candidates.length) return;
  addText(container, 'p',
    'Possible adjacent duplicate words are retained by default. Check only those you confirm are accidental, then select Optimize again.',
    '');
  candidates.forEach(candidate => {
    const label = document.createElement('label');
    label.style.display = 'block';
    label.style.marginTop = '6px';
    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.name = 'native-duplicate-candidate';
    checkbox.value = candidate.id;
    checkbox.checked = confirmedIds.includes(candidate.id);
    label.appendChild(checkbox);
    label.appendChild(document.createTextNode(
      ` Confirm accidental duplicate: ${candidate.word} (the second occurrence)`
    ));
    container.appendChild(label);
  });
}

function renderNativeOptimizerChanges(result) {
  const container = document.getElementById('native-optimizer-changes');
  container.replaceChildren();
  result.changes.forEach(change => {
    const row = document.createElement('div');
    if (change.type === 'duplicate_word') {
      row.appendChild(document.createTextNode('Removed confirmed duplicate: '));
      const removed = document.createElement('del');
      removed.textContent = change.removed_text;
      removed.style.color = 'var(--danger)';
      row.appendChild(removed);
    } else if (change.type === 'formatting_whitespace') {
      addText(row, 'strong', 'Normalized formatting whitespace: ', '');
      (change.examples || []).forEach(example => {
        const removed = document.createElement('del');
        removed.textContent = JSON.stringify(example.before);
        removed.style.color = 'var(--danger)';
        row.appendChild(removed);
        row.appendChild(document.createTextNode(` → ${JSON.stringify(example.after)} `));
      });
      if (!(change.examples || []).length) {
        row.appendChild(document.createTextNode(change.description));
      }
    } else {
      row.textContent = change.description;
    }
    container.appendChild(row);
  });
  result.detected_invisible_characters.forEach(item => {
    addText(container, 'div',
      `${item.codepoint} ${item.name}: ${item.action}.`,
      '');
  });
}

async function optimizeNativeInput() {
  const text = document.getElementById('native-optimizer-input').value;
  const status = document.getElementById('native-optimizer-status');
  const confirmedIds = nativeOptimizerCandidateSource === text
    ? Array.from(document.querySelectorAll('input[name="native-duplicate-candidate"]:checked'))
      .map(input => input.value)
    : [];
  status.textContent = 'Optimizing locally...';
  status.className = 'status-neutral';
  try {
    const response = await fetch('/api/native-optimize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text,
        language: document.getElementById('native-optimizer-language').value,
        encoding: document.getElementById('native-optimizer-encoding').value,
        confirmed_duplicate_ids: confirmedIds
      })
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Optimization failed.');
    document.getElementById('native-original-output').textContent = result.original_text;
    document.getElementById('native-optimized-output').textContent = result.optimized_text;
    document.getElementById('native-original-tokens').textContent = result.original_tokens;
    document.getElementById('native-optimized-tokens').textContent = result.optimized_tokens;
    document.getElementById('native-tokens-saved').textContent = result.tokens_saved;
    document.getElementById('native-tokens-saved').classList.toggle('positive', result.tokens_saved > 0);
    document.getElementById('native-tokens-saved').classList.toggle('negative', result.tokens_saved < 0);
    document.getElementById('native-savings-percent').textContent = `${result.savings_percentage}%`;
    renderNativeOptimizerChanges(result);
    renderNativeOptimizerCandidates(result.candidates, text, result.confirmed_duplicate_ids || []);
    const messages = {
      no_safe_optimization: 'No safe optimization was found; the original was retained.',
      review_required: 'Possible repeated words were found and retained pending your confirmation.',
      rejected_token_increase: 'The proposed change increased token count and was rejected; the original was retained.',
      normalized_no_token_savings: 'Safe formatting normalization was applied, but this tokenizer measured no token savings.'
    };
    status.textContent = messages[result.status] ||
      'Optimization measured locally. Text comparison is not proof of semantic equivalence.';
    status.className = result.status === 'optimized'
      ? 'status-success'
      : result.status === 'review_required'
        ? 'status-warning'
        : result.status === 'rejected_token_increase'
          ? 'status-danger'
          : 'status-neutral';
  } catch (error) {
    status.textContent = error.message;
    status.className = 'status-danger';
  }
}

function clearNativeOptimizerOutput() {
  nativeOptimizerCandidateSource = null;
  document.getElementById('native-duplicate-candidates').replaceChildren();
  document.getElementById('native-optimizer-status').textContent = '';
  document.getElementById('native-optimizer-status').className = 'status-neutral';
  document.getElementById('native-original-output').textContent = '';
  document.getElementById('native-optimized-output').textContent = '';
  document.getElementById('native-optimizer-changes').replaceChildren();
  ['native-original-tokens', 'native-optimized-tokens', 'native-tokens-saved',
    'native-savings-percent'].forEach(id => {
      document.getElementById(id).textContent = '—';
    });
  document.getElementById('native-tokens-saved').classList.remove('positive', 'negative');
}

document.getElementById('native-optimizer-input').addEventListener('input', clearNativeOptimizerOutput);
document.getElementById('native-optimizer-language').addEventListener('change', clearNativeOptimizerOutput);
document.getElementById('native-optimizer-encoding').addEventListener('change', clearNativeOptimizerOutput);

// ---- INIT ----
inspectLive();
renderFertility();
renderPipelineInit();
renderVocab();
</script>
</body>
</html>
"""

@app.route("/")
def index():
    data = get_data()
    return render_template_string(HTML, data=data)

@app.route("/api/data")
def api_data():
    return jsonify(get_data())

@app.route("/api/vocab/inspect", methods=["POST"])
def api_vocab_inspect():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
        return jsonify({"error": "Expected a JSON object with a string text field."}), 400
    text = payload["text"]
    if len(text) > 100_000:
        return jsonify({"error": "Text must be no more than 100,000 characters."}), 413
    get_data()
    try:
        result = _vocab_component.evaluate_custom_text(text)
    except (TypeError, ValueError) as error:
        return jsonify({"error": str(error)}), 400
    return jsonify(result)

@app.route("/api/native-optimize", methods=["POST"])
def api_native_optimize():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Expected a JSON object with text, language, and encoding fields."}), 400
    required = ("text", "language", "encoding")
    missing = [field for field in required if field not in payload]
    if missing:
        return jsonify({"error": f"Missing required field(s): {', '.join(missing)}."}), 400
    for field in required:
        if not isinstance(payload[field], str):
            return jsonify({"error": f"The {field} field must be a string."}), 400
    if len(payload["text"]) > native_optimizer.MAX_OPTIMIZER_INPUT_LENGTH:
        return jsonify({
            "error": (
                "Text must be no more than "
                f"{native_optimizer.MAX_OPTIMIZER_INPUT_LENGTH} characters."
            )
        }), 413
    try:
        result = native_optimizer.optimize_native_text(
            payload["text"],
            payload["language"],
            payload["encoding"],
            payload.get("confirmed_duplicate_ids"),
        )
    except (TypeError, ValueError) as error:
        return jsonify({"error": str(error)}), 400
    return jsonify(result)

@app.route("/api/inspect", methods=["POST"])
def api_inspect():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Expected a JSON object with a text field."}), 400
    text = payload.get("text", "")
    if not isinstance(text, str):
        return jsonify({"error": "The text field must be a string."}), 400
    if len(text) > 100_000:
        return jsonify({"error": "Text must be no more than 100,000 characters."}), 413

    word_count = count_unicode_words(text)
    char_count = len(text)
    
    script_name, script_code = detect_script(text)
    
    cl_pieces, cl_count = get_tokens_detail(text, enc_cl100k)
    o2_pieces, o2_count = get_tokens_detail(text, enc_o200k)
    _, p50_count = get_tokens_detail(text, enc_p50k)
    
    latin = {
        "available": False,
        "text": "",
        "count": None,
        "savings_pct": None,
    }
    if not HAS_INDIC:
        latin["reason"] = "Install indic-transliteration to measure script transliteration."
    elif script_code:
        latin_text = transliterate(text, script_code, sanscript.ITRANS)
        _, latin_count = get_tokens_detail(latin_text, enc_cl100k)
        latin.update({
            "available": True,
            "text": latin_text,
            "count": latin_count,
            "savings_pct": round((1 - latin_count / cl_count) * 100, 1)
            if cl_count
            else 0,
        })
    else:
        latin["reason"] = (
            "Arabic-script transliteration is unsupported by the installed library; "
            "the text's language cannot be inferred from script alone."
            if script_name.startswith("Arabic script")
            else "Text is not in a supported Indic script; no transliteration was attempted."
        )

    return jsonify({
        "text": text,
        "word_count": word_count,
        "char_count": char_count,
        "detected_script": script_name,
        "cl100k": {
            "count": cl_count,
            "fertility": round(cl_count / word_count, 2) if word_count else 0,
            "pieces": cl_pieces
        },
        "o200k": {
            "count": o2_count,
            "fertility": round(o2_count / word_count, 2) if word_count else 0,
            "pieces": o2_pieces
        },
        "p50k": {
            "count": p50_count,
            "fertility": round(p50_count / word_count, 2) if word_count else 0
        },
        "latin": latin,
    })


def count_unicode_words(text: str) -> int:
    count = 0
    in_word = False
    for char in text:
        category = unicodedata.category(char)
        if category[0] in ("L", "N"):
            if not in_word:
                count += 1
            in_word = True
        elif category[0] == "M" and in_word:
            continue
        else:
            in_word = False
    return count


def main() -> None:
    print("Starting Token Tax Dashboard at http://127.0.0.1:5000")
    app.run(debug=False, host="127.0.0.1", port=5000)


if __name__ == "__main__":
    main()
