# Local Intelligence Brief

A single-page comparison of local AI hardware, model capability, memory, cost and single-user decode speed. Research snapshot: **10 October 2026**.

- [Read the report](index.html) — open locally in a browser; expand **Sources & calculations** for the evidence ledger and methodology.
- [Download the single-page A3 PDF](local-ai-field-guide.pdf).
- [Research notes as text](data/research-notes.txt).
- [Hardware data](data/hardware.json), [model data](data/models.json), and [source links](data/sources.json).
- [Reproduced arithmetic](data/calculations.json).

## What the ranking means

Choose the highest-scoring fitting model **within the report's shortlist** whose measured or estimated decode **upper bound reaches 35 tokens/second**. Sort devices by model intelligence, then speed-range midpoint, ascending. This is not a guaranteed 35 tok/s minimum or an exhaustive model search.

Intelligence uses Artificial Analysis Intelligence Index v4.3.2. Scores describe the tested model and reasoning setting, not a fresh evaluation of the local quantization. Opus 5.5 and GPT-6 Astra are proprietary comparison ceilings, not local deployment options.

Decode is one active request, warm model, with workload/context differences recorded per source. Model-memory budgets use **131,072 total cached tokens**, concurrency one, decimal GB, and separately list SSD lookup tables and runtime allowances. Speed estimates are not 128K speed forecasts.

## Re-evaluate the report

1. Start with `index.html`: it contains the complete source ledger, benchmark conditions, hardware/pricing references, formulas, and selection notes. It is the authoritative report source.
2. Check the linked original benchmark logs, runtime versions, model artifacts and quantization. Do not transfer a result between models or GPU counts as if it were measured.
3. Consult `data/configs/` for captured model architecture configurations and `data/config-manifest.json` for URLs, UTC capture times and SHA-256 hashes. These snapshots support cache calculations; they do not pin the full weight artifacts or runtime code.
4. Change assumptions in the report and, where relevant, `scripts/calculations.py`. Keep both consistent. Regenerate the derived data and PDF.
5. Validate the 35 tok/s upper-bound rule, ordering, table structure and links with `python scripts/export_report.py`.

### Important unresolved evidence

- Single RTX PRO 6000 / Flash-Next NVFP4: **~95 tok/s, 65–125 scenario range**. Dual PRO / GLM-Flash IQ4_XS: **~80, 60–110**. Both are low-confidence roofline-plus-overhead forecasts; exact serving fits are unverified. The matched hardware anchor is **two PRO cards running a different deployment, Flash-Next FP8, at 81.45 tok/s**. Four cards reached 64.61. Neither is a direct measurement of the headline pairing.
- M5 Pro / Qwen3.8-27B: **21–44 estimated**. The upper end extrapolates a favorable repetitive-editing workload; fresh coding extrapolates to roughly 21–29. It does not establish a general 35 tok/s floor.
- R9700 / Qwen3.8-27B: **30–50 conditional on working AMD MTP**, not a matched R9700 measurement. Without drafting, the 24–32 baseline fails; the smaller-active Qwen3.6 MoE is the fallback.
- Halo's qualifying 36 tok/s peak is near-greedy, short-prompt and version-sensitive. Longer prompts can remain below 35.
- Dual Spark and Station include assumed drafting gains. Their exact GLM deployments and gains are unverified. The forthcoming 512 GB Ultra is a projection.
- Compressed MLA versus expanded per-head KV changes GLM-Flash's 128K cache dramatically. The selected memory budget requires the compressed-cache backend scenario.
- Prices are approximate, hardware availability is snapshot-specific, and Apple Neural Engine TOPS are explicitly rough estimates rather than official whole-chip ratings.

These limitations and all other assumptions are detailed in the research drawer. No in-house hardware benchmark was performed for this report. No raw third-party benchmark logs are represented as our own measurements.

## Rebuild (Python 3; no Python dependencies)

```sh
python scripts/calculations.py
python scripts/export_report.py
```

Capture or refresh architecture configs (requires internet; overwrites snapshot files):

```sh
python scripts/capture_configs.py
```

On Windows, generate the PDF with installed Google Chrome:

```powershell
./scripts/render_pdf.ps1
```

On other platforms, open `index.html` in Chromium/Chrome and print to PDF using its CSS A3 page size, background graphics, and no browser headers/footers. The expandable research drawer is omitted from print; its content remains available in the HTML and `data/research-notes.txt`. Inspect the PDF to confirm it remains one complete, unclipped page after changes.

## Files

`index.html` and the scripts are editable sources. The PDF and `data/hardware.json`, `data/models.json`, `data/sources.json`, `data/research-notes.txt`, and `data/calculations.json` are derived outputs. `data/configs/` contains timestamped upstream JSON snapshots, not model weights. Source pages can change; citations and these snapshots are not a complete offline archive of the web.
