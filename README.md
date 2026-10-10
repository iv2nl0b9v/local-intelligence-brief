# Local Intelligence Brief

A single-page comparison of local AI hardware, model capability, memory, cost and single-user decode speed. Research snapshot: **10 October 2026**.

**[Open the live report →](https://iv2nl0b9v.github.io/local-intelligence-brief/)**

Hosted with GitHub Pages from the root of `main`. Pushing updates to that branch automatically republishes the site; `.nojekyll` serves the files as plain static content.

- [Read the report](index.html) — open locally in a browser; expand **Sources & calculations** for the evidence ledger and methodology.
- [Download the single-page A3 PDF](local-ai-field-guide.pdf).
- [Research notes as text](data/research-notes.txt).
- [Hardware data](data/hardware.json), [model data](data/models.json), and [source links](data/sources.json).
- [Reproduced arithmetic](data/calculations.json).

## What the ranking means

Choose the highest-scoring fitting model **within the report's shortlist** whose measured or estimated decode **upper bound reaches 35 tokens/second**. Sort devices by model intelligence, then speed-range midpoint, ascending. This is not a guaranteed 35 tok/s minimum or an exhaustive model search.

Intelligence uses Artificial Analysis Intelligence Index v4.3.2. Scores describe the tested model and reasoning setting, not a fresh evaluation of the local quantization. Four Claude Opus releases (4.6 → 4.8 → 5 → 5.5: 32* → 42 → 51 → 58, February to September 2026; *AA estimate) mark the frontier trajectory. They are proprietary references, not local deployment options. The best local-fitting model, GLM-5.3 (45), sits between Opus 4.8 and Opus 5.

Decode is one active request, warm model, with workload/context differences recorded per source. Model-memory budgets use **131,072 total cached tokens**, concurrency one, decimal GB, and separately list SSD lookup tables and runtime allowances. Speed estimates are not 128K speed forecasts.

## Re-evaluate the report

1. Start with `index.html`: it contains the complete source ledger, benchmark conditions, hardware/pricing references, formulas, and selection notes. It is the authoritative report source.
2. Check the linked original benchmark logs, runtime versions, model artifacts and quantization. Do not transfer a result between models or GPU counts as if it were measured.
3. Consult `data/configs/` for captured model architecture configurations and `data/config-manifest.json` for URLs, UTC capture times and SHA-256 hashes. These snapshots support cache calculations; they do not pin the full weight artifacts or runtime code.
4. Change assumptions in the report and, where relevant, `scripts/calculations.py`. Keep both consistent. Regenerate the derived data and PDF.
5. Validate the 35 tok/s upper-bound rule, ordering, table structure, links, and that every displayed estimate range equals its computed endpoints rounded to the nearest integer, with `python scripts/export_report.py`.

### Important unresolved evidence

- Measured rows are third-party or vendor results, labelled **M**, not this report's replication. The R9700's **51.8 tok/s** is AMD's preliminary vendor number (Vulkan, MTP=2, quantization unstated). Halo's **32–47** comes from one community llama.cpp thread whose same-build runs varied 25–36 tok/s across days.
- Four rows remain estimates:
  - **M5 Pro / Qwen3.8-27B: 21–44.** The upper end extrapolates favorable repetitive editing. An independent 2.33× MTP multiplier verified on M5 Pro supports roughly 40, but there is no direct measurement.
  - **M5 Ultra 512 GB:** extrapolated from the measured 256 GB system; the hardware is not yet shipping.
  - **2 × RTX PRO 6000 / GLM-5.3-Flash IQ4_XS: 41–70, best ~55.** Assumes llama.cpp's sequential layer split. Four cards measured 63.7 tok/s with NVFP4, which does not fit on two.
  - **DGX Station / GLM-5.3: 22–45.** Assumes drafting gains on split HBM/CPU memory.
- Dual Spark no longer pairs with GLM-5.3-Flash: measured single-request speed is 14.6–30.8 tok/s, which fails the gate. It now runs Flash-Next, which fits on one node.
- The single RTX PRO 6000 Flash-Next result (81–88 tok/s without MTP) keeps the 51B n-gram table in about 100 GB of host RAM. The card-only price excludes that host.
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
