# Local Intelligence Brief

A single-page comparison of local AI hardware, model capability, memory, cost and single-user decode speed. Research snapshot: **10 October 2026**.

**[Open the live report →](https://iv2nl0b9v.github.io/local-intelligence-brief/)**

Hosted with GitHub Pages from the root of `main`. Pushing updates to that branch automatically republishes the site; `.nojekyll` serves the files as plain static content.

- [Read the report](index.html) — open locally in a browser; expand **Sources & calculations** for the evidence ledger and methodology.
- [Download the single-page A3 PDF](local-ai-field-guide.pdf).
- [Research notes as text](data/research-notes.txt).
- [Hardware data](data/hardware.json), [model data](data/models.json), and [source links](data/sources.json).
- [Reproduced forecast and memory arithmetic](data/calculations.json).

## What the ranking means

Choose the highest-scoring fitting model **within the report's shortlist** whose measured or estimated decode **upper bound reaches 35 tokens/second**. Sort devices by model intelligence, then speed-range midpoint, ascending. This is not a guaranteed 35 tok/s minimum or an exhaustive model search.

Intelligence uses Artificial Analysis Intelligence Index v4.3.2. Scores describe the tested model and reasoning setting, not a fresh evaluation of the local quantization. Four Claude Opus releases (4.6 → 4.8 → 5 → 5.5: 32* → 42 → 51 → 58, February to September 2026; *AA estimate) mark the frontier trajectory. They are proprietary references, not local deployment options. The best local-fitting model, GLM-5.3 (45), sits between Opus 4.8 and Opus 5.

Decode is one active request, warm model, with workload/context differences recorded per source. Model-memory budgets use **131,072 total cached tokens**, concurrency one, decimal GB, and separately list SSD lookup tables and runtime allowances. Speed estimates are not 128K speed forecasts. Every model row includes a linked release date for that version; GLM-5.3 is explicitly labelled with its API release-note date. AI compute reports dense peaks only, without sparsity multipliers, and selects the higher INT8 TOPS estimate for Macs, with TFLOPS for other GPUs and precision labels throughout. Paired-device bandwidth cells separately list per-direction interconnect ceilings: 25 GB/s for Spark and up to 63 GB/s for a full-width PCIe 5.0 ×16 PRO configuration; actual transfers depend on topology and overhead. Apple GPU FP16 and INT8 matrix rates describe the same accelerator and are not added; the separate Neural Engine has no verified dense rating for the listed configurations. Halo shows a theoretical mixed-operation sum of separate GPU and NPU ratings: 60 FP16 TFLOPS + 50 INT8 TOPS = 110 trillion operations/s. Concurrent engines share power and memory; this is not measured combined LLM throughput. CPU compute and unverified Apple Neural Engine ratings are excluded.

## Re-evaluate the report

1. Start with `index.html`: it contains the complete source ledger, benchmark conditions, hardware/pricing references, formulas, and selection notes. It is the authoritative report source.
2. Check the linked original benchmark logs, runtime versions, model artifacts and quantization. Do not transfer a result between models or GPU counts as if it were measured.
3. Consult `data/configs/` for captured model architecture configurations and `data/config-manifest.json` for URLs, UTC capture times and SHA-256 hashes. These snapshots support cache calculations; they do not pin the full weight artifacts or runtime code.
4. Headline identities and displayed rates are pinned in `data/configurations.json`. Change assumptions in the report and, where relevant, `scripts/calculations.py`. Keep both consistent. Regenerate the derived data and PDF.
5. Validate the 35 tok/s upper-bound rule, ordering, table structure, links, and that every displayed estimate range equals its computed endpoints rounded to the nearest integer, with `python scripts/export_report.py`.

### Important unresolved evidence

- Measured rows are third-party or vendor results, labelled **M**, not this report's replication. The R9700's **51.8 tok/s** is AMD's preliminary vendor number (Vulkan, MTP=2, quantization unstated). Halo's **32–47** comes from one community llama.cpp thread whose same-build runs varied 25–36 tok/s across days.
- Four rows remain estimates:
  - **M5 Pro / Qwen3.8-27B: 21–44.** The upper end extrapolates favorable repetitive editing. An independent 2.33× MTP multiplier verified on M5 Pro supports roughly 40, but there is no direct measurement.
  - **2 × RTX PRO 6000 / GLM-5.3-Flash: 165 tok/s projected.** Preferred expert-NVFP4/FP8-WO checkpoint with SGLang TP2 and adaptive EAGLE. B19 measured 164.8 tok/s on Server Edition cards; the Workstation number assumes this recipe transfers without a penalty and is labelled E. Matching Workstation throughput and runtime fit remain unverified. Target + draft weights use 84.64 GB per rank before caches and recurrent state.
  - **M5 Ultra 512 GB / GLM-5.3: 18–40** and **DGX Station / GLM-5.3: 19–40.** Both qualify only with a DFlash2 drafter. The drafter exists for GLM-5.3, but no full-model run on these systems was found. The 1.22–1.61× gain includes all three B17 prompt lengths, including the previously omitted 1K test. This transfer is a sensitivity scenario, not a conservative performance bound. Drafter memory and full-model backend support remain unverified. Without drafting both fall to 15–25 tok/s, and GLM-5.3-Flash is the fallback.
- Dual Spark runs GLM-5.3-Flash EXL3 4-bit at 47–63 tok/s (prose to coding); the separate NVFP4 configuration reports 28–52. The EXL3 result uses the DFlash2 drafter, which is licensed for non-commercial use only. Without drafting it measures ~14–18 tok/s; the fallback is Flash-Next split across both nodes at 42–52.

- The single RTX PRO 6000 Flash-Next result (120–173 tok/s with MTP; 81–88 baseline) keeps the 51B n-gram table in about 100 GB of host RAM. The card-only price excludes that host.
- Compressed MLA versus expanded per-head KV changes GLM-Flash's 128K cache dramatically. The selected memory budget requires the compressed-cache backend scenario.
- Apple compute figures are third-party estimates for the matrix units in each M5 GPU core (INT8, selected over FP16), scaled from A19 microbenchmarks; Apple publishes none. They are peak values that mainly affect prefill. These figures cannot establish a bottleneck without profiling the exact runtime and workload.
- Prices are approximate and hardware availability is snapshot-specific.

These limitations and all other assumptions are detailed in the research drawer. No in-house hardware benchmark was performed for this report. No raw third-party benchmark logs are represented as our own measurements.

## Rebuild (Python 3; no Python dependencies)

```sh
python scripts/calculations.py
python scripts/export_report.py
python scripts/test_report.py
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

## Audit evidence and reproducibility

`data/spark-decode-audit.json` preserves the numeric fields needed to reproduce the two headline workload medians, with upstream revision, paths and full-original-file hashes. It is a third-party log extract, not an in-house benchmark. The regression tests reject changed HTML headlines and check decode versus delivered rate formulas.

The report uses B19’s two-PRO drafting recipe as a clearly labelled transfer estimate for Workstation cards, and updates the DGX Station price range to the source's $99,000–188,500 OEM span. Model scores and 128K architecture-cache arithmetic were retained; they do not validate local quantization quality or an exact runtime's total memory fit.
