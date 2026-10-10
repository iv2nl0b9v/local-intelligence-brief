# Local Intelligence Brief

A single-page comparison of local AI hardware, model capability, memory, cost and single-user decode speed. Research snapshot: **10 October 2026**.

**[Open the live report →](https://iv2nl0b9v.github.io/local-intelligence-brief/)**

Hosted with GitHub Pages from the root of `main`. Pushing updates to that branch automatically republishes the site; `.nojekyll` serves the files as plain static content.

- [Read the report](index.html) — open locally in a browser; expand **Sources & calculations** for the evidence ledger and methodology.
- [Download the single-page A3 PDF](local-ai-field-guide.pdf).
- [Research notes as text](data/research-notes.txt).
- [Hardware data](data/hardware.json), [model data](data/models.json), and [source links](data/sources.json).
- [Reproduced arithmetic](data/calculations.json) and [roofline](data/roofline.json).

## What the ranking means

Choose the highest-scoring fitting model **within the report's shortlist** whose measured or estimated decode **upper bound reaches 35 tokens/second**. Sort devices by model intelligence, then speed-range midpoint, ascending. This is not a guaranteed 35 tok/s minimum or an exhaustive model search.

Intelligence uses Artificial Analysis Intelligence Index v4.3.2. Scores describe the tested model and reasoning setting, not a fresh evaluation of the local quantization. Four Claude Opus releases (4.6 → 4.8 → 5 → 5.5: 32* → 42 → 51 → 58, February to September 2026; *AA estimate) mark the frontier trajectory. They are proprietary references, not local deployment options. The best local-fitting model, GLM-5.3 (45), sits between Opus 4.8 and Opus 5.

Decode is one active request, warm model, with workload/context differences recorded per source. Model-memory budgets use **131,072 total cached tokens**, concurrency one, decimal GB, and separately list SSD lookup tables and runtime allowances. Speed estimates are not 128K speed forecasts.

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
  - **2 × RTX PRO 6000 / GLM-5.3-Flash IQ4_XS: 41–70, best ~55.** Assumes llama.cpp's sequential layer split. The original four-card NVFP4 artifact does not fit on two. A different expert-NVFP4/FP8-WO artifact now measures **164.8 tok/s** on two **Server Edition** cards with SGLang and adaptive EAGLE (B19); that result and its price are not interchangeable with this Workstation IQ4_XS scenario.
  - **M5 Ultra 512 GB / GLM-5.3: 18–40** and **DGX Station / GLM-5.3: 19–40.** Both qualify only with a DFlash2 drafter. The drafter exists for GLM-5.3, but no full-model run on these systems was found. The 1.22–1.61× gain includes all three B17 prompt lengths, including the previously omitted 1K test. This transfer is a sensitivity scenario, not a conservative performance bound. Drafter memory and full-model backend support remain unverified. Without drafting both fall to 15–25 tok/s, and GLM-5.3-Flash is the fallback.
- Dual Spark runs GLM-5.3-Flash EXL3 4-bit at 47–63 tok/s (prose to coding); the separate NVFP4 configuration reports 28–52. The EXL3 result uses the DFlash2 drafter, which is licensed for non-commercial use only. Without drafting it measures ~14–18 tok/s; the fallback is Flash-Next split across both nodes at 42–52.
- Quantitative rooflines are retained for all 14 pairings: BW demand, precision-specific target + drafter compute demand and the gap to the modeled roof, with detailed ceilings, ideal resource times, a normalized chart and traffic sensitivity ranges. These are explicit scenarios, not elapsed-time attribution. Blue cells mark the lower bandwidth reference roof; shaded device rows remain forecasts, not demonstrated eligibility.
- Header accounting for the single PRO 6000 gives **9.951 GB/pass**, because dense/attention/shared weights remain BF16. That implies **45–49% conditional weights-only bandwidth attainment**, not the old 15–17%. This is not measured DRAM traffic and the historical benchmark did not pin the audited weight revision.
- A nearby dual-Spark EXL3 profile attributes ~82–83% of verifier time to bandwidth-bound kernels and ~4% to idle gaps. This source-author classification does not establish every device's bottleneck. The raw prose/coding logs reproduce the reported decode rates; including TTFT lowers the medians to 44.39/57.96 tok/s. See the report for prompt sizes, modified output head, and quality limitations.

- The single RTX PRO 6000 Flash-Next result (81–88 tok/s without MTP) keeps the 51B n-gram table in about 100 GB of host RAM. The card-only price excludes that host.
- Compressed MLA versus expanded per-head KV changes GLM-Flash's 128K cache dramatically. The selected memory budget requires the compressed-cache backend scenario.
- Apple compute figures are third-party estimates for the matrix units in each M5 GPU core (FP16/INT8), scaled from A19 microbenchmarks; Apple publishes none. They are peak values that mainly affect prefill. These figures cannot establish a bottleneck without profiling the exact runtime and workload.
- Prices are approximate and hardware availability is snapshot-specific.

These limitations and all other assumptions are detailed in the research drawer. No in-house hardware benchmark was performed for this report. No raw third-party benchmark logs are represented as our own measurements.

## Rebuild (Python 3; no Python dependencies)

```sh
python scripts/calculations.py
python scripts/roofline.py
python scripts/export_report.py
python scripts/roofline.py --check
python scripts/check_weight_traffic.py
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

`data/weight-headers/` contains five captured safetensors JSON headers and a revision/URL/hash manifest. `scripts/check_weight_traffic.py` reproduces their category totals and the 9.951 GB parameter-read estimate offline. It extrapolates the selected experts from layer 0 across 48 layers; physical reads, cache effects and historical artifact identity remain unmeasured.

`data/spark-decode-audit.json` preserves the numeric fields needed to reproduce the two headline workload medians, with upstream revision, paths and full-original-file hashes. It is a third-party log extract, not an in-house benchmark. The regression tests reject changed HTML headlines, check decode versus delivered rate formulas, test expert-reuse sensitivity and profile accounting, and ensure `roofline.py --check` does not write files.

The report now includes the B19 two-PRO Server Edition result, distinguishes it from the Workstation estimate, and updates the DGX Station price range to the source's $99,000–188,500 OEM span. Model scores and 128K architecture-cache arithmetic were retained; they do not validate local quantization quality or an exact runtime's total memory fit.

## Interpreting the restored bottleneck analysis

The useful comparison is `bandwidth / modeled bytes` versus `compute reference / modeled FLOPs`. The peak-roof gap cannot distinguish memory inefficiency, kernel latency, missing traffic and host overhead. A workload requiring 45% of peak bandwidth can still spend almost its entire duration in memory-limited kernels if they sustain 45% of peak.

`data/roofline-inputs.json` preserves the original planning inputs with the audited PRO weight correction, GLM cache/attention-dimension corrections, and EXL3 draft/emission/context corrections. Unverified inputs remain explicitly marked assumptions. The central expert-union scenario recovers the old multiplier, while the drawer tests reuse from full to none, routed fraction 25–75%, and payload ±25%. This is a sensitivity exercise, not a statistical bound. Implausible demand above 100% remains visible.

Compute is rebuilt in `scripts/compute_roofline.py`. For each speculative cycle, count target verification (k+1 positions for a linear chain), plus sequential MTP forwards or one parallel DFlash block. Divide each component's operations by the dense hardware capacity for its execution precision, sum those ideal times, and divide by emitted tokens. Demand is observed tokens/s times ideal seconds/token. DFlash counts block−1 query rows, full vocabulary head, context KV and target-feature projection using captured drafter configs. Integer work uses GOP/TOPS, not FLOPs. Flash-Next matrix counts come from tensor headers; other target active counts remain approximations. Dequantization, selector/convolution, recurrence, padding and non-matmul work are excluded, explicitly, rather than assigned an arbitrary FLOP surcharge.

`data/compute-inputs.json` records every capacity and conditional dispatch path; `data/compute-evidence/manifest.json` links and hashes the source snapshots. EXL3/FP8-stored Spark weights execute FP16/BF16 MMA with FP32 accumulation, so their compute ceiling is 16-bit tensor capacity. Native NVFP4 expert scenarios use FP4 only for experts. CUDA GGUF INT8-dot and MLX FP32-qmv alternatives are distinct from tensor paths. Several historical dispatches and capacity figures are unverified or extrapolated, so their numerical envelopes are conditional, not measurements of actual utilization. No generic FP32 ceiling is used for all devices, and no structured-sparsity bonus is applied to MoE routing.

Spark work and emitted-token counts are matched per raw request before taking workload medians. VTRIM live expert rows and graph padding are not fully recoverable from those counters; the model reports nominal verification work. Full-vocabulary drafter-head arithmetic is a reference rather than an exact account of optimized candidate-head instructions. B17 enumerates block 4/8 with emitted lengths 2.4–4.3. Other unreported counts remain planning inputs. Bandwidth retains its separate traffic/reuse assumptions and is not a trace-matched endpoint for each compute alternative. The ledger exports all stages and scenarios. Shape-matched sustained ceilings and per-layer row counters are the next calibration step; low demand against peak does not identify removable overhead.

`data/spark-profile.json` records the nearby two-rank profile separately. At rank 0 and six verification rows, 61.92 of 75.09 ms belongs to bandwidth-class kernels, and 3.23 ms to idle gaps. Removing idle alone would imply 4.5% speedup at fixed acceptance; the optimistic ceiling from removing everything outside those kernels is 21.3%. These are hypothetical bounds for that trace, not promises for the headline run.

The improvement plan in the report prioritizes artifact-specific byte accounting, shape-matched sustained ceilings, critical-path profiling, controlled context/speculation interventions, and separate decode/delivered metrics. No new hardware profiling was performed.
