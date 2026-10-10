"""Single-user decode roofline for every device-model pairing.

Inputs are spec-sheet bandwidth and compute peaks, model-config sizes, and the measured
(or estimated) speeds shown in the hardware map. Everything else is derived here, written to
data/roofline.json, and rendered into index.html between the roofline markers and into each
hardware row's bottleneck cell. Run with --check to verify index.html matches without writing.

Model of one decode step (k drafted tokens verified, a tokens emitted on average):
  bytes per emitted token = (weights read per step + cache read) / a
      weights read per step = D (dense) or D x (1 + 0.25k) (sparse MoE: drafted tokens partly hit
      different experts; bounded at <= 4.6 D for k = 7 by the two-Spark GLM-5.3-Flash run)
  ops per emitted token   = (2N(k+1) + 3 x weights read [dequantization] + attention x (k+1)) / a
      attention per token = 4 x full-attention layers x heads x head dim x attended context
  arithmetic intensity = ops / bytes; ridge point = vector compute peak / bandwidth
  roof = min(bandwidth / bytes, vector peak / ops) tokens/s; attainment = speed / roof
Vector (non-tensor) FP32 peaks are used for decode because batch-1 matrix-vector work does not
run efficiently on tensor/matrix units; FP32 is also the lower, more conservative ceiling.
"""
from math import log10
from pathlib import Path
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]

# key: label, bandwidth GB/s, vector FP32 TFLOPS, matrix FP16 dense TFLOPS (prefill), compute source
DEVICES = {
    'm5': ('MacBook Pro · M5', 153, 3.98, 16, 'Derived: A19 benchmark 257 FP32 FLOP/core/clock × 10 cores × ~1.55 GHz'),
    'm5pro': ('MacBook Pro · M5 Pro', 307, 9.0, 36, 'Derived: 20 cores × ~1.75 GHz'),
    'm5max': ('MacBook Pro · M5 Max', 614, 18.0, 70, 'Derived: 40 cores × ~1.75 GHz (A19 benchmark author’s extrapolation)'),
    'm5ultra': ('Mac Studio · M5 Ultra', 1200, 36.0, 140, 'Derived: 80 cores × ~1.75 GHz'),
    'rtx4090': ('GeForce RTX 4090', 1008, 82.6, 165, 'NVIDIA Ada specifications'),
    'r9700': ('Radeon AI PRO R9700', 640, 47.8, 191, 'AMD specifications (FP32 vector)'),
    'rtx5090': ('GeForce RTX 5090', 1792, 104.8, 209, 'NVIDIA Blackwell whitepaper'),
    'halo': ('Ryzen AI Max+ 395 / Halo', 256, 14.8, 59, '2,560 shaders × 2 × 2.9 GHz, single-issue (dual-issue counts 29.7)'),
    'spark': ('1 × DGX Spark', 273, 30.7, 125, 'Derived: 6,144 CUDA cores × 2 × ~2.5 GHz; NVIDIA publishes no FP32 figure'),
    'spark2': ('2 × DGX Spark', 546, 61.4, 250, 'Two nodes summed (tensor parallel)'),
    'pro6000': ('RTX PRO 6000 Blackwell', 1792, 125.0, 504, 'NVIDIA specifications (125 TFLOPS FP32)'),
    'pro2': ('2 × RTX PRO 6000 Blackwell', 1792, 125.0, 504, 'Sequential layer split: one card active at a time'),
    'm5ultra512': ('M5 Ultra · 512 GB', 1200, 36.0, 140, 'Same chip as M5 Ultra'),
    'station': ('DGX Station · GB300', 914, 80.0, 2500, 'GB300 ~80 TFLOPS FP32 (third-party); bandwidth blended 60% HBM 7,100 / 40% CPU 396 GB/s'),
    'station_hbm': ('DGX Station · GB300 (HBM only)', 7100, 80.0, 2500, 'GB300; model entirely in HBM'),
}

# key: active params (B), full-attention layers, heads, head dim for attention FLOPs,
#      attended-context cap (sparse attention) or None, cache bytes read per attended token,
#      extra cache bytes per context token (sparse-attention indexer keys)
MODELS = {
    'qwen36': (3, 10, 16, 256, None, 10 * 2 * 256 * 2 * 2, 0),
    'qwen27': (26, 16, 24, 256, None, 16 * 4 * 256 * 2 * 2, 0),
    'flashnext': (6, 12, 24, 256, 2048, 12 * 2 * 256 * 2 * 2, 12 * 128 * 2 // 4),
    'glmflash': (18, 11, 64, 544, 2048, 11 * 576 * 2, 11 * 128 * 2 // 4),
    'glm53': (40, 78, 64, 544, 2048, 78 * 576 * 2, 22 * 128 * 2),
}

# Hardware-map rows in display order. Speeds are (low, high); context and tokens-per-step can
# differ per endpoint. D = GB of active weights per forward pass at the deployed precision.
# key, device, model, D, speed, context, sparse MoE, k, a, estimate, source note
PAIRS = [
    ('m5', 'm5', 'qwen36', 1.7, (51.0, 59.5), (4096, 1024), True, 0, (1, 1), False, 'B12, 4-bit MLX'),
    ('m5pro', 'm5pro', 'qwen27', 19.5, (21, 44), (4096, 4096), False, 3, (3.0, 3.0), True, 'Estimate, MTPLX dynamic Q4'),
    ('rtx4090', 'rtx4090', 'qwen27', 17.9, (38.41, 46.16), (65536, 4096), False, 0, (1, 1), False, 'B10, Q4_K_S 16.68 GiB'),
    ('r9700', 'r9700', 'qwen27', 17.5, (51.8, 51.8), (4096, 4096), False, 2, (2.5, 2.5), False, 'B8, Q4-class, MTP=2'),
    ('rtx5090', 'rtx5090', 'qwen27', 17.5, (59, 59), (6091, 6091), False, 0, (1, 1), False, 'B3, Q4_K_M'),
    ('halo', 'halo', 'flashnext', 3.2, (31.7, 47.1), (8192, 8192), True, 4, (2.5, 3.5), False, 'B7, UD-IQ4_XS, MTP + n-gram'),
    ('spark', 'spark', 'flashnext', 3.4, (36.83, 45.75), (2048, 2048), True, 3, (2.3, 2.3), False, 'B5, NVFP4, MTP3'),
    ('m5max', 'm5max', 'flashnext', 3.5, (63.5, 63.5), (65502, 65502), True, 3, (2.3, 2.3), False, 'B1, MTPLX dynamic Q4'),
    ('pro6000', 'pro6000', 'flashnext', 3.4, (81, 88), (1024, 1024), True, 0, (1, 1), False, 'B9, NVFP4, no MTP'),
    ('spark2', 'spark2', 'glmflash', 9.9, (46.58, 62.61), (2048, 2048), True, 7, (3.5, 4.87), False, 'B16, EXL3 4-bit, DFlash2'),
    ('m5ultra', 'm5ultra', 'glmflash', 10.2, (44.0, 62.4), (4096, 200), True, 5, (2.4, 4.3), False, 'B17, oQ4e, DFlash2'),
    ('pro2', 'pro2', 'glmflash', 10.8, (41, 70), (4096, 4096), True, 0, (1, 1), True, 'Estimate, IQ4_XS'),
    ('m5ultra512', 'm5ultra512', 'glm53', 24.0, (20, 40), (4096, 4096), True, 5, (3.3, 3.3), True, 'Estimate, IQ4_XS, DFlash2'),
    ('station', 'station', 'glm53', 24.0, (21, 40), (4096, 4096), True, 5, (3.3, 3.3), True, 'Estimate, IQ4_XS split HBM/CPU, DFlash2'),
]
# Same-pairing references without drafting, and the Station fallback in HBM (drawer only).
REFERENCES = [
    ('halo_nospec', 'halo', 'flashnext', 3.2, (16.8, 16.8), (8192, 8192), True, 0, (1, 1), False, 'B7 without speculation'),
    ('spark2_nospec', 'spark2', 'glmflash', 9.9, (18.0, 18.0), (2048, 2048), True, 0, (1, 1), False, 'B16 without speculation'),
    ('m5ultra_nospec', 'm5ultra', 'glmflash', 10.2, (32.9, 38.8), (4096, 200), True, 0, (1, 1), False, 'B17 without drafting'),
    ('station_glmflash', 'station_hbm', 'glmflash', 10.1, (137, 137), (2048, 2048), True, 0, (1, 1), False, 'B15, NVFP4 in HBM'),
]
# Prompt processing (compute-bound side), against the matrix FP16 peak. chunk = prompt tokens
# sharing one weight read. key, device, model, D, measured prefill tok/s, context, note
PREFILL = [
    ('m5_prefill', 'm5', 'qwen36', 1.7, 1144, 4096, 'B12 at 4K'),
    ('spark_prefill', 'spark', 'flashnext', 3.4, 1606.5, 8215, 'B5 at 8K'),
    ('m5max_prefill', 'm5max', 'flashnext', 3.5, 1094, 65502, 'B1 at 65K'),
    ('pro6000_prefill', 'pro6000', 'flashnext', 3.4, 12000, 100000, 'B9 at 100K'),
]
PREFILL_CHUNK = 2048
MODEL_NAMES = {'qwen36': 'Qwen3.6-35B-A3B', 'qwen27': 'Qwen3.8-27B', 'flashnext': 'Qwen3.8-Flash-Next',
               'glmflash': 'GLM-5.3-Flash', 'glm53': 'GLM-5.3'}
ATTAINMENT_BANDS = ((0.55, 'bw', 'Bandwidth-bound'), (0.40, 'mix', 'Mixed'), (0.0, 'lat', 'Latency-bound'))
COLORS = {'bw': '#2a78d6', 'mix': '#eb6834', 'lat': '#1baf7a'}  # validated categorical slots 1-3


def attended(model, ctx):
    cap = MODELS[model][4]
    return min(ctx, cap) if cap else ctx


def step(device, model, d, ctx, moe, k, a):
    n, layers, heads, hd, _, cache_per_attended, cache_per_ctx = MODELS[model]
    bw, vec = DEVICES[device][1], DEVICES[device][2]
    weights = d * (1 + 0.25 * k) if moe else d
    cache = (cache_per_attended * attended(model, ctx) + cache_per_ctx * ctx) / 1e9
    attention = 4 * layers * heads * hd * attended(model, ctx) / 1e9
    gb = (weights + cache) / a
    gops = (2 * n * (k + 1) + 3 * weights / d * n + attention * (k + 1)) / a
    return {
        'gb_per_token': gb, 'gops_per_token': gops, 'intensity': gops / gb, 'ridge': vec * 1e3 / bw,
        'bandwidth_roof': bw / gb, 'compute_roof': vec * 1e3 / gops,
    }


def analyse(row):
    key, device, model, d, speed, ctx, moe, k, a, estimate, note = row
    ends = [step(device, model, d, c, moe, k, aa) for c, aa in zip(ctx, a)]
    roofs = [min(e['bandwidth_roof'], e['compute_roof']) for e in ends]
    attain = [s / r for s, r in zip(speed, roofs)]
    mean = sum(attain) / 2
    kind, label = next((k_, l_) for t, k_, l_ in ATTAINMENT_BANDS if mean >= t)
    return {
        'device': DEVICES[device][0], 'model': model, 'estimate': estimate, 'note': note, 'speed': list(speed),
        'endpoints': ends, 'roof': roofs, 'attainment': attain, 'kind': kind, 'label': label,
        'binding_roof': 'bandwidth' if all(e['bandwidth_roof'] <= e['compute_roof'] for e in ends) else 'compute',
        'compute_share_of_vector_peak': [s / e['compute_roof'] for s, e in zip(speed, ends)],
    }


def prefill(row):
    key, device, model, d, tps, ctx, note = row
    n, layers, heads, hd = MODELS[model][:4]
    bw, matrix = DEVICES[device][1], DEVICES[device][3]
    attention = 4 * layers * heads * hd * attended(model, ctx) / 2 / 1e9  # average over the prompt
    gops = 2 * n + 3 * n / PREFILL_CHUNK + attention
    gb = d / PREFILL_CHUNK
    return {'device': DEVICES[device][0], 'note': note, 'speed': tps, 'intensity': gops / gb,
            'ridge': matrix * 1e3 / bw, 'share_of_matrix_peak': tps * gops / (matrix * 1e3)}


def rng(values, fmt):
    lo, hi = fmt(min(values)), fmt(max(values))
    if lo == hi:
        return lo
    return f'{lo[:-1]}–{hi}' if lo.endswith('%') else f'{lo}–{hi}'


def pct(x):
    return f'{round(x * 100)}%'


def tps(x):
    return f'{x:.0f}'


def cell(r):
    est = ' (E)' if r['estimate'] else ''
    hollow = ' hollow' if r['estimate'] else ''
    return (f'<a href="#roofline" onclick="document.getElementById(\'research-details\').open=true">'
            f'<strong><i class="rf-dot{hollow}" style="--c:{COLORS[r["kind"]]}"></i>{r["label"].split("-")[0]}{est}</strong>'
            f'<span class="sub">{rng(r["attainment"], pct)} of {rng(r["roof"], tps)} tok/s roof</span></a>')


def chart(rows, refs, pre):
    w, h, ml, mr, mt, mb = 760, 430, 64, 16, 18, 52
    x0, x1, y0, y1 = -3, 2, -3, 0

    def px(v):
        return ml + (log10(v) - x0) / (x1 - x0) * (w - ml - mr)

    def py(v):
        return mt + (y1 - log10(v)) / (y1 - y0) * (h - mt - mb)

    out = [f'<svg class="roofline-chart" viewBox="0 0 {w} {h}" role="img" aria-labelledby="rf-title rf-desc">',
           '<title id="rf-title">Normalized roofline: every decode pairing sits far left of its ridge point</title>',
           '<desc id="rf-desc">Log-log chart. X is arithmetic intensity divided by the device ridge point; Y is attained '
           'compute divided by peak compute. The diagonal is the bandwidth roof and the flat line is the compute roof. '
           'All decode points lie on the bandwidth side below the diagonal; prompt-processing points lie on the compute side.</desc>']
    for e in range(x0, x1 + 1):
        x = px(10 ** e)
        out.append(f'<line class="rf-grid" x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{h - mb}"/>'
                   f'<text class="rf-tick" x="{x:.1f}" y="{h - mb + 16}" text-anchor="middle">{10 ** e:g}×</text>')
    for e in range(y0, y1 + 1):
        y = py(10 ** e)
        out.append(f'<line class="rf-grid" x1="{ml}" y1="{y:.1f}" x2="{w - mr}" y2="{y:.1f}"/>'
                   f'<text class="rf-tick" x="{ml - 8}" y="{y + 4:.1f}" text-anchor="end">{10 ** e * 100:g}%</text>')
    out.append(f'<polyline class="rf-roof" points="{px(1e-3):.1f},{py(1e-3):.1f} {px(1):.1f},{py(1):.1f} {px(100):.1f},{py(1):.1f}"/>'
               f'<text class="rf-note" x="{px(0.012):.1f}" y="{py(0.03):.1f}" transform="rotate(-31 {px(0.012):.1f} {py(0.03):.1f})">bandwidth roof</text>'
               f'<text class="rf-note" x="{px(3):.1f}" y="{py(1) - 6:.1f}">compute roof</text>'
               f'<text class="rf-note" x="{px(1) + 4:.1f}" y="{h - mb - 6}">ridge</text>'
               f'<line class="rf-ridge" x1="{px(1):.1f}" y1="{py(1):.1f}" x2="{px(1):.1f}" y2="{h - mb}"/>')
    out.append(f'<text class="rf-axis" x="{(ml + w - mr) / 2}" y="{h - 12}" text-anchor="middle">Arithmetic intensity ÷ device ridge point (log)</text>'
               f'<text class="rf-axis" x="14" y="{(mt + h - mb) / 2}" text-anchor="middle" transform="rotate(-90 14 {(mt + h - mb) / 2})">Attained ÷ peak compute (log)</text>')
    for p in pre.values():
        x, y = px(p['intensity'] / p['ridge']), py(p['share_of_matrix_peak'])
        out.append(f'<g class="rf-pt"><rect x="{x - 5:.1f}" y="{y - 5:.1f}" width="10" height="10" transform="rotate(45 {x:.1f} {y:.1f})" class="rf-prefill"/>'
                   f'<title>{p["device"]} prompt processing: {p["speed"]:,.0f} tok/s = {pct(p["share_of_matrix_peak"])} of matrix FP16 peak ({p["note"]})</title></g>')
    labels = {'rtx4090': ('RTX 4090', 'end'), 'pro6000': ('PRO 6000', 'start'), 'm5pro': ('M5 Pro (E)', 'start')}
    for key, r in rows.items():
        e = r['endpoints']
        xi = sum(x['intensity'] / x['ridge'] for x in e) / 2
        yi = sum(s * x['gops_per_token'] / (DEVICES_BY_LABEL[r['device']][2] * 1e3) for s, x in zip(r['speed'], e)) / 2
        x, y = px(xi), py(yi)
        fill = 'none' if r['estimate'] else COLORS[r['kind']]
        out.append(f'<g class="rf-pt"><circle cx="{x:.1f}" cy="{y:.1f}" r="8" class="rf-ring"/><circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="{fill}" stroke="{COLORS[r["kind"]]}" class="rf-dot-mark"/>'
                   f'<title>{r["device"]}: {rng(r["speed"], lambda v: f"{v:g}")} tok/s = {rng(r["attainment"], pct)} of the '
                   f'{rng(r["roof"], tps)} tok/s bandwidth roof; compute {rng(r["compute_share_of_vector_peak"], pct)} of vector peak '
                   f'({r["note"]})</title></g>')
        if key in labels:
            text, anchor = labels[key]
            dx = 11 if anchor == 'start' else -14
            out.append(f'<text class="rf-label" x="{x + dx:.1f}" y="{y + 4:.1f}" text-anchor="{anchor}">{text}</text>')
    out.append(f'<text class="rf-label" x="{px(30):.1f}" y="{py(0.08):.1f}" text-anchor="middle">prompt processing</text></svg>')
    legend = ''.join(f'<span><i class="rf-dot" style="--c:{COLORS[k]}"></i>{l}</span>' for _, k, l in ATTAINMENT_BANDS)
    legend += '<span><i class="rf-dot hollow" style="--c:#596765"></i>Estimate</span><span><i class="rf-diamond"></i>Prompt processing</span>'
    return f'<div class="rf-legend">{legend}</div>' + ''.join(out)


def table(rows, refs, pre):
    head = ('<table class="method-table rf-table"><thead><tr><th>Pairing</th><th>GB / token</th><th>Ops / token (G)</th>'
            '<th>Intensity · ridge (ops/byte)</th><th>Bandwidth roof</th><th>Compute roof</th><th>Speed</th><th>Attainment</th></tr></thead><tbody>')
    body = []
    for r in list(rows.values()) + list(refs.values()):
        e = r['endpoints']
        body.append(f'<tr><td>{r["device"]} · {MODEL_NAMES[r["model"]]}{" (E)" if r["estimate"] else ""}<br><span class="sub">{r["note"]}</span></td>'
                    f'<td>{rng([x["gb_per_token"] for x in e], lambda v: f"{v:.1f}")}</td>'
                    f'<td>{rng([x["gops_per_token"] for x in e], lambda v: f"{v:.0f}")}</td>'
                    f'<td>{rng([x["intensity"] for x in e], lambda v: f"{v:.1f}")} · {e[0]["ridge"]:.0f}</td>'
                    f'<td>{rng([x["bandwidth_roof"] for x in e], tps)}</td>'
                    f'<td>{rng([x["compute_roof"] for x in e], tps)}</td>'
                    f'<td>{rng(r["speed"], lambda v: f"{v:g}")}</td>'
                    f'<td>{rng(r["attainment"], pct)} · {r["label"]}</td></tr>')
    for p in pre.values():
        body.append(f'<tr><td>{p["device"]} · prompt processing<br><span class="sub">{p["note"]}</span></td><td colspan="2">'
                    f'{PREFILL_CHUNK}-token chunks share each weight read</td><td>{p["intensity"]:,.0f} · {p["ridge"]:.0f}</td>'
                    f'<td colspan="3">{p["speed"]:,.0f} tok/s; compute-bound</td><td>{pct(p["share_of_matrix_peak"])} of matrix peak</td></tr>')
    inputs = ''.join(f'<tr><td>{v[0]}</td><td>{v[1]:,}</td><td>{v[2]:g}</td><td>{v[3]:g}</td><td>{v[4]}</td></tr>'
                     for k, v in DEVICES.items())
    return (head + ''.join(body) + '</tbody></table>'
            '<table class="method-table rf-table"><thead><tr><th>Device input</th><th>Bandwidth GB/s</th><th>Vector FP32 TFLOPS</th>'
            '<th>Matrix FP16 TFLOPS</th><th>Compute source</th></tr></thead><tbody>' + inputs + '</tbody></table>')


DEVICES_BY_LABEL = {v[0]: v for v in DEVICES.values()}


def build():
    rows = {r[0]: analyse(r) for r in PAIRS}
    refs = {r[0]: analyse(r) for r in REFERENCES}
    pre = {r[0]: prefill(r) for r in PREFILL}
    return rows, refs, pre


def render(html, rows, refs, pre):
    block = chart(rows, refs, pre) + table(rows, refs, pre)
    html = re.sub(r'(<!-- roofline:start -->).*?(<!-- roofline:end -->)', lambda m: m[1] + block + m[2], html, flags=re.S)
    for key, r in rows.items():
        pattern = rf'(<td data-roofline="{key}">).*?(</td>)'
        html, count = re.subn(pattern, lambda m: m[1] + cell(r) + m[2], html, flags=re.S)
        assert count == 1, f'missing bottleneck cell for {key}'
    return html


def main():
    rows, refs, pre = build()
    (ROOT / 'data' / 'roofline.json').write_text(
        json.dumps({'pairs': rows, 'references': refs, 'prefill': pre, 'devices': DEVICES}, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8', newline='\n')
    path = ROOT / 'index.html'
    html = path.read_text(encoding='utf-8')
    new = render(html, rows, refs, pre)
    if '--check' in sys.argv:
        if new != html:
            raise SystemExit('Validation failed: index.html roofline content is stale; run python scripts/roofline.py')
        print('Roofline content in index.html matches the calculation.')
    else:
        path.write_text(new, encoding='utf-8', newline='\n')
        print('Wrote data/roofline.json and updated index.html roofline content.')


if __name__ == '__main__':
    main()
