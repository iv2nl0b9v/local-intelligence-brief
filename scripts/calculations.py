"""Reproduce planning arithmetic; this script is not a hardware benchmark.

Any scenario with a 'displayed' range is shown in index.html; export_report.py checks that
it equals the computed endpoints rounded to the nearest integer. Superseded forecasts keep
their endpoints next to the measurement that replaced them, as a calibration record.
"""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
N = 131072
GB = 1e9


def rate(traffic_gb, bandwidth_gbps, utilization, overhead_ms):
    return 1 / (traffic_gb / (bandwidth_gbps * utilization) + overhead_ms / 1000)


def split_memory(traffic_gb, cpu_fraction, utilization, overhead_s, hbm=7100, cpu=396):
    """DGX Station: active-weight traffic split between HBM and CPU LPDDR5X; bandwidths not added."""
    seconds = (traffic_gb * (1 - cpu_fraction) / (hbm * utilization)
               + traffic_gb * cpu_fraction / (cpu * utilization) + overhead_s)
    return 1 / seconds


results = {
    'units': {'memory': 'decimal GB', 'speed': 'tokens/second', 'context_tokens': N, 'concurrency': 1},
    'warning': 'Assumed traffic, utilization, overhead and drafting factors; not measurements or confidence intervals.',
    'pro_single_flashnext_superseded': {
        'traffic_gb': [5, 7], 'bandwidth_gbps': 1792, 'utilization': [.5, .7], 'overhead_ms': [4, 7],
        'central': rate(6, 1792, .6, 5), 'endpoints': [rate(7, 1792, .5, 7), rate(5, 1792, .7, 4)],
        'measured_no_mtp': [81, 88], 'measured_with_mtp': [120, 173], 'measurement': 'B9',
        'ideal_compute_roof': 2015e12 / 12e9,
    },
    'pro_dual_glmflash_layer_split': {
        'traffic_gb': [10.8, 13], 'bandwidth_gbps': 1792, 'utilization': [.5, .65], 'overhead_ms': [5, 10],
        'central': rate(12, 1792, .6, 7), 'endpoints': [rate(13, 1792, .5, 10), rate(10.8, 1792, .65, 5)],
        'displayed': [41, 70], 'anchor_4x_pro_tp4_nvfp4_measured': 63.7,
    },
    'pro_dual_glmflash_tensor_parallel_upside': {
        'traffic_gb': [10.8, 13], 'bandwidth_gbps': 3584, 'utilization': [.5, .65], 'overhead_ms': [5, 10],
        'central': rate(12, 3584, .6, 7), 'endpoints': [rate(13, 3584, .5, 10), rate(10.8, 3584, .65, 5)],
        'ideal_fp4_compute_roof_not_iq4_kernel_performance': 4030e12 / 36e9,
    },
    'pro_anchor': {'measured_tp2_flashnext_fp8': 81.45, 'measured_tp4': 64.61,
                   'tp2_latency_ms': 1000 / 81.45,
                   'residual_ms_for_6_to_10GB_at_60pct': [1000 / 81.45 - d / (3584 * .6) * 1000 for d in [6, 10]]},
    'm5_qwen35_moe_superseded': {
        'endpoints': [rate(2.5, 153, .6, 8), rate(2, 153, .8, 4)],
        'measured': [51.0, 59.5], 'measurement': 'B12',
        'max_traffic_gb_at_measured_peak_no_overhead': 153 / 59.5,
    },
    'm5pro_qwen38': {'fresh_coding': [52.6 * .5 * .8, 58.7 * .5], 'repetitive_editing': [87.6 * .5 * .8, 87.6 * .5],
                     'endpoints': [52.6 * .5 * .8, 87.6 * .5], 'displayed': [21, 44],
                     'cross_check_2_33x_times_17_to_18_baseline': [17 * 2.33, 18 * 2.33]},
    'r9700_qwen38_superseded': {'baseline': [24, 32], 'assumed_net_drafting': [1.25, 1.6],
                                'endpoints': [24 * 1.25, 32 * 1.6], 'measured_vendor': 51.8, 'measurement': 'B8'},
    'dual_spark_glmflash_rejected': {
        'endpoints': [546 * .3 * 1.3 / 10.8, 546 * .5 * 1.5 / 10.8],
        'measured_single_request': [14.6, 30.84], 'measurement': 'B11',
    },
    'halo_flashnext_rejected_projection': {
        'endpoints': [256 * .4 * 1.2 / 8, 256 * .6 * 1.5 / 8], 'measured': [31.7, 47.1], 'measurement': 'B7',
    },
    'rtx4090_qwen27_superseded': {'endpoints': [1008 * .6 / 16, 1008 * .8 / 16],
                                  'measured': [38.41, 46.16], 'measurement': 'B10',
                                  'rtx5090_measured_utilization_at_16GB': 59 * 16 / 1792},
    'cache_gb_128k_scenarios': {
        'qwen27_with_one_mtp_and_recurrent': (N * 17 * 2 * 4 * 256 * 2 + 48 * 48 * 128 * 128 * 4) / GB,
        'qwen35_with_one_mtp_and_recurrent': (N * 11 * 2 * 2 * 256 * 2 + 30 * 32 * 128 * 128 * 4) / GB,
        'flashnext_with_mtp_index_and_recurrent': (N * 13 * 2 * 2 * 256 * 2 + N / 4 * 12 * 128 * 2 + 36 * 48 * 128 * 128 * 4) / GB,
        'glm53_mla_with_mtp_and_index': (N * 79 * 576 * 2 + N * 22 * 128 * 2) / GB,
        'glmflash_compressed_with_mtp_and_recurrent': (N * 12 * 512 * 2 + N / 4 * 12 * 128 * 2 + 34 * 64 * 128 * 128 * 4) / GB,
        'glmflash_expanded_11_layers_no_mtp': N * 11 * 64 * 512 * 2 / GB,
        'deepseek4_compressed_bf16': ((21 * N / 4 + 20 * N / 128) * 512 * 2 + 21 * N / 4 * 128 * 2 + 46 * 128 * 512 * 2) / GB,
        'deepseek41_packed_global_only': N * 890 / GB,
        'deepseek41_bf16_with_local_windows': (N * (2.5 * 512 + 6.5 * 128) * 2 + 43 * 128 * 512 * 2) / GB,
    },
    'deepseek41_storage_gb': {'resident_weights': 294.15 * 2**30 / GB, 'ssd_engram': 188.83 * 2**30 / GB},
}
# GLM-5.3 on split or Apple memory: no-drafting roofline, then the smallest measured DFlash2 gain on
# a GLM MoE (oMLX, GLM-5.3-Flash on M5 Ultra: 32.9 -> 44.0 at 4K, 38.8 -> 62.4 at 200 tokens).
# The smallest gain is used because these systems are the most bandwidth-limited.
DFLASH2_GAIN = [44.0 / 32.9, 62.4 / 38.8]
baseline = [split_memory(24, x, u, h) for x, u, h in [(.42, .5, .008), (.38, .7, .004)]]
results['station_glm53'] = {
    'baseline_no_drafting': baseline, 'dflash2_gain': DFLASH2_GAIN,
    'endpoints': [baseline[0] * DFLASH2_GAIN[0], baseline[1] * DFLASH2_GAIN[1]], 'displayed': [21, 40],
}
ultra = [1200 * .3 / 24, 1200 * .5 / 24]
results['ultra512_glm53'] = {
    'baseline_no_drafting': ultra, 'dflash2_gain': DFLASH2_GAIN,
    'endpoints': [ultra[0] * DFLASH2_GAIN[0], ultra[1] * DFLASH2_GAIN[1]], 'displayed': [20, 40],
    'cross_check_m3_ultra_glm52_oq4_scaled_to_1200': 14.5 * 1200 / 819,
}
mimo = [split_memory(23, x, u, h) for x, u, h in [(.63, .5, .008), (.60, .7, .004)]]
results['station_mimo_v26_pro_rejected'] = {
    'weights_gb': 566, 'cpu_resident_gb': [566 - 228, 566 - 212], 'cpu_traffic_fraction': [.60, .63],
    'baseline_no_drafting': mimo, 'endpoints': [mimo[0] * DFLASH2_GAIN[0], mimo[1] * DFLASH2_GAIN[1]],
    'displayed': [16, 29],
}

# Decode bottleneck: share of peak bandwidth and FP16 compute implied by each row's speed.
# D = GB of active weights per forward pass, N = active params (B), k drafted tokens per step,
# a = tokens emitted per step. Dense: a verify step reads the weights once (D/a per token).
# Sparse MoE: drafted tokens share some experts; assume a step reads D x (1 + 0.25k). Two-Spark
# GLM-5.3-Flash at 99.4 tok/s with 8 tokens per 80 ms step bounds this at <= 4.6 D for k=7; the
# assumed 2.75 D puts that run, and its coding and prose runs, at a consistent ~60% of bandwidth.
# FLOPs per token: 2N(k+1)/a (rejected drafts are still computed). Cache reads excluded except
# where noted, so long-context rows understate bandwidth use.
BOTTLENECK_ROWS = [
    # pairing, GB/s, FP16 dense TFLOPS, N, D, (tok/s lo, hi), sparse MoE, k, a
    ('M5 / Qwen3.6-35B-A3B', 153, 16, 3, 1.7, (51, 60), True, 0, 1),
    ('M5 Pro / Qwen3.8-27B', 307, 36, 26, 19.5, (21, 44), False, 3, 3.0),
    ('RTX 4090 / Qwen3.8-27B', 1008, 330, 26, 17.9, (38, 46), False, 0, 1),
    ('R9700 / Qwen3.8-27B', 640, 191, 26, 17.5, (51.8, 51.8), False, 2, 2.5),
    ('RTX 5090 / Qwen3.8-27B', 1792, 419, 26, 17.5, (59, 59), False, 0, 1),
    ('Halo / Flash-Next', 256, 59, 6, 3.2, (32, 47), True, 4, 3.5),
    ('Halo / Flash-Next no drafting', 256, 59, 6, 3.2, (16.8, 16.8), True, 0, 1),
    ('1x Spark / Flash-Next', 273, 125, 6, 3.4, (37, 46), True, 3, 2.3),
    ('M5 Max / Flash-Next (65K, incl. ~1.75 GB cache)', 614, 70, 6, 5.25, (63.5, 63.5), True, 3, 2.3),
    ('RTX PRO 6000 / Flash-Next', 1792, 504, 6, 3.4, (81, 88), True, 0, 1),
    ('2x Spark / GLM-5.3-Flash prose', 546, 250, 18, 9.5, (46.6, 46.6), True, 7, 3.5),
    ('2x Spark / GLM-5.3-Flash coding', 546, 250, 18, 9.5, (62.6, 62.6), True, 7, 4.87),
    ('2x Spark / GLM-5.3-Flash no drafting', 546, 250, 18, 9.5, (18.0, 18.0), True, 0, 1),
    ('M5 Ultra / GLM-5.3-Flash DFlash2', 1200, 140, 18, 10.1, (44, 62), True, 5, 3.3),
    ('M5 Ultra / GLM-5.3-Flash no drafting', 1200, 140, 18, 10.1, (32.9, 38.8), True, 0, 1),
    ('2x RTX PRO 6000 / GLM-5.3-Flash', 1792, 1008, 18, 10.8, (41, 70), True, 0, 1),
    ('DGX Station / GLM-5.3-Flash in HBM (reference)', 7100, 2500, 18, 10.1, (137, 137), True, 0, 1),
]
bottleneck = {}
for name, bw, fp16, n, d, (lo, hi), moe, k, a in BOTTLENECK_ROWS:
    step_reads = d * (1 + 0.25 * k) if moe else d
    gb_per_token = step_reads / a
    gflop_per_token = 2 * n * (k + 1) / a
    bottleneck[name] = {
        'gb_per_token': gb_per_token, 'gflop_per_token': gflop_per_token,
        'bandwidth_utilization': [lo * gb_per_token / bw, hi * gb_per_token / bw],
        'fp16_compute_utilization': [lo * gflop_per_token / 1000 / fp16, hi * gflop_per_token / 1000 / fp16],
    }
results['decode_bottleneck'] = bottleneck
results['moe_drafting_read_bound'] = {
    'two_spark_glmflash_structured_tps': 99.44, 'tokens_per_step': 8, 'step_ms': 80.05,
    'max_step_reads_over_single_token_reads': 546 * 0.08005 / 9.5,
}
results['prefill_compute_utilization_fp16'] = {
    'rtx_pro_6000_flashnext_12000tps': 12000 * 12 / 1000 / 504,
    'spark_flashnext_1606tps': 1606 * 12 / 1000 / 125,
    'm5max_flashnext_1094tps_65k': 1094 * 12 / 1000 / 70,
    'm5_qwen36_1144tps_4k': 1144 * 6 / 1000 / 16,
}
out = ROOT / 'data'
out.mkdir(exist_ok=True)
(out / 'calculations.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8', newline='\n')
print('Wrote data/calculations.json')
