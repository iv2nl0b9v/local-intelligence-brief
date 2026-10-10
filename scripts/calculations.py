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
# Conditional transfer scenario, not a measured bound: B17 includes 38.7 -> 47.3 at 1K,
# 32.9 -> 44.0 at 4K, and 38.8 -> 62.4 at 200 tokens. Full-model support is unverified.
DFLASH2_GAIN = [47.3 / 38.7, 62.4 / 38.8]
baseline = [split_memory(24, x, u, h) for x, u, h in [(.42, .5, .008), (.38, .7, .004)]]
results['station_glm53'] = {
    'baseline_no_drafting': baseline, 'dflash2_gain': DFLASH2_GAIN,
    'endpoints': [baseline[0] * DFLASH2_GAIN[0], baseline[1] * DFLASH2_GAIN[1]], 'displayed': [19, 40],
}
ultra = [1200 * .3 / 24, 1200 * .5 / 24]
results['ultra512_glm53'] = {
    'baseline_no_drafting': ultra, 'dflash2_gain': DFLASH2_GAIN,
    'endpoints': [ultra[0] * DFLASH2_GAIN[0], ultra[1] * DFLASH2_GAIN[1]], 'displayed': [18, 40],
    'cross_check_m3_ultra_glm52_oq4_scaled_to_1200': 14.5 * 1200 / 819,
}
mimo = [split_memory(23, x, u, h) for x, u, h in [(.63, .5, .008), (.60, .7, .004)]]
results['station_mimo_v26_pro_rejected'] = {
    'weights_gb': 566, 'cpu_resident_gb': [566 - 228, 566 - 212], 'cpu_traffic_fraction': [.60, .63],
    'baseline_no_drafting': mimo, 'endpoints': [mimo[0] * DFLASH2_GAIN[0], mimo[1] * DFLASH2_GAIN[1]],
    'displayed': [15, 29],
}

# These scenarios estimate speed; they do not diagnose bottlenecks.
out = ROOT / 'data'
out.mkdir(exist_ok=True)
(out / 'calculations.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8', newline='\n')
print('Wrote data/calculations.json')
