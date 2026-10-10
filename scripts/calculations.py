"""Reproduce planning arithmetic; this script is not a hardware benchmark."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
N = 131072
GB = 1e9


def rate(traffic_gb, bandwidth_gbps, utilization, overhead_ms):
    return 1 / (traffic_gb / (bandwidth_gbps * utilization) + overhead_ms / 1000)


results = {
    'units': {'memory': 'decimal GB', 'speed': 'tokens/second', 'context_tokens': N, 'concurrency': 1},
    'warning': 'Assumed traffic, utilization, overhead and drafting factors; not measurements or confidence intervals.',
    'pro_single_flashnext': {
        'traffic_gb': [5, 7], 'bandwidth_gbps': 1792, 'utilization': [.5, .7], 'overhead_ms': [4, 7],
        'central': rate(6, 1792, .6, 5), 'endpoints': [rate(7, 1792, .5, 7), rate(5, 1792, .7, 4)],
        'displayed': [65, 125], 'ideal_compute_roof': 2015e12 / 12e9,
    },
    'pro_dual_glmflash': {
        'traffic_gb': [10.8, 13], 'bandwidth_gbps': 3584, 'utilization': [.5, .65], 'overhead_ms': [5, 10],
        'central': rate(12, 3584, .6, 7), 'endpoints': [rate(13, 3584, .5, 10), rate(10.8, 3584, .65, 5)],
        'displayed': [60, 110], 'ideal_fp4_compute_roof_not_iq4_kernel_performance': 4030e12 / 36e9,
    },
    'pro_anchor': {'measured_tp2_flashnext_fp8': 81.45, 'measured_tp4': 64.61,
                   'tp2_latency_ms': 1000 / 81.45,
                   'residual_ms_for_6_to_10GB_at_60pct': [1000 / 81.45 - d / (3584 * .6) * 1000 for d in [6, 10]]},
    'm5_qwen35_moe': {'endpoints': [rate(2.5, 153, .6, 8), rate(2, 153, .8, 4)], 'displayed': [30, 50]},
    'm5pro_qwen38': {'fresh_coding': [52.6 * .5 * .8, 58.7 * .5], 'repetitive_editing': [87.6 * .5 * .8, 87.6 * .5], 'displayed': [21, 44]},
    'r9700_qwen38': {'baseline': [24, 32], 'assumed_net_drafting': [1.25, 1.6], 'endpoints': [24 * 1.25, 32 * 1.6], 'displayed': [30, 50]},
    'r9700_moe_fallback': {'endpoints': [rate(2.5, 640, .4, 6), rate(2, 640, .6, 4)], 'displayed': [65, 110]},
    'dual_spark_glmflash': {'endpoints': [546 * .3 * 1.3 / 10.8, 546 * .5 * 1.5 / 10.8], 'displayed': [20, 40]},
    'rtx4090_qwen27': {'endpoints': [1008 * .6 / 16, 1008 * .8 / 16], 'displayed': [38, 50]},
    'ultra512_rejected_glm53': {'endpoints': [1200 * .3 / 24, 1200 * .5 / 24], 'displayed': [15, 25]},
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
baseline = [1 / (24 * (1-x) / (7100*u) + 24*x / (396*u) + h)
            for x, u, h in [(.42, .5, .008), (.38, .7, .004)]]
results['station_glm53'] = {'baseline': baseline, 'conditional_drafting': [baseline[0]*1.4, baseline[1]*1.8], 'displayed': [22, 45]}
out = ROOT / 'data'
out.mkdir(exist_ok=True)
(out / 'calculations.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
print('Wrote data/calculations.json')
