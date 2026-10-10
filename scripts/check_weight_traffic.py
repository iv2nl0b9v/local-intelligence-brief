"""Reproduce parameter-read accounting from captured safetensors headers, offline."""
from collections import Counter
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
totals = Counter()
for path in (ROOT / 'data/weight-headers').glob('*.header.json'):
    for name, tensor in json.loads(path.read_text()).items():
        if name == '__metadata__':
            continue
        if path.name.startswith('experts'):
            category = 'routed_one_layer_all_experts'
        elif 'mtp' in name:
            category = 'mtp_excluded'
        elif 'visual' in name:
            category = 'vision_excluded'
        elif 'embed_tokens' in name:
            category = 'input_embedding_excluded_full_table'
        elif '.ple.' in name:
            category = 'ple_side_projection'
        else:
            category = 'target_dense_and_output'
        totals[category] += tensor['data_offsets'][1] - tensor['data_offsets'][0]
audit = json.loads((ROOT / 'data/weight-traffic-audit.json').read_text())
if dict(totals) != audit['category_bytes']:
    raise SystemExit('Weight category totals differ from audit')
active = totals['routed_one_layer_all_experts'] * 48 * 10 / 512
traffic = totals['target_dense_and_output'] + totals['ple_side_projection'] + active
if traffic != audit['estimated_target_weight_bytes_per_pass']:
    raise SystemExit('Traffic total differs from audit')
config = json.loads((ROOT / 'data/configurations.json').read_text(encoding='utf-8'))['pro6000']
if abs(config['weight_gb_per_pass'] - traffic / 1e9) > 1e-10:
    raise SystemExit('Headline traffic differs from header accounting')
print(f'Confirmed {traffic / 1e9:.9f} GB/pass parameter-read estimate (not measured DRAM traffic).')
