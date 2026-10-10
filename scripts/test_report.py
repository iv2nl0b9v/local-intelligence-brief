"""Regression checks for the actual failure modes found in the report audit."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class AuditRegression(unittest.TestCase):
    def test_changed_headline_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix='.audit-test-', dir=ROOT) as tmp:
            root = Path(tmp)
            shutil.copytree(ROOT / 'scripts', root / 'scripts')
            shutil.copytree(ROOT / 'data', root / 'data')
            html = (ROOT / 'index.html').read_text(encoding='utf-8')
            self.assertIn('<strong>41–70</strong>', html)
            (root / 'index.html').write_text(html.replace('<strong>41–70</strong>', '<strong>41–71</strong>'), encoding='utf-8')
            p = subprocess.run([sys.executable, str(root / 'scripts/export_report.py')], capture_output=True, text=True, encoding='utf-8')
            self.assertNotEqual(p.returncode, 0)
            self.assertIn('HTML headline differs', p.stderr)

    def test_check_is_read_only(self):
        paths = [ROOT / 'index.html', ROOT / 'data/roofline.json']
        before = [(p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]
        subprocess.run([sys.executable, str(ROOT / 'scripts/roofline.py'), '--check'], check=True)
        self.assertEqual(before, [(p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths])

    def test_rooflines_retain_all_pairs_without_causal_time_shares(self):
        from roofline import build
        rows, _, _ = build()
        self.assertEqual(len(rows), 14)
        for r in rows.values():
            self.assertEqual(len(r['bandwidth_attainment']), 2)
            self.assertNotIn('other_time_share', r)
            if r['estimate']:
                self.assertTrue(r['scenario_only'])
            for m, c, gap in zip(r['bandwidth_attainment'], r['compute_attainment'], r['roof_gap']):
                self.assertAlmostEqual(gap, 1-max(m,c))
        self.assertAlmostEqual(rows['pro6000']['inputs']['weight_gb_per_pass'], 9.95111452)
        self.assertTrue(.45 < rows['pro6000']['bandwidth_attainment'][0] < .46)
        self.assertGreater(rows['spark2']['bandwidth_demand_sensitivity'][1], 1)
        self.assertEqual(rows['spark2']['display'], '47–63')
        self.assertIn('EXL3', rows['spark2']['note'])

    def test_expert_reuse_and_profile_accounting(self):
        from roofline import step, build
        a = step('spark2','glmflash',10,133,True,5,3,.5,1)
        b = step('spark2','glmflash',10,133,True,5,3,.5,6)
        self.assertEqual(a['weight_gb_per_step'],10)
        self.assertEqual(b['weight_gb_per_step'],35)
        self.assertEqual(a['gops_per_token'],b['gops_per_token'])
        _,_,profile=build()
        for r in profile['rows']:
            self.assertLess(abs(r['bandwidth_ms']+r['nccl_ms']+r['small_ms']+r['idle_ms']-r['wall_ms']),.03)

    def test_raw_decode_arithmetic(self):
        from statistics import median
        data = json.loads((ROOT / 'data/spark-decode-audit.json').read_text())
        rates = []
        for workload in data['workloads']:
            for r in workload['runs']:
                self.assertAlmostEqual((r['completion_tokens']-1)/(r['wall_s']-r['ttft_s']), r['tok_s'])
                self.assertAlmostEqual(r['completion_tokens']/r['wall_s'], r['delivered_tok_s'])
            rates.append(median(r['tok_s'] for r in workload['runs']))
        canonical = json.loads((ROOT / 'data/configurations.json').read_text(encoding='utf-8'))
        for actual, expected in zip(rates, canonical['spark2']['speed']):
            self.assertAlmostEqual(actual, expected, places=5)


if __name__ == '__main__':
    unittest.main()
