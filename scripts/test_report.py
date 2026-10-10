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

    def test_unknown_traffic_and_forecasts_have_no_attainment(self):
        from roofline import build
        rows, _, _ = build()
        for key, r in rows.items():
            if key != 'pro6000':
                self.assertIsNone(r['bandwidth_attainment'])
        self.assertAlmostEqual(rows['pro6000']['bandwidth_attainment'][0], .4497992612276786)
        self.assertEqual(rows['spark2']['display'], '47–63')
        self.assertIn('EXL3', rows['spark2']['note'])

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
