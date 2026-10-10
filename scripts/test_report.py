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
