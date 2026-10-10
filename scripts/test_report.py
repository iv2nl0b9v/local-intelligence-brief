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
            expected=[1-max(m,c) for m,e in zip(r['bandwidth_attainment'],r['endpoints'])
                      for c in e['compute']['demand_range']]
            self.assertEqual(r['roof_gap'],expected)
            self.assertTrue(all(c>0 for c in r['compute_attainment']))
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

    def test_speculative_work_and_precision_costs(self):
        from compute_roofline import target_components, draft_components, component_costs, INPUTS
        one=sum(x for _,x,_ in target_components('glmflash',133,1))
        six=sum(x for _,x,_ in target_components('glmflash',133,6))
        self.assertAlmostEqual(six,6*one)  # rejected proposals still require target work
        self.assertEqual(draft_components('none','glmflash',7,3,133),[])
        mtp1=sum(x for _,x,_ in draft_components('mtp','flashnext',1,2,1024))
        mtp3=sum(x for _,x,_ in draft_components('mtp','flashnext',3,2,1024))
        self.assertAlmostEqual(mtp3,3*mtp1)
        # Adaptive verify length does not multiply or shrink a fixed DFlash forward.
        self.assertEqual(draft_components('dflash','glmflash',4,3,133,8),
                         draft_components('dflash','glmflash',7,3,133,8))
        parts=[('dense',100,'dense'),('expert',200,'expert')]
        costs=component_costs(parts,'pro6000','fp4_mixed')
        self.assertEqual([c['capacity_key'] for c in costs],['f16_matrix','fp4_matrix'])
        expected=100/503.8+200/2015.2
        self.assertAlmostEqual(sum(c['ideal_ms'] for c in costs),expected)
        self.assertGreater(expected,300/2015.2) # cannot apply FP4 peak to BF16 weights
        self.assertEqual(INPUTS['devices']['pro2']['capacity_tops'],
                         INPUTS['devices']['pro6000']['capacity_tops']) # sequential layer split

    def test_spark_compute_uses_matched_requests(self):
        from compute_roofline import estimate
        from statistics import median
        raw=json.loads((ROOT/'data/spark-decode-audit.json').read_text())['workloads'][0]['runs']
        result=estimate('spark2','spark2','glmflash',133,99,99,999,0)
        self.assertEqual(len(result['scenarios']),len(raw))
        for c,r in zip(result['scenarios'],raw):
            self.assertEqual(c['k'],r['spec']['draft_tokens']/r['spec']['drafts'])
            self.assertEqual(c['emitted'],(r['completion_tokens']-1)/r['spec']['drafts'])
            self.assertEqual(c['draft_block_rows'],7)
            self.assertTrue(all(x['capacity_key']=='f16_matrix' for x in c['components']))
            # GOps / TOp/s is ms; normalize to delivered decode tokens, not proposals.
            ms=sum(x['gops']/x['capacity_tops'] for x in c['components'])/c['emitted']
            self.assertAlmostEqual(c['demand'],r['tok_s']*ms/1000)
        self.assertEqual(result['demand_range'],[median(c['demand'] for c in result['scenarios'])])


if __name__ == '__main__':
    unittest.main()
