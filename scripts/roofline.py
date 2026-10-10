"""Render conditional bandwidth attainment; never infer elapsed-time attribution."""
from pathlib import Path
from html import escape
import json, re, sys
ROOT = Path(__file__).resolve().parents[1]
def build():
    rows = json.loads((ROOT / 'data/configurations.json').read_text(encoding='utf-8'))
    for r in rows.values():
        d, bw = r['weight_gb_per_pass'], r['bandwidth_gbps']
        r['bandwidth_attainment'] = [s*d/bw for s in r['speed']] if d is not None and bw is not None and not r['estimate'] else None
    return rows, {}, {}
def attainment(r):
    v = r['bandwidth_attainment']
    return '—' if v is None else f'{min(v):.0%}–{max(v):.0%}'
def cells(key, r):
    return (f'<td data-roofline="{key}" class="tt">{attainment(r)}</td>'
            f'<td data-roofline-part class="tt"><a href="#roofline" onclick="document.getElementById(\'research-details\').open=true">{r["diagnosis"]}</a></td>')
def render(html, rows, refs, pre):
    block = '<table class="method-table"><thead><tr><th>Configuration</th><th>Decode tok/s</th><th>Conditional weights-only bandwidth attainment</th><th>Evidence</th></tr></thead><tbody>'
    for r in rows.values():
        block += f'<tr><td>{escape(r["device"])}<br>{escape(r["note"])}</td><td>{r["display"]}{" (scenario)" if r["estimate"] else ""}</td><td>{attainment(r)}</td><td>{r["diagnosis"]}</td></tr>'
    block += '</tbody></table>'
    html = re.sub(r'(<!-- roofline:start -->).*?(<!-- roofline:end -->)', lambda m:m[1]+block+m[2], html, flags=re.S)
    for key, r in rows.items():
        pattern = rf'<td data-roofline="{key}"[^>]*>.*?</td>(?:<td data-roofline-part[^>]*>.*?</td>)*'
        html, count = re.subn(pattern, lambda m:cells(key,r), html, flags=re.S)
        if count != 1: raise ValueError(f'Expected one evidence cell for {key}; found {count}')
    return html
def serialized(rows, refs, pre):
    return json.dumps({'method':'Conditional weights-only attainment; no time shares or residual attribution.', 'pairs':rows}, indent=2, ensure_ascii=False)+'\n'
def main():
    result = build()
    path = ROOT / 'index.html'
    html = path.read_text(encoding='utf-8')
    new, data = render(html,*result), serialized(*result)
    target = ROOT / 'data/roofline.json'
    if '--check' in sys.argv:
        if new != html or target.read_text(encoding='utf-8') != data: raise SystemExit('Validation failed: stale bandwidth evidence')
        print('Bandwidth evidence matches; no files changed.')
    else:
        path.write_text(new,encoding='utf-8',newline='\n')
        target.write_text(data,encoding='utf-8',newline='\n')
        print('Updated bandwidth evidence.')
if __name__ == '__main__': main()
