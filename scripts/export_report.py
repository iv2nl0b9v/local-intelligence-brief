"""Export the HTML tables and evidence, and validate report invariants."""
from html.parser import HTMLParser
from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parents[1]


class ReportParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables, self.links, self.ids, self.fragments = [], [], [], []
        self.table = self.row = self.cell = None
        self.evidence = False
        self.notes = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            self.ids.append(attrs['id'])
        if tag == 'a':
            href = attrs.get('href', '')
            if href.startswith('http'):
                self.links.append(href)
            elif href.startswith('#'):
                self.fragments.append(href[1:])
        if tag == 'aside':
            self.evidence = True
        if tag == 'table':
            self.table = []
        elif tag == 'tr':
            self.row = []
        elif tag in ('td', 'th'):
            self.cell = []
        elif tag in ('br', 'span', 'div') and self.cell is not None:
            self.cell.append(' ')

    def handle_endtag(self, tag):
        if self.evidence and tag in ('td', 'th'):
            self.notes.append('\t')
        if tag in ('td', 'th') and self.cell is not None:
            self.row.append(' '.join(''.join(self.cell).split()))
            self.cell = None
        elif tag == 'tr' and self.row is not None:
            self.table.append(self.row)
            self.row = None
        elif tag == 'table':
            self.tables.append(self.table)
            self.table = None
        if self.evidence and tag in ('p', 'li', 'h3', 'tr', 'summary'):
            self.notes.append('\n\n')
        if tag == 'aside':
            self.evidence = False

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)
        if self.evidence:
            self.notes.append(data)


def main():
    parser = ReportParser()
    parser.feed((ROOT / 'index.html').read_text(encoding='utf-8'))
    hardware, models = parser.tables[:2]
    assert len(hardware) == 15 and all(len(r) == 7 for r in hardware)
    assert len(models) == 10 and all(len(r) == 4 for r in models)
    assert len(parser.ids) == len(set(parser.ids)), 'Duplicate element IDs'
    assert set(parser.fragments) <= set(parser.ids), 'Broken fragment links'
    scores = {'Qwen3.6-35B-A3B': 18, 'Qwen3.8-27B': 34,
              'Qwen3.8-Flash-Next': 40, 'GLM-5.3-Flash': 42, 'GLM-5.3': 45}
    keys = []
    for row in hardware[1:]:
        model = next(m for m in scores if row[5].startswith(m))
        speed = re.match(r'(\d+(?:\.\d+)?)(?:–(\d+(?:\.\d+)?))?', row[6])
        lo = float(speed[1])
        hi = float(speed[2]) if speed[2] else lo
        assert hi >= 35, f'Speed gate failed: {row[0]}'
        keys.append((scores[model], (lo + hi) / 2))
    assert keys == sorted(keys), 'Hardware ordering is incorrect'
    model_scores = [int(re.match(r'\d+', row[1])[0]) for row in models[1:]]
    assert model_scores == sorted(model_scores, reverse=True)
    out = ROOT / 'data'
    out.mkdir(exist_ok=True)
    for name, table in [('hardware', hardware), ('models', models)]:
        records = [dict(zip(table[0], row)) for row in table[1:]]
        (out / f'{name}.json').write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (out / 'sources.json').write_text(json.dumps(sorted(set(parser.links)), indent=2) + '\n', encoding='utf-8')
    notes = re.sub(r'\n{3,}', '\n\n', ''.join(parser.notes)).strip()
    (out / 'research-notes.txt').write_text(notes + '\n', encoding='utf-8')
    print('Validated 14 devices, 9 models, score/speed ordering, upper-bound gate and internal links. Exported data/.')


if __name__ == '__main__':
    main()
