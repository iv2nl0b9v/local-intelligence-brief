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
        self.names, self.strong = [], None  # first <strong> text per table, used as each row's name
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
        if tag == 'strong' and self.row is not None and not self.row:
            self.strong = []
        if tag == 'table':
            self.table = []
            self.names.append([])
        elif tag == 'tr':
            self.row = []
        elif tag in ('td', 'th'):
            self.cell = []
        elif tag in ('br', 'span', 'div') and self.cell is not None:
            self.cell.append(' ')

    def handle_endtag(self, tag):
        if tag == 'strong' and self.strong is not None:
            self.names[-1].append(''.join(self.strong).strip())
            self.strong = None
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
        if self.strong is not None:
            self.strong.append(data)
        if self.cell is not None:
            self.cell.append(data)
        if self.evidence:
            self.notes.append(data)


def check(condition, message):
    # Explicit checks rather than assert, so validation still runs under python -O.
    if not condition:
        raise SystemExit(f'Validation failed: {message}')


def parse_score(cell):
    return int(re.match(r'\d+', cell)[0])


def main():
    parser = ReportParser()
    parser.feed((ROOT / 'index.html').read_text(encoding='utf-8'))
    hardware, models = parser.tables[:2]
    check(all(len(r) == 8 for r in hardware), 'hardware rows must have 8 cells')
    check(all(len(r) == 4 for r in models), 'model rows must have 4 cells')
    check(len(parser.ids) == len(set(parser.ids)), 'duplicate element IDs')
    check(set(parser.fragments) <= set(parser.ids), 'broken fragment links')
    model_scores = [parse_score(row[1]) for row in models[1:]]
    check(model_scores == sorted(model_scores, reverse=True), 'model table is not sorted by score')

    # Hardware pairings may only use local models; cloud rows are trajectory references.
    model_names = parser.names[1]
    check(len(model_names) == len(models) - 1, 'every model row needs a <strong> name')
    scores = {name: parse_score(row[1]) for name, row in zip(model_names, models[1:])
              if 'cloud' not in row[0].lower()}
    check(len(scores) == sum('cloud' not in row[0].lower() for row in models[1:]), 'duplicate model names')
    keys = []
    for row in hardware[1:]:
        # Longest prefix, so 'GLM-5.3-Flash' is never scored as 'GLM-5.3'.
        names = [m for m in scores if row[5].startswith(m)]
        check(names, f'no shortlist model matches pairing: {row[5]}')
        model = max(names, key=len)
        speed = re.match(r'(\d+(?:\.\d+)?)(?:–(\d+(?:\.\d+)?))?', row[6])
        check(speed, f'unparseable speed: {row[6]}')
        lo = float(speed[1])
        hi = float(speed[2]) if speed[2] else lo
        check(hi >= 35, f'speed gate failed: {row[0]}')
        keys.append((scores[model], (lo + hi) / 2))
    check(keys == sorted(keys), 'hardware ordering is incorrect')

    # Displayed estimate ranges must be the computed endpoints rounded to the nearest integer.
    calculations = json.loads((ROOT / 'data' / 'calculations.json').read_text(encoding='utf-8'))
    for name, scenario in calculations.items():
        if isinstance(scenario, dict) and 'displayed' in scenario:
            computed = [round(x) for x in scenario['endpoints']]
            check(scenario['displayed'] == computed,
                  f'{name}: displayed {scenario["displayed"]} != rounded endpoints {computed}')

    # Roofline cells, chart and table must match scripts/roofline.py exactly.
    import roofline
    rows, refs, pre = roofline.build()
    html = (ROOT / 'index.html').read_text(encoding='utf-8')
    check(roofline.render(html, rows, refs, pre) == html, 'roofline content is stale; run python scripts/roofline.py')

    out = ROOT / 'data'
    out.mkdir(exist_ok=True)

    def write(name, text):
        (out / name).write_text(text, encoding='utf-8', newline='\n')

    for name, table in [('hardware', hardware), ('models', models)]:
        records = [dict(zip(table[0], row)) for row in table[1:]]
        write(f'{name}.json', json.dumps(records, ensure_ascii=False, indent=2) + '\n')
    write('sources.json', json.dumps(sorted(set(parser.links)), indent=2) + '\n')
    notes = re.sub(r'\n{3,}', '\n\n', ''.join(parser.notes)).strip()
    write('research-notes.txt', notes + '\n')
    print(f'Validated {len(hardware) - 1} devices, {len(models) - 1} model rows, score/speed ordering, '
          'upper-bound gate, estimate rounding, roofline consistency and internal links. Exported data/.')


if __name__ == '__main__':
    main()
