"""Capture official model configs with source URLs and content hashes."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from urllib.request import Request, urlopen
import json

ROOT = Path(__file__).resolve().parents[1]
REPOS = ['Qwen/Qwen3.8-27B', 'Qwen/Qwen3.8-Flash-Next', 'Qwen/Qwen3.6-35B-A3B',
         'zai-org/GLM-5.3', 'zai-org/GLM-5.3-Flash',
         'deepseek-ai/DeepSeek-V4-Flash-0731', 'deepseek-ai/DeepSeek-V4.1-Flash']


def capture(repo):
    url = f'https://huggingface.co/{repo}/resolve/main/config.json'
    record = {'repo': repo, 'url': url, 'captured_at_utc': datetime.now(timezone.utc).isoformat()}
    try:
        with urlopen(Request(url, headers={'User-Agent': 'local-intelligence-brief/1'}), timeout=60) as response:
            content = response.read()
            record['resolved_url'] = response.url
            record['upstream_commit'] = response.headers.get('x-repo-commit')
        json.loads(content)
        relative = f'data/configs/{repo.replace("/", "--")}.json'
        (ROOT / relative).write_bytes(content)
        record.update(path=relative, sha256=sha256(content).hexdigest(), status='captured')
    except Exception as exc:
        record.update(status='failed', error=str(exc))
    return record


if __name__ == '__main__':
    (ROOT / 'data/configs').mkdir(parents=True, exist_ok=True)
    manifest = ROOT / 'data/config-manifest.json'
    previous = {r['repo']: r for r in json.loads(manifest.read_text(encoding='utf-8'))} if manifest.exists() else {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(capture, REPOS))
    # A failed refresh leaves the old snapshot on disk, so keep the record that describes it.
    for i, record in enumerate(records):
        old = previous.get(record['repo'])
        if record['status'] != 'captured' and old and old.get('status') == 'captured':
            records[i] = {**old, 'last_refresh_error': record['error'],
                          'last_refresh_attempt_utc': record['captured_at_utc']}
    manifest.write_text(json.dumps(records, indent=2) + '\n', encoding='utf-8', newline='\n')
    for record in records:
        print(record['repo'], 'refresh failed; kept previous snapshot' if 'last_refresh_error' in record else record['status'])
    if any(record['status'] != 'captured' or 'last_refresh_error' in record for record in records):
        raise SystemExit('Some configs could not be captured; see manifest.')
