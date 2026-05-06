from pathlib import Path
import re

EXTS = {'.md', '.py', '.js', '.yaml', '.yml', '.txt'}
IGNORE_DIRS = {'.git', '.venv', 'venv', '__pycache__', '.pytest_cache', 'runs', 'node_modules'}
BAD_TERMS = [
    'p' + 'seudocode',
    'p' + 'suedocode',
    'pse' + 'udo' + '-code',
    'pse' + 'udo' + ' code',
    '`' * 3 + 'pse' + 'udo',
    'algo' + 'rithm' + ' sk' + 'etch',
    'Not' + 'Implemented' + 'Error',
    'place' + 'holder',
    'st' + 'ub',
]


def test_repo_has_no_banned_terms():
    root = Path(__file__).resolve().parents[1]
    patt = re.compile('|'.join(re.escape(x) for x in BAD_TERMS), re.IGNORECASE)
    hits = []
    for p in root.rglob('*'):
        if not p.is_file() or p.suffix.lower() not in EXTS:
            continue
        if any(part in IGNORE_DIRS for part in p.parts):
            continue
        txt = p.read_text(encoding='utf-8', errors='ignore')
        if p.name == Path(__file__).name:
            txt = txt.replace('p' + 'seudocode', '').replace('p' + 'suedocode', '')
        m = patt.search(txt)
        if m:
            hits.append(f"{p.relative_to(root)}:{m.group(0)}")
    assert not hits, 'Found banned terms: ' + '; '.join(hits)
