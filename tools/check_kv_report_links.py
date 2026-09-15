"""Check local targets and fragment IDs in the rebuilt KV report chain."""
from __future__ import annotations

import argparse
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.targets = []
        self.ids = set()
        self.duplicate_ids = set()
        self.in_evidence = False
        self.embedded = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'script' and attrs.get('class') == 'kv-evidence-data':
            self.in_evidence = True
        if 'id' in attrs:
            if attrs['id'] in self.ids:
                self.duplicate_ids.add(attrs['id'])
            self.ids.add(attrs['id'])
        if tag in {'a', 'img'}:
            target = attrs.get('href' if tag == 'a' else 'src')
            if target:
                self.targets.append(target)

    def handle_endtag(self, tag):
        if tag == 'script':
            self.in_evidence = False

    def handle_data(self, data):
        if self.in_evidence:
            self.embedded.append(data)

    def parse(self, content):
        self.feed(content)
        for raw in self.embedded:
            nested = Links()
            nested.feed(json.loads(raw))
            self.duplicate_ids.update(self.ids & nested.ids)
            self.duplicate_ids.update(nested.duplicate_ids)
            self.ids.update(nested.ids)
            self.targets.extend(nested.targets)
        return self


def check(root: Path) -> dict:
    pages = [root / 'results/index.html', root / 'results/all-experiments.html']
    pages += list((root / 'results').rglob('kvcache_report.html'))
    cache = {}
    def parsed(path):
        if path not in cache:
            parser = Links()
            parser.parse(path.read_text(encoding='utf-8'))
            cache[path] = parser
        return cache[path]
    errors = []
    count = 0
    for page in pages:
        if not page.is_file():
            errors.append({'page': str(page), 'error': 'missing report'})
            continue
        parser = parsed(page.resolve())
        for duplicate in parser.duplicate_ids:
            errors.append({'page': str(page.relative_to(root)), 'error': f'duplicate ID: {duplicate}'})
        for target in parser.targets:
            url = urlsplit(target)
            if url.scheme or url.netloc:
                continue
            count += 1
            dest = (page.parent / unquote(url.path)).resolve() if url.path else page.resolve()
            if not dest.exists():
                errors.append({'page': str(page.relative_to(root)), 'target': target, 'error': 'missing target'})
            elif url.fragment and dest.is_file() and dest.suffix == '.html' and unquote(url.fragment) not in parsed(dest).ids:
                errors.append({'page': str(page.relative_to(root)), 'target': target, 'error': 'missing fragment'})
    from agent_io_tracing.analysis.kvcache.catalog import Catalog, eviction_available
    catalog = Catalog(root)
    for row in catalog.records:
        page = catalog.page(row)
        if not page.is_file():
            errors.append({'run':str(row['rep'].relative_to(root)), 'error':'missing owning report'})
            continue
        ids = parsed(page.resolve()).ids
        fragments = [row['id'], row['arm_id']]
        if eviction_available(catalog.data(row)):
            fragments.append(row['id']+'-eviction')
        for fragment in fragments:
            if fragment not in ids:
                errors.append({'run':str(row['rep'].relative_to(root)), 'error':'missing run target', 'fragment':fragment})
        for other in pages:
            if other.resolve() != page.resolve() and row['id'] in parsed(other.resolve()).ids:
                errors.append({'run':str(row['rep'].relative_to(root)), 'error':'run duplicated in another report', 'page':str(other.relative_to(root))})
    return {'pages': len(pages), 'local_links': count, 'runs_checked':len(catalog.records), 'errors': errors}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    result = check(args.root.resolve())
    print(json.dumps(result, indent=2))
    raise SystemExit(bool(result['errors']))
