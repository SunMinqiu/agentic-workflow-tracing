"""Rebuild task KVCache sections and migrate superseded report links."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from agent_io_tracing.analysis.kvcache.catalog import Catalog
from agent_io_tracing.serving.kv_report_data import prepare_report_data


def migration_targets(root, catalog):
    targets = {}
    pages = sorted(set((root/'replay').rglob('kvcache_report.html')) | set((root/'replay').rglob('kv_block_report.html')))
    for page in pages:
        if page.name == 'kv_block_report.html':
            row = catalog.by_rep.get(page.parent.resolve())
            if row:
                targets[(page.resolve(), '')] = (catalog.page(row), row['id'])
                for fragment in re.findall(r'\bid="([^"]+)"', page.read_text()):
                    targets[(page.resolve(), fragment)] = (catalog.page(row), row['id'] + '-' + fragment)
        else:
            records = []
            for source in page.parent.glob('*/rep*/summary.json'):
                row = catalog.by_rep.get(source.parent.resolve())
                if row:
                    records.append(row)
                    arm = source.parent.parent.name
                    old = 'arm-' + re.sub(r'[^a-zA-Z0-9_-]', '-', arm) + '-' + hashlib.sha256(arm.encode()).hexdigest()[:8]
                    targets[(page.resolve(), old)] = (catalog.page(row), row['arm_id'])
            if records:
                sources = {r['source'] for r in records}
                dest = catalog.page(records[0]) if len(sources) == 1 else root/'results/all-experiments.html'
                targets[(page.resolve(), '')] = (dest, 'replay-experiments' if len(sources) == 1 else '')
                targets[(page.resolve(), 'run-details')] = targets[(page.resolve(), '')]
    return pages, targets


def rewrite_links(root, targets, deleted_pages):
    changed = []
    deleted = {p.resolve() for p in deleted_pages}
    for pattern in ('*.html', '*.md'):
        for page in root.rglob(pattern):
            if page.resolve() in deleted or any(part in ('.git', 'node_modules') for part in page.parts):
                continue
            content = page.read_text(encoding='utf-8')
            if 'kvcache_report.html' not in content and 'kv_block_report.html' not in content:
                continue
            def rewrite(raw):
                parsed = urlsplit(html.unescape(raw))
                if parsed.scheme or parsed.netloc:
                    return raw
                dest = (page.parent / unquote(parsed.path)).resolve()
                target = targets.get((dest, unquote(parsed.fragment))) or targets.get((dest, ''))
                if not target:
                    return raw
                return os.path.relpath(target[0], page.parent) + ('#' + target[1] if target[1] else '')
            updated = re.sub(r'(href=")([^"]+)(")', lambda m: m[1] + html.escape(rewrite(m[2]), quote=True) + m[3], content)
            updated = re.sub(r'(\]\()([^\s)]+)(\))', lambda m: m[1] + rewrite(m[2]) + m[3], updated)
            if updated != content:
                page.write_text(updated,encoding='utf-8')
                changed.append(str(page.relative_to(root)))
    return changed


def rebuild(root: Path) -> dict:
    root = root.resolve()
    for rep in sorted({p.parent.resolve() for p in (root/'replay').rglob('rep*/summary.json')}):
        prepare_report_data(rep)
    catalog = Catalog(root)
    old_pages, targets = migration_targets(root, catalog)
    from agent_io_tracing.analysis.kvcache.replay_report import publish_catalog
    publish_catalog(catalog)
    removable = [p for p in old_pages if (p.resolve(),'') in targets]
    changed = rewrite_links(root,targets,removable)
    actions = []
    for page in removable:
        dest, fragment = targets[(page.resolve(),'')]
        actions.append({'old':str(page.relative_to(root)), 'replacement':str(dest.relative_to(root)) + ('#'+fragment if fragment else '')})
        page.unlink()
    manifest_path = root/'output/kv_report_integration_manifest.json'
    manifest_path.parent.mkdir(exist_ok=True)
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    merged = {a['old']:a for a in previous.get('removed_reports',[]) + actions}
    manifest = {'removed_obsolete_artifacts':previous.get('removed_obsolete_artifacts',[]),'removed_reports':list(merged.values()),'rewritten_files':sorted(set(previous.get('rewritten_files',[])+changed)),
                'runs': [{'rep':str(r['rep'].relative_to(root)), 'source':str(r['source'].relative_to(root)) if r['source'] else r['source_kind'], 'target':catalog.href(r,root)} for r in catalog.records],
                'unmigrated_reports':[str(p.relative_to(root)) for p in old_pages if p not in removable]}
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args()
    result=rebuild(args.root)
    print(f"Integrated {len(result['runs'])} runs; removed {len(result['removed_reports'])} superseded reports")
