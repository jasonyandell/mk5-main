#!/usr/bin/env python
"""wiki_lint.py — mechanical health checks for wiki/.

Every convention in wiki/AGENTS.md that can be checked mechanically, is.
Report mode (default) prints findings grouped by check. Strict mode exits
nonzero if any ERROR-level findings remain, for use as a quality gate once
the corpus is clean.

Usage:
    python -u scripts/wiki_lint.py             # full report
    python -u scripts/wiki_lint.py --strict    # exit 1 on ERROR findings
    python -u scripts/wiki_lint.py --check IX  # run only checks with this prefix

Checks:
    FM01  page missing frontmatter block
    FM02  frontmatter missing a required field (title, kind, status, last_updated)
    FM03  kind does not match directory
    FM04  status outside the enum {active, complete, retired, superseded}
    FM05  last_updated is neither a short sha nor YYYY-MM-DD
    LN01  dead link — [[target]] resolves to no file (ERROR)
    LN02  ambiguous bare name — two files share a basename (ERROR)
    LN03  path-qualified [[dir/page]] link in body text (should be bare)
    OR01  content page with zero inbound links from other content pages
    RT01  experiments/ page not routed — no inbound link from entities/ or
          trails/, directly or via one rollup hop
    IX01  page on disk missing from index.md (ERROR)
    IX02  index.md entry pointing at no file (ERROR)
    LG01  log.md holds more than 15 entries (rotation overdue)
    SC01  sources/ file never linked from any page

AGENTS.md is excluded (it contains placeholder links by design).
log-archive.md is frozen history: its links are reported informationally,
never as errors, and it does not count toward inbound links.
"""

import argparse
import collections
import os
import re
import sys

WIKI = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "wiki")

CONTENT_DIRS = ("entities/", "topics/", "experiments/", "decisions/", "trails/", "playbooks/")
KIND_FOR_DIR = {
    "entities": "entity", "topics": "topic", "experiments": "experiment",
    "decisions": "decision", "trails": "trail", "playbooks": "playbook",
    "sources": "source",
}
STATUS_ENUM = {"active", "complete", "retired", "superseded"}
REQUIRED_FIELDS = ("title", "kind", "status", "last_updated")
EXCLUDED = {"AGENTS.md"}          # placeholder links by design
FROZEN = {"log-archive.md"}       # history; report-only

FM_RE = re.compile(r"\A---\n(.*?)\n---\n", re.S)
LINK_RE = re.compile(r"\[\[([^\]|#\n]+)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]")
FENCE_RE = re.compile(r"^```.*?^```", re.S | re.M)
SHA_RE = re.compile(r"^[0-9a-f]{7,10}$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def base(name):
    """Basename without a trailing .md — never strip other dots (qwen3-1.7b)."""
    b = os.path.basename(name)
    return b[:-3] if b.endswith(".md") else b


def load():
    pages = {}
    for root, _, files in os.walk(WIKI):
        for f in files:
            if f.endswith(".md"):
                rel = os.path.relpath(os.path.join(root, f), WIKI)
                with open(os.path.join(root, f)) as fh:
                    pages[rel] = fh.read()
    return pages


def links_of(text):
    body = FM_RE.sub("", text)
    body = FENCE_RE.sub("", body)
    return [m.group(1).strip() for m in LINK_RE.finditer(body)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--check", default="", help="only run checks whose code starts with this prefix")
    args = ap.parse_args()

    pages = load()
    findings = []  # (code, level, message)

    def add(code, level, msg):
        if args.check and not code.startswith(args.check):
            return
        findings.append((code, level, msg))

    basemap = collections.defaultdict(set)
    for rel in pages:
        basemap[base(rel)].add(rel)

    # --- frontmatter ---
    fm = {}
    for rel, text in sorted(pages.items()):
        if rel in EXCLUDED or rel in FROZEN or rel == "index.md" or rel.startswith("questions/"):
            continue
        m = FM_RE.match(text)
        if not m:
            if rel.startswith(CONTENT_DIRS) or rel.startswith("sources/"):
                add("FM01", "ERROR", rel)
            continue
        fields = dict(re.findall(r"^(\w[\w_-]*):\s*(.*)$", m.group(1), re.M))
        fm[rel] = fields
        if not rel.startswith(CONTENT_DIRS):
            continue
        for req in REQUIRED_FIELDS:
            if req not in fields:
                add("FM02", "ERROR", f"{rel}: missing '{req}'")
        expect = KIND_FOR_DIR[rel.split("/")[0]]
        kind = fields.get("kind", "").strip()
        if kind and kind != expect:
            add("FM03", "ERROR", f"{rel}: kind '{kind}' in {rel.split('/')[0]}/")
        status = fields.get("status", "").strip().strip('"')
        if status and status not in STATUS_ENUM:
            add("FM04", "ERROR", f"{rel}: status '{status}'")
        lu = fields.get("last_updated", "").strip().strip('"')
        if lu and not (SHA_RE.match(lu) or DATE_RE.match(lu)):
            add("FM05", "ERROR", f"{rel}: last_updated '{lu}'")

    # --- link graph ---
    inbound = collections.Counter()          # from live pages (not index/log/archive)
    inbound_router = collections.defaultdict(set)  # experiments page -> linking entities/trails pages
    inbound_any = collections.Counter()      # from anything at all
    for rel, text in pages.items():
        if rel in EXCLUDED:
            continue
        src_is_live = rel not in FROZEN and rel not in ("index.md", "log.md")
        for target in links_of(text):
            tbase = base(target)
            hits = basemap.get(tbase, set())
            if not hits:
                if rel in FROZEN:
                    add("LN01", "INFO", f"[[{target}]] in {rel} (frozen archive)")
                else:
                    add("LN01", "ERROR", f"[[{target}]] in {rel}")
                continue
            if len(hits) > 1:
                add("LN02", "ERROR", f"[[{tbase}]] matches {sorted(hits)}")
            for h in hits:
                inbound_any[h] += 1
                if src_is_live and h != rel:
                    inbound[h] += 1
                    if h.startswith("experiments/") and rel.startswith(("entities/", "trails/")):
                        inbound_router[h].add(rel)
            if "/" in target and src_is_live and rel != "index.md":
                add("LN03", "WARN", f"{rel}: [[{target}]]")

    # --- orphans and routing ---
    for rel in sorted(pages):
        if not rel.startswith(CONTENT_DIRS):
            continue
        if inbound[rel] == 0:
            add("OR01", "WARN", rel)
    routed = set(inbound_router)
    for rel in sorted(pages):
        if not rel.startswith("experiments/"):
            continue
        if rel in routed:
            continue
        # one rollup hop: linked from an experiments/ page that is itself routed
        hop = any(
            src in routed
            for src, text in pages.items() if src.startswith("experiments/")
            for t in links_of(text) if rel in basemap.get(base(t), set())
        )
        if not hop:
            add("RT01", "WARN", rel)

    # --- index coverage (bidirectional) ---
    idx_targets = {base(t) for t in links_of(pages.get("index.md", ""))}
    for rel in sorted(pages):
        if rel in EXCLUDED or rel in FROZEN or rel in ("index.md", "log.md"):
            continue
        if base(rel) not in idx_targets:
            add("IX01", "ERROR", rel)
    for t in sorted(idx_targets):
        if t not in basemap:
            add("IX02", "ERROR", f"[[{t}]]")

    # --- log rotation ---
    n_entries = len(re.findall(r"^## \[", pages.get("log.md", ""), re.M))
    if n_entries > 15:
        add("LG01", "ERROR", f"log.md has {n_entries} entries (rotate at 15)")

    # --- unlinked sources ---
    for rel in sorted(pages):
        if rel.startswith("sources/") and inbound_any[rel] == 0:
            add("SC01", "WARN", rel)

    # --- report ---
    by_code = collections.defaultdict(list)
    for code, level, msg in findings:
        by_code[code].append((level, msg))
    for code in sorted(by_code):
        rows = by_code[code]
        levels = collections.Counter(l for l, _ in rows)
        print(f"\n== {code} — {len(rows)} finding(s) ({dict(levels)}) ==")
        for level, msg in rows[:50]:
            print(f"  [{level}] {msg}")
        if len(rows) > 50:
            print(f"  ... and {len(rows) - 50} more")

    n_err = sum(1 for _, lvl, _ in findings if lvl == "ERROR")
    n_warn = sum(1 for _, lvl, _ in findings if lvl == "WARN")
    print(f"\n{len(pages)} pages checked: {n_err} errors, {n_warn} warnings")
    if args.strict and n_err:
        sys.exit(1)


if __name__ == "__main__":
    main()
