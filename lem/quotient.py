# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""Decision quotient of the saved Gus contingent-teacher course.

Reads every eligible root's teacher.json (read-only) from one study, rebuilds
the focal player's decision tree, and computes the coarsest right congruence
that preserves the decision interface of *A minimal comparison algebra for a
declared 42 family* (2026-10-21):

  state  = a focal information node (the Gus token history at a viewer decision)
  output = (legal menu, complete maximizing set)
  block  = own action followed by the observed plays up to the next focal
           decision; the final tile leads to HALT; absent blocks are REJECT

Two independent constructions must agree: bottom-up signature hashing and
Moore partition refinement from the output partition.  The refinement rounds
are the block-depth ladder.  Everything is exact (Fractions); no floats.

Interfaces (what a block label and an output are allowed to mention):
  raw    tiles as the student sees them: menu/best as tile ids, observation as
         (relative seat, tile) pairs.
  shape  no tile identity at all: menu size, best as slots (rank by tile id in
         the menu), and under each slot the *multiset* of successor classes.
         This counts distinct decision-problem shapes, not a controller.

Run: python -u -m lem.quotient <study> [--out report.json] [--all-solved]

Side audits performed on the saved rows alone (no native field, no models):
  * every non-root node has a parent row; every non-final node has at least one
    child under every legal action;
  * the children under an action partition the cell's mass exactly;
  * Bellman: q[a]*mass == sum over observations of child.mass * max(child.q).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

PREFIX = 13
YOU = 28
ROOT_PLY = 19


@dataclass
class Node:
    root_id: str
    split: str
    key: str
    tokens: tuple
    ply: int
    legal: tuple
    best: tuple
    q: dict
    mass: Fraction
    settled: bool
    parent: str | None = None
    block: tuple | None = None            # (action, observed pairs) from parent
    children: dict = field(default_factory=dict)   # block -> child key


def pairs(tokens):
    """The key is prefix, (seat, tile) pairs, then the acting seat's query token."""
    if tokens[-1] != YOU:
        raise ValueError("information key does not end at the focal seat's query")
    body = tokens[PREFIX:-1]
    if len(body) % 2:
        raise ValueError("odd play body")
    return tuple((body[i], body[i + 1]) for i in range(0, len(body), 2))


def load_root(directory: Path) -> dict[str, Node]:
    graph = json.loads((directory / "teacher.json").read_text())
    nodes = {}
    for row in graph["decisions"]:
        tokens = tuple(row["tokens"])
        node = Node(root_id=graph["root_id"], split=graph["split"], key=row["key"], tokens=tokens,
                    ply=row["ply"], legal=tuple(sorted(row["legal"])), best=tuple(sorted(row["optimal"])),
                    q={int(a): Fraction(v) for a, v in row["q"].items()}, mass=Fraction(row["mass"]),
                    settled=bool(row["settled"]))
        if set(node.q) != set(node.legal):
            raise ValueError(f"{node.root_id}: q support differs from legal menu")
        if not node.best or not set(node.best) <= set(node.legal):
            raise ValueError(f"{node.root_id}: maximizing set outside menu")
        top = max(node.q.values())
        if set(node.best) != {a for a, v in node.q.items() if v == top}:
            raise ValueError(f"{node.root_id}: saved maximizing set disagrees with saved q")
        if tokens in nodes:
            raise ValueError(f"{node.root_id}: duplicate information key")
        nodes[tokens] = node
    by_tokens = nodes
    # Link parents: the last own play at pair index >= ROOT_PLY starts the block.
    root = None
    for node in by_tokens.values():
        ps = pairs(node.tokens)
        if len(ps) == ROOT_PLY:
            if root is not None:
                raise ValueError("two roots")
            root = node
            continue
        own = [i for i in range(ROOT_PLY, len(ps)) if ps[i][0] == YOU]
        if not own:
            raise ValueError(f"{node.root_id}: non-root node without an own play after the root")
        i = own[-1]
        parent_tokens = node.tokens[: PREFIX + 2 * i] + (YOU,)
        parent = by_tokens.get(parent_tokens)
        if parent is None:
            raise ValueError(f"{node.root_id}: parent row missing for ply {node.ply}")
        action = ps[i][1]
        if action not in parent.legal:
            raise ValueError(f"{node.root_id}: block action not in parent's menu")
        block = (action, ps[i + 1:])
        if block in parent.children:
            raise ValueError(f"{node.root_id}: duplicate block")
        parent.children[block] = node.tokens
        node.parent, node.block = parent.tokens, block
    if root is None:
        raise ValueError("no root")
    if Fraction(graph["value"]) != max(root.q.values()):
        raise ValueError(f"{root.root_id}: graph value differs from root optimum")
    # Key the course by (root, history): histories may coincide across roots.
    rid = root.root_id
    keyed = {}
    for tokens, node in by_tokens.items():
        node.children = {b: (rid, c) for b, c in node.children.items()}
        node.parent = None if node.parent is None else (rid, node.parent)
        keyed[(rid, tokens)] = node
    return keyed


def audit(nodes: dict[str, Node]):
    """Exact checks on the saved rows alone."""
    for node in nodes.values():
        under = defaultdict(list)
        for (a, _o), child in node.children.items():
            under[a].append(nodes[child])
        if not node.children:
            # The final own tile: a singleton menu, nothing to decide afterwards.
            if len(node.legal) != 1:
                raise ValueError(f"{node.root_id}: childless node with a real menu at ply {node.ply}")
            continue
        for a in node.legal:
            kids = under.get(a)
            if not kids:
                raise ValueError(f"{node.root_id}: legal action {a} explored nowhere at ply {node.ply}")
            if sum((k.mass for k in kids), Fraction()) != node.mass:
                raise ValueError(f"{node.root_id}: children under {a} do not partition the mass")
            bellman = sum((k.mass * max(k.q.values()) for k in kids), Fraction())
            if bellman != node.q[a] * node.mass:
                raise ValueError(f"{node.root_id}: Bellman mismatch at ply {node.ply} action {a}")


# ---- interfaces -----------------------------------------------------------

def output_raw(node):
    return ("out", node.legal, node.best)


def output_shape(node):
    slots = {t: i for i, t in enumerate(node.legal)}
    return ("out", len(node.legal), tuple(slots[t] for t in node.best))


def successors_raw(node, nodes, cls):
    return tuple(sorted((block, cls[child]) for block, child in node.children.items()))


def successors_shape(node, nodes, cls):
    slots = {t: i for i, t in enumerate(node.legal)}
    per = [[] for _ in node.legal]
    for (a, _o), child in node.children.items():
        per[slots[a]].append(cls[child])
    return tuple(tuple(sorted(p)) for p in per)


def successors_fork(node, nodes, cls):
    """Per action slot, the *set* of successor classes: which next decisions the
    public evidence can lead to, without counting how many observations realize each."""
    slots = {t: i for i, t in enumerate(node.legal)}
    per = [set() for _ in node.legal]
    for (a, _o), child in node.children.items():
        per[slots[a]].add(cls[child])
    return tuple(tuple(sorted(p)) for p in per)


INTERFACES = {"raw": (output_raw, successors_raw), "shape": (output_shape, successors_shape),
              "fork": (output_shape, successors_fork)}


def signatures(nodes, out, succ):
    """Bottom-up: a node's signature is its output plus its labelled successor signatures."""
    sig = {}

    def visit(key):
        if key in sig:
            return sig[key]
        node = nodes[key]
        for child in node.children.values():
            visit(child)
        sig[key] = (out(node), succ(node, nodes, sig))
        return sig[key]

    for key in nodes:
        visit(key)
    return sig


def intern(values):
    """Map hashable values to dense class ids, deterministically by first appearance in sorted key order."""
    ids = {}
    return {k: ids.setdefault(v, len(ids)) for k, v in values.items()}


def refine(nodes, out, succ):
    """Moore partition refinement from the output partition; returns every round's classes.
    Round k distinguishes exactly what block sequences of length <= k can reveal."""
    rounds = [intern({k: out(n) for k, n in sorted(nodes.items())})]
    while True:
        cls = rounds[-1]
        new = intern({k: (cls[k], succ(n, nodes, cls)) for k, n in sorted(nodes.items())})
        rounds.append(new)
        if len(set(new.values())) == len(set(cls.values())):
            return rounds


def same_partition(a, b):
    blocks_a = defaultdict(set)
    blocks_b = defaultdict(set)
    for k in a:
        blocks_a[a[k]].add(k)
        blocks_b[b[k]].add(k)
    return set(map(frozenset, blocks_a.values())) == set(map(frozenset, blocks_b.values()))


def weak_order(node):
    """Dense rank of each menu slot by q, descending: the decision class of the node."""
    levels = sorted(set(node.q.values()), reverse=True)
    rank = {v: i for i, v in enumerate(levels)}
    return tuple(rank[node.q[t]] for t in node.legal)


def stage(node):
    """Root, the second own decision (one to four plays later), or the forced-or-not third."""
    return "root" if node.ply == ROOT_PLY else "second" if node.ply <= ROOT_PLY + 4 else "third"


def analyse(study: Path, *, eligible_only=True):
    roots = []
    for p in sorted((study / "roots").iterdir()):
        if not (p / "teacher.json").exists():
            continue
        if eligible_only:
            complete = json.loads((p / "complete.json").read_text())
            if not complete.get("eligible"):
                continue
        roots.append(p)
    course = {}
    per_root = {}
    for directory in roots:
        nodes = load_root(directory)
        audit(nodes)
        rid = next(iter(nodes.values())).root_id
        course.update(nodes)
        per_root[rid] = nodes
    report = {"study": str(study), "eligible_only": eligible_only, "roots": len(per_root),
              "splits": dict(Counter(next(iter(n.values())).split for n in per_root.values())),
              "nodes": len(course),
              "nodes_by_ply": dict(sorted(Counter(n.ply for n in course.values()).items())),
              "nodes_by_stage": dict(Counter(stage(n) for n in course.values())),
              "menu_sizes": dict(sorted(Counter(len(n.legal) for n in course.values()).items())),
              "interfaces": {}}
    disc = [n for n in course.values() if len(n.legal) > 1 and not n.settled]
    report["discretionary_unsettled"] = {
        "nodes": len(disc), "nonflat": sum(1 for n in disc if len(n.best) < len(n.legal)),
        "by_split": {s: {"nodes": sum(1 for n in disc if n.split == s),
                         "nonflat": sum(1 for n in disc if n.split == s and len(n.best) < len(n.legal))}
                     for s in sorted({n.split for n in disc})},
        "by_stage": {st: {"nodes": sum(1 for n in disc if stage(n) == st),
                          "nonflat": sum(1 for n in disc if stage(n) == st and len(n.best) < len(n.legal))}
                     for st in ("root", "second", "third")}}
    # Decision classes (weak orders) at discretionary unsettled nodes, by menu size.
    classes = {}
    for size in sorted({len(n.legal) for n in disc}):
        group = [n for n in disc if len(n.legal) == size]
        orders = Counter(weak_order(n) for n in group)
        types = Counter(tuple(sorted(Counter(weak_order(n)).values(), reverse=True)) for n in group)
        classes[size] = {"nodes": len(group), "distinct_weak_orders": len(orders),
                         "weak_orders": {"".join(map(str, k)): v for k, v in sorted(orders.items())},
                         "tier_types": {"/".join(map(str, k)): v for k, v in sorted(types.items())}}
    report["decision_classes"] = classes
    for name, (out, succ) in INTERFACES.items():
        sig = signatures(course, out, succ)
        sig_cls = intern({k: sig[k] for k in sorted(course)})
        rounds = refine(course, out, succ)
        ref_cls = rounds[-1]
        if not same_partition(sig_cls, ref_cls):
            raise ValueError(f"{name}: signature and refinement partitions differ")
        ladder = [len(set(r.values())) for r in rounds]
        root_keys = [k for k, n in course.items() if stage(n) == "root"]
        root_ladder = [len({r[k] for k in root_keys}) for r in rounds]
        per = {rid: len({sig_cls[k] for k in nodes}) for rid, nodes in per_root.items()}
        ratios = sorted(Fraction(len(nodes), per[rid]) for rid, nodes in per_root.items())
        roots_of = defaultdict(set)
        for k, n in course.items():
            roots_of[sig_cls[k]].add(n.root_id)
        train_classes = {sig_cls[k] for k, n in course.items() if n.split == "train"}
        coverage = {}
        for st in ("root", "second", "third"):
            keys = [k for k, n in course.items() if n.split == "test" and stage(n) == st]
            dkeys = [k for k in keys if len(course[k].legal) > 1 and not course[k].settled]
            coverage[st] = {"all": {"seen": sum(1 for k in keys if sig_cls[k] in train_classes), "of": len(keys)},
                            "discretionary_unsettled": {"seen": sum(1 for k in dkeys if sig_cls[k] in train_classes),
                                                        "of": len(dkeys)}}
        states_by_stage = {st: len({sig_cls[k] for k, n in course.items() if stage(n) == st})
                           for st in ("root", "second", "third")}
        report["interfaces"][name] = {
            "states": len(set(sig_cls.values())), "halt_and_reject": 2,
            "ladder_by_block_depth": ladder,
            "root_ladder_by_block_depth": root_ladder,
            "states_by_stage": states_by_stage,
            "per_root_states": {"min": min(per.values()), "max": max(per.values()), "sum": sum(per.values())},
            "per_root_compression_nodes_per_state": {"min": str(ratios[0]), "median": str(ratios[len(ratios) // 2]),
                                                     "max": str(ratios[-1])},
            "classes_in_more_than_one_root": sum(1 for s in roots_of.values() if len(s) > 1),
            "test_coverage_by_train_classes": coverage,
        }
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("study", type=Path, help="a contingent-teacher study root holding roots/*/teacher.json")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--all-solved", action="store_true", help="include solved roots the study rejected as ineligible")
    args = ap.parse_args()
    report = analyse(args.study, eligible_only=not args.all_solved)
    text = json.dumps(report, indent=1)
    if args.out:
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
