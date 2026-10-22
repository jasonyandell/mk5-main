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

Run: python -u -m lem.quotient <study> [--out report.json] [--all-solved] [--labels] [--export course.json]

--export writes the audited course as a neutral graph (silas-decision-course-v1) for
Silas's independent implementation, `silas quotient --course course.json`.

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


# ---- tile-free observation labels -----------------------------------------
#
# Each observed play is described by facts that name no tile.  The hand-relative facts say how it
# stands in its trick and against the focal seat's remaining tiles, which are named only by slot
# (rank by tile id among them, as menu slots are).  The posterior facts are statistics of what the
# focal seat can infer about the tiles it has not seen, once the play joins the record: unseen
# trumps and count, per seat the suits it has shown void in and the suits of which it can hold no
# unseen tile, how many unseen tiles outrank the play in its trick, and how many unseen tiles still
# follow the trick's suit.  One block fact, the likelihood of the whole observation to within a
# factor of two, reads the course's own masses.  A label keeps a subset of the facts; the block
# label is the action's menu slot, the sequence of play labels and the kept block facts.  A label
# routes when every (slot, label) out of a state leads to one state.
#
# The `next_*` block facts are not candidates.  They name the next decision's own answer, which
# takes solving it: the ceiling, showing what a label would have to carry to route.

HAND_FIELDS = ("seat", "trump", "follows", "count", "beats_all", "winning", "follow_slots", "beats_slots")
POSTERIOR_FIELDS = ("unseen_trumps", "unseen_count", "voids", "exhausted", "outranked_by", "suit_left")
FIELDS = HAND_FIELDS + POSTERIOR_FIELDS
BLOCK_FIELDS = ("likelihood", "next_output", "next_fork")
POSTERIOR = ("seat",) + POSTERIOR_FIELDS
POSTERIOR_LABELS = {
    "unseen": ("seat", "unseen_trumps", "unseen_count"),
    "voids": ("seat", "voids"),
    "exhausted": ("seat", "exhausted"),
    "outrank": ("seat", "outranked_by", "suit_left"),
    "likelihood": ("seat", "likelihood"),
    "posterior": POSTERIOR,
    "posterior_likelihood": POSTERIOR + ("likelihood",),
    "everything": FIELDS + ("likelihood",),
    "solved_next_output": ("next_output",),
    "solved_next_fork": ("next_fork",),
}
LABELS = {
    "hand_relative": HAND_FIELDS,
    "brief_follow_slots": ("seat", "trump", "follows", "count", "beats_all", "follow_slots"),
    "brief_winning": ("seat", "trump", "follows", "count", "beats_all", "winning"),
    "brief": ("seat", "trump", "follows", "count", "beats_all"),
    "seat_follows": ("seat", "follows"),
    "seat": ("seat",),
}


def replay(tokens):
    """Rules, the focal seat's original tiles, and the public record after the history's plays."""
    from scenario.record import empty_record
    from scenario.rules import rules_for
    rules = rules_for(tokens[3] - 32)
    ps = pairs(tokens)
    record = empty_record(ps[0][0] - YOU)
    for seat, tile in ps:
        if seat - YOU != record.seat:
            raise ValueError("history out of turn")
        record = record.advance(tile, rules)
    return rules, set(tokens[5:12]), record


def play_facts(rules, record, left, tile):
    """The hand-relative facts of one observed play, before it joins the record; `left` holds the focal seat's tiles."""
    led = record.led
    suit = rules.led_suit[tile if led is None else led]
    rank = rules.rank[suit]
    mine = sorted(left)
    top = max((rank[t] for t in record.trick), default=-1)
    return (record.seat, (rules.follow_bits[7] >> tile) & 1,
            2 if led is None else int(rules.follows(tile, led)), rules.count[tile],
            int(all(rank[tile] > rank[u] for u in mine)), int(rank[tile] > top),
            tuple(i for i, u in enumerate(mine) if (rules.follow_bits[suit] >> u) & 1) if led is None else (),
            tuple(i for i, u in enumerate(mine) if rank[tile] > rank[u]))


def posterior_facts(rules, record, own, tile):
    """What the focal seat can infer about the tiles it has not seen once `tile` joins the record;
    `own` is its original hand.  Per-seat facts list the other seats in relative order 1, 2, 3."""
    led = record.led
    suit = rules.led_suit[tile if led is None else led]
    after = record.advance(tile, rules)
    unseen = [t for t in range(28) if not (after.played >> t) & 1 and t not in own]
    held = sum(1 << t for t in unseen)
    suits = [s for s in range(8) if rules.follow_bits[s]]
    return (sum(1 for t in unseen if (rules.follow_bits[7] >> t) & 1), sum(rules.count[t] for t in unseen),
            tuple(sum(1 for s in suits if not rules.follow_bits[s] & ~after.forbid[seat]) for seat in (1, 2, 3)),
            tuple(sum(1 for s in suits if not rules.follow_bits[s] & held & ~after.forbid[seat]) for seat in (1, 2, 3)),
            sum(1 for t in unseen if rules.rank[suit][t] > rules.rank[suit][tile]),
            (rules.follow_bits[suit] & held).bit_count())


def likelihood_bucket(p):
    """The k with 2^-(k+1) < p <= 2^-k: an observation's probability to within a factor of two."""
    k = 0
    while p <= Fraction(1, 2 ** (k + 1)):
        k += 1
    return k


def block_facts(course, fork=None):
    """For every non-root node: its action's slot in the parent menu, the facts of each observed play
    (hand-relative, then posterior) and the block facts.  `fork`, the fork classes, feeds the solved
    ceiling `next_fork`; without it that fact is None."""
    facts = {}
    for key, node in course.items():
        if not node.children:
            continue
        rules, own, history = replay(node.tokens)
        played = {t for s, t in pairs(node.tokens) if s == YOU}
        slots = {t: i for i, t in enumerate(node.legal)}
        for (action, observed), child in node.children.items():
            record, left = history.advance(action, rules), own - played - {action}
            plays = []
            for seat, tile in observed:
                if seat - YOU != record.seat or seat == YOU:
                    raise ValueError("observed play out of turn")
                plays.append(play_facts(rules, record, left, tile) + posterior_facts(rules, record, own, tile))
                record = record.advance(tile, rules)
            nxt = course[child]
            block = (likelihood_bucket(nxt.mass / node.mass), output_shape(nxt), None if fork is None else fork[child])
            facts[child] = (slots[action], tuple(plays), block)
    return facts


def project(entry, fields):
    """A block's label under `fields`: its slot, its play labels and its kept block facts.  A label
    that keeps no play fact does not see how many plays there were either."""
    slot, plays, block = entry
    at = [FIELDS.index(f) for f in fields if f in FIELDS]
    bt = [BLOCK_FIELDS.index(f) for f in fields if f in BLOCK_FIELDS]
    return slot, tuple(tuple(p[i] for i in at) for p in plays) if at else (), tuple(block[i] for i in bt)


def successors_labelled(facts, fields):
    """Per (action slot, play-label sequence), the set of successor classes."""
    def succ(node, nodes, cls):
        per = defaultdict(set)
        for child in node.children.values():
            per[project(facts[child], fields)].add(cls[child])
        return tuple(sorted((k, tuple(sorted(v))) for k, v in per.items()))
    return succ


def routing(course, cls, succ, facts, fields):
    """Where a label fails to route: a (slot, label) out of one node that reaches two states.
    Each colliding group is a *decision* collision when the states it reaches answer differently
    (menu size or maximizing slots), a *domain* collision when they answer alike but accept
    different label sequences afterwards, else *deeper*.  For consecutive colliding blocks, the
    facts outside the label that tell them apart; "nothing" means no field of the family does.
    The alphabet is the number of distinct observation labels (the label less its slot) read out
    of nodes at each stage."""
    groups_by_kind, nodes_by_kind, told_apart, examples = Counter(), defaultdict(set), defaultdict(Counter), []
    failing, alphabet = set(), defaultdict(set)
    for node in course.values():
        for child in node.children.values():
            alphabet[stage(node)].add(project(facts[child], fields)[1:])

    def keys(node):
        return {project(facts[c], fields) for c in node.children.values()}

    for key in sorted(course):
        node = course[key]
        groups = defaultdict(list)
        for block, child in sorted(node.children.items()):
            groups[project(facts[child], fields)].append((block, child))
        for label, group in sorted(groups.items()):
            reps = {}
            for block, child in group:
                reps.setdefault(cls[child], (block, child))
            if len(reps) < 2:
                continue
            reps = sorted(reps.values())
            outs = [output_shape(course[c]) for _, c in reps]
            kind = ("decision: menu size" if len({o[1] for o in outs}) > 1 else
                    "decision: maximizing slots" if len(set(outs)) > 1 else
                    "domain" if len({frozenset(keys(course[c])) for _, c in reps}) > 1 else "deeper")
            groups_by_kind[kind] += 1
            nodes_by_kind[kind.split(":")[0]].add(key)
            failing.add(cls[key])
            for (b1, c1), (b2, c2) in zip(reps, reps[1:]):
                (_, f1, g1), (_, f2, g2) = facts[c1], facts[c2]
                differ = [f for i, f in enumerate(FIELDS) if any(x[i] != y[i] for x, y in zip(f1, f2))]
                differ += [f for f, x, y in zip(BLOCK_FIELDS[:1], g1, g2) if x != y]
                told_apart[kind.split(":")[0]]["+".join(differ) or "nothing"] += 1
            if kind.startswith("decision") and len(examples) < 8:
                examples.append({"root": node.root_id, "ply": node.ply, "kind": kind,
                                 "blocks": [[b[0], [list(o) for o in b[1]]] for b, _ in reps],
                                 "next_menus": [list(course[c].legal) for _, c in reps],
                                 "next_q": [{str(t): str(v) for t, v in course[c].q.items()} for _, c in reps]})
    return {"routes": not failing, "states_that_fail": len(failing), "of_states": len(set(cls.values())),
            "alphabet_by_stage": {st: len(v) for st, v in sorted(alphabet.items())},
            "nodes_that_fail_by_stage": dict(Counter(stage(course[k]) for k in set().union(*nodes_by_kind.values()))),
            "colliding_groups_by_kind": dict(sorted(groups_by_kind.items())),
            "nodes_with_collision_by_kind": {k: len(v) for k, v in sorted(nodes_by_kind.items())},
            "consecutive_colliding_blocks_told_apart_by": {k: dict(v.most_common()) for k, v in sorted(told_apart.items())},
            "examples": examples}


def decision_only(course, cls, facts, fields):
    """The label read by a controller that answers decisions and checks no domain.  It is the subset
    construction: the controller's state is the set of states its label history cannot tell apart,
    and a label sequence one of them never meets constrains nothing.  A node *routes decision-only*
    when, started there, every reachable set answers alike (menu size, maximizing slots).  A train
    root *covers* a fresh root when every label sequence the fresh root accepts, the train root
    accepts too, and the two sets reached answer alike: a controller built for the first is exact
    wherever the second can go.  Dropping domains is not an equivalence, so no state count is given."""
    rep = {}
    for k in sorted(course):
        rep.setdefault(cls[k], k)
    groups = {}
    for c, k in rep.items():
        per = defaultdict(set)
        for child in course[k].children.values():
            per[project(facts[child], fields)].add(cls[child])
        groups[c] = per
    out = {c: output_shape(course[k]) for c, k in rep.items()}

    def step(states):
        per = defaultdict(set)
        for c in states:
            for lab, nxt in groups[c].items():
                per[lab] |= nxt
        return {lab: frozenset(v) for lab, v in per.items()}

    consistent_memo, cover_memo = {}, {}

    def consistent(states):
        if states not in consistent_memo:
            consistent_memo[states] = len({out[c] for c in states}) == 1 and all(
                consistent(v) for v in step(states).values())
        return consistent_memo[states]

    def covers(t, s):
        if (t, s) not in cover_memo:
            st = step(t)
            cover_memo[(t, s)] = len({out[c] for c in t | s}) == 1 and all(
                lab in st and covers(st[lab], v) for lab, v in step(s).items())
        return cover_memo[(t, s)]

    failing = {c for c in rep if not consistent(frozenset([c]))}
    train = sorted({cls[k] for k, n in course.items() if stage(n) == "root" and n.split == "train"})
    fresh = [cls[k] for k, n in course.items() if stage(n) == "root" and n.split == "test"]
    return {"routes": not failing, "states_that_fail": len(failing),
            "nodes_that_fail_by_stage": dict(Counter(stage(n) for k, n in course.items() if cls[k] in failing)),
            "fresh_roots_covered_by_a_train_root": sum(
                1 for s in fresh if any(covers(frozenset([t]), frozenset([s])) for t in train)),
            "of": len(fresh)}


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


def load_course(study: Path, *, eligible_only=True):
    """Every (eligible) root of a study, loaded and audited: the course and its per-root parts."""
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
    return course, per_root


def export(course):
    """The course as a neutral decision graph, schema silas-decision-course-v1, which
    `silas quotient` reads: menus, maximizing sets, q, masses and block labels, no hidden
    state.  Nodes are numbered in sorted key order; q is aligned with the menu."""
    ids = {k: i for i, k in enumerate(sorted(course))}
    nodes = []
    for k in sorted(course):
        n = course[k]
        nodes.append({"group": n.root_id, "split": n.split, "stage": stage(n), "ply": n.ply,
                      "menu": list(n.legal), "best": list(n.best), "q": [str(n.q[a]) for a in n.legal],
                      "mass": str(n.mass), "settled": n.settled,
                      "children": [{"play": a, "seen": [list(p) for p in o], "next": ids[c]}
                                   for (a, o), c in sorted(n.children.items())]})
    return {"schema": "silas-decision-course-v1", "nodes": nodes}


def analyse(study: Path, *, eligible_only=True, labels=False):
    course, per_root = load_course(study, eligible_only=eligible_only)
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
    classes_of = {}
    for name, (out, succ) in INTERFACES.items():
        report["interfaces"][name], classes_of[name] = measure(course, per_root, name, out, succ)
    if labels:
        facts = block_facts(course, classes_of["fork"])
        every = {**LABELS, **POSTERIOR_LABELS}
        report["label_fields"] = {name: list(fields) for name, fields in every.items()}
        report["label_interfaces"] = {}
        for name, fields in every.items():
            succ = successors_labelled(facts, fields)
            entry, cls = measure(course, per_root, name, output_shape, succ)
            entry["routing"] = routing(course, cls, succ, facts, fields)
            entry["decision_only"] = decision_only(course, cls, facts, fields)
            report["label_interfaces"][name] = entry
    return report


def measure(course, per_root, name, out, succ):
    """States, ladders, per-root compression and TEST coverage of one interface; and its classes."""
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
    return {
        "states": len(set(sig_cls.values())), "halt_and_reject": 2,
        "ladder_by_block_depth": ladder,
        "root_ladder_by_block_depth": root_ladder,
        "states_by_stage": states_by_stage,
        "per_root_states": {"min": min(per.values()), "max": max(per.values()), "sum": sum(per.values())},
        "per_root_compression_nodes_per_state": {"min": str(ratios[0]), "median": str(ratios[len(ratios) // 2]),
                                                 "max": str(ratios[-1])},
        "classes_in_more_than_one_root": sum(1 for s in roots_of.values() if len(s) > 1),
        "test_coverage_by_train_classes": coverage,
    }, sig_cls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("study", type=Path, help="a contingent-teacher study root holding roots/*/teacher.json")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--all-solved", action="store_true", help="include solved roots the study rejected as ineligible")
    ap.add_argument("--labels", action="store_true", help="also measure the tile-free observation-label interfaces")
    ap.add_argument("--export", type=Path, help="also write the course graph for `silas quotient`")
    args = ap.parse_args()
    if args.export:
        course, _ = load_course(args.study, eligible_only=not args.all_solved)
        args.export.write_text(json.dumps(export(course)))
    report = analyse(args.study, eligible_only=not args.all_solved, labels=args.labels)
    text = json.dumps(report, indent=1)
    if args.out:
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
