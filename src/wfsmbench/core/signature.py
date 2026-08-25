"""Canonical signature for small labeled patterns.

Two patterns that are isomorphic as node- and edge-labeled undirected graphs receive
the same signature; non-isomorphic ones receive different signatures. The signature is
the lexicographically smallest edge-list encoding taken over every node ordering, so it
is an exact canonical form (not a hash) — the property the duplicate-resistant search
archive relies on. Patterns are bounded in size, so the brute force over orderings is
cheap.
"""

from __future__ import annotations

from itertools import permutations

from .graph import Graph


def canonical_signature(pattern: Graph) -> str:
    nodes = sorted(pattern.node_labels)
    n = len(nodes)
    best = None
    for perm in permutations(range(n)):
        # perm[i] is the fresh id given to the i-th original node.
        position = {nodes[i]: perm[i] for i in range(n)}
        labels: list[int] = [0] * n
        for i in range(n):
            labels[perm[i]] = pattern.node_labels[nodes[i]]
        edges = []
        for (u, v), edge_label in pattern.edge_items():
            a, b = position[u], position[v]
            if a > b:
                a, b = b, a
            edges.append((a, b, edge_label))
        edges.sort()
        candidate = (tuple(labels), tuple(edges))
        if best is None or candidate < best:
            best = candidate
    return _encode(best)


def _encode(form) -> str:
    labels, edges = form
    label_part = ",".join(str(x) for x in labels)
    edge_part = ";".join(f"{a}-{b}:{el}" for (a, b, el) in edges)
    return f"V[{label_part}]E[{edge_part}]"
