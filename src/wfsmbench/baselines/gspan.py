"""gSpan as published, for use as a baseline.

Yan and Han, *gSpan: Graph-Based Substructure Pattern Mining*, ICDM 2002. The definitions are
transcribed in `docs/GSPAN_SOURCE.md`, read from the paper's own pages, and the code below is
written against that transcription rather than against a recollection of how gSpan works.

Two things the paper states without proof are implemented as specified and checked by
measurement instead of by citation: that the minimum DFS code is a canonical label -- the
paper's isomorphism theorem, stated proof omitted -- and that a non-minimal code's subtree
contains no minimum code, which licenses the pruning and is proved only in a technical
report this project does not hold. GSPAN_SOURCE.md maps each object here to the numbered
statement it comes from; the numbers stay there so that nothing in the code depends on
them. The check is that the
mined set equals what the exhaustive enumerator finds -- an enumerator that uses no DFS codes
at all, so an error in either would show as a difference.

Deliberate departures, each recorded in GSPAN_SOURCE.md: the frequency relabelling that opens the
paper's outer procedure is omitted, and a maximum pattern size is added because every
comparison in this project is made at one.
"""

from __future__ import annotations

from collections.abc import Iterable

from ..core.graph import Graph

# A DFS code entry is (i, j, l_i, l_edge, l_j) with i, j the DFS ids of the endpoints.
Entry = tuple[int, int, int, int, int]
Code = tuple[Entry, ...]


def _is_forward(entry: Entry) -> bool:
    return entry[0] < entry[1]


def _entry_lt(a: Entry, b: Entry) -> bool:
    """The order on single edges: `<_T` first, then labels lexicographically.

    The paper gives `<_T` by three generating rules on one DFS tree and then defines `<_e` as
    "the lexicographic combination of `<_T` and `<_L`". Spelled out for the four cases that
    arise when comparing edges from *different* trees:
    """
    ai, aj, bi, bj = a[0], a[1], b[0], b[1]
    a_fwd, b_fwd = _is_forward(a), _is_forward(b)

    if a_fwd and b_fwd:
        if aj != bj:
            return aj < bj          # the edge reaching the earlier new vertex comes first
        if ai != bi:
            return ai > bi          # from deeper on the rightmost path comes first
    elif not a_fwd and not b_fwd:
        if ai != bi:
            return ai < bi
        if aj != bj:
            return aj < bj
    elif a_fwd and not b_fwd:
        return aj <= bi             # a forward edge precedes a backward one it reaches past
    else:
        return ai < bj

    return (a[2], a[3], a[4]) < (b[2], b[3], b[4])   # same shape: compare by labels


def _code_lt(left: Code, right: Code) -> bool:
    """The DFS lexicographic order, as an ordinary strict less-than."""
    for a, b in zip(left, right):
        if a == b:
            continue
        return _entry_lt(a, b)
    return len(left) < len(right)


def rightmost_path(code: Code) -> list[int]:
    """DFS ids on the rightmost path, root first.

    The rightmost path is the straight path from `v_0` to the rightmost vertex `v_n`. It is
    read off the forward edges from the end backwards: each one whose target is where the
    previous step started continues the path.
    """
    if not code:
        return []
    path: list[int] = []
    expected: int | None = None
    for entry in reversed(code):
        if not _is_forward(entry):
            continue
        source, target = entry[0], entry[1]
        if expected is None or target == expected:
            path.append(target)
            expected = source
    if expected is not None:
        path.append(expected)
    return list(reversed(path))


def _neighbours(graph: Graph, node: int) -> list[tuple[int, int]]:
    """(neighbour, edge label) pairs, in a deterministic order."""
    out = []
    for (u, v), label in graph.edge_items():
        if u == node:
            out.append((v, label))
        elif v == node:
            out.append((u, label))
    return sorted(out)


def min_dfs_code(graph: Graph) -> Code:
    """The minimum DFS code, which the paper's isomorphism theorem makes a canonical label.

    Built by depth-first growth over every possible starting edge, keeping the smallest
    complete code. A partial code that is already larger than the best complete code's
    prefix cannot become smaller, so it is abandoned -- the codes here are at most a handful
    of edges, so this is fast enough without the incremental construction the paper uses for
    large patterns.
    """
    total = graph.num_edges()
    if total == 0:
        return ()
    best: Code | None = None

    def grow(code: list[Entry], dfs_of: dict[int, int], used: set[frozenset]) -> None:
        nonlocal best
        current = tuple(code)
        if best is not None and len(current) <= len(best) and _code_lt(tuple(best[:len(current)]), current):
            return                                   # this branch can only get larger
        if len(code) == total:
            if best is None or _code_lt(current, best):
                best = current
            return
        graph_of = {d: g for g, d in dfs_of.items()}
        path = rightmost_path(current)
        rightmost = path[-1]
        candidates: list[tuple[Entry, int | None]] = []

        # Backward edges grow only from the rightmost vertex, to a vertex already on the path.
        for target, label in _neighbours(graph, graph_of[rightmost]):
            if target not in dfs_of:
                continue
            if frozenset((graph_of[rightmost], target)) in used:
                continue
            if dfs_of[target] in path[:-1]:
                candidates.append(((rightmost, dfs_of[target], graph.label(graph_of[rightmost]),
                                    label, graph.label(target)), None))

        # Forward edges grow from any vertex on the rightmost path, to a new vertex.
        next_id = max(dfs_of.values()) + 1
        for source in reversed(path):
            for target, label in _neighbours(graph, graph_of[source]):
                if target in dfs_of:
                    continue
                candidates.append(((source, next_id, graph.label(graph_of[source]),
                                    label, graph.label(target)), target))

        for entry, new_node in sorted(candidates, key=_EntryKey):
            edge = frozenset((graph_of[entry[0]] if new_node is None else graph_of[entry[0]],
                              graph_of[entry[1]] if new_node is None else new_node))
            code.append(entry)
            if new_node is not None:
                dfs_of[new_node] = entry[1]
            used.add(edge)
            grow(code, dfs_of, used)
            used.discard(edge)
            if new_node is not None:
                del dfs_of[new_node]
            code.pop()

    for (u, v), label in graph.edge_items():
        for a, b in ((u, v), (v, u)):
            first: Entry = (0, 1, graph.label(a), label, graph.label(b))
            grow([first], {a: 0, b: 1}, {frozenset((a, b))})
    assert best is not None
    return best


class _EntryKey:
    """Sort key wrapping `_entry_lt`, so candidates are tried in code order."""

    __slots__ = ("value",)

    def __init__(self, pair: tuple[Entry, int | None]) -> None:
        self.value = pair[0]

    def __lt__(self, other: "_EntryKey") -> bool:
        return _entry_lt(self.value, other.value)


def code_to_graph(code: Code) -> Graph:
    """The graph a DFS code describes, with DFS ids as vertex ids."""
    labels: dict[int, int] = {}
    edges = []
    for i, j, li, le, lj in code:
        labels.setdefault(i, li)
        labels.setdefault(j, lj)
        edges.append((i, j, le))
    return Graph.of(labels, edges)


class _Occurrence:
    """One embedding of the current pattern in one database graph."""

    __slots__ = ("graph_index", "vertex_of", "edges")

    def __init__(self, graph_index: int, vertex_of: dict[int, int],
                 edges: frozenset) -> None:
        self.graph_index = graph_index
        self.vertex_of = vertex_of
        self.edges = edges


def _extensions(graph: Graph, code: Code, occurrence: _Occurrence):
    """Every rightmost extension of one occurrence, as (entry, new vertex or None).

    "b must be an edge which only grows from the vertices on the rightmost path": backward
    edges from the rightmost vertex only, forward edges from any vertex on the path.
    """
    path = rightmost_path(code)
    rightmost = path[-1]
    taken = set(occurrence.vertex_of.values())
    host_of = occurrence.vertex_of

    for target, label in _neighbours(graph, host_of[rightmost]):
        if target not in taken:
            continue
        key = frozenset((host_of[rightmost], target))
        if key in occurrence.edges:
            continue
        dfs_target = next(d for d, g in host_of.items() if g == target)
        if dfs_target in path[:-1]:
            yield ((rightmost, dfs_target, graph.label(host_of[rightmost]), label,
                    graph.label(target)), None, key)

    next_id = max(host_of) + 1
    for source in reversed(path):
        for target, label in _neighbours(graph, host_of[source]):
            if target in taken:
                continue
            yield ((source, next_id, graph.label(host_of[source]), label,
                    graph.label(target)), target, frozenset((host_of[source], target)))


def gspan(graphs: list[Graph], min_support: int, max_edges: int,
          max_nodes: int | None = None) -> list[tuple[Code, int]]:
    """Frequent connected subgraphs and their support, by the published algorithm.

    Returns (minimum DFS code, support) for every pattern whose support reaches
    ``min_support``, with at most ``max_edges`` edges and ``max_nodes`` vertices.
    """
    if max_nodes is None:
        max_nodes = max_edges + 1
    found: dict[Code, int] = {}

    def support_of(occurrences: Iterable[_Occurrence]) -> int:
        return len({o.graph_index for o in occurrences})

    def mine(code: Code, occurrences: list[_Occurrence]) -> None:
        # The published inner procedure opens by discarding a non-minimal code and its subtree,
        # because that graph has already been reached under its minimum code.
        if code != min_dfs_code(code_to_graph(code)):
            return
        found[code] = support_of(occurrences)
        if len(code) >= max_edges:
            return

        grown: dict[Entry, list[_Occurrence]] = {}
        for occurrence in occurrences:
            graph = graphs[occurrence.graph_index]
            for entry, new_vertex, edge_key in _extensions(graph, code, occurrence):
                if new_vertex is not None and len(occurrence.vertex_of) + 1 > max_nodes:
                    continue
                mapping = dict(occurrence.vertex_of)
                if new_vertex is not None:
                    mapping[entry[1]] = new_vertex
                grown.setdefault(entry, []).append(
                    _Occurrence(occurrence.graph_index, mapping,
                                occurrence.edges | {edge_key}))

        for entry in sorted(grown, key=lambda e: _EntryKey((e, None))):
            child = grown[entry]
            if support_of(child) >= min_support:
                mine(code + (entry,), child)

    # Start from every frequent one-edge pattern, as the outer procedure does. The
    # relabelling by frequency it opens with is omitted -- see GSPAN_SOURCE.md.
    seeds: dict[Entry, list[_Occurrence]] = {}
    for index, graph in enumerate(graphs):
        for (u, v), label in graph.edge_items():
            for a, b in ((u, v), (v, u)):
                entry: Entry = (0, 1, graph.label(a), label, graph.label(b))
                seeds.setdefault(entry, []).append(
                    _Occurrence(index, {0: a, 1: b}, frozenset({frozenset((a, b))})))

    for entry in sorted(seeds, key=lambda e: _EntryKey((e, None))):
        if support_of(seeds[entry]) >= min_support:
            mine((entry,), seeds[entry])
    return sorted(found.items())
