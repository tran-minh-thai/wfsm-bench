"""Subgraph matching via NetworkX VF2 (proposal 4.6).

Matching uses subgraph **monomorphism**, not node-induced isomorphism: a pattern edge
must map to a host edge (with a matching edge label), but the host may carry extra edges
among the matched nodes. This is the correct semantics for subgraph mining and is
essential for the dense ``_MD`` datasets, whose host graphs are complete.

Only node labels and edge labels constrain the match — edge weights are read from the
host afterwards (decision: weights score, they do not gate matching). Each embedding is a
distinct occurrence: mappings that differ only by a pattern automorphism (same matched
host nodes and edges) are collapsed to one.
"""

from __future__ import annotations

from networkx.algorithms.isomorphism import GraphMatcher

from ..core.graph import Graph


def _node_match(host_attr: dict, pattern_attr: dict) -> bool:
    return host_attr.get("label") == pattern_attr.get("label")


def _edge_match(host_attr: dict, pattern_attr: dict) -> bool:
    return host_attr.get("label") == pattern_attr.get("label")


def embedding_weights(pattern: Graph, host: Graph) -> list[list[float]]:
    """The matched edge weights of every distinct embedding of ``pattern`` in ``host``.

    One list per embedding, holding that embedding's individual edge weights rather than
    their total. How those weights become a single W(e) is a scoring decision that depends on
    the weight mode -- a plain sum, a per-edge mean, a discounted sum -- and the matcher has
    no business knowing which is wanted. See measure.evaluation.
    """
    if pattern.num_nodes() > host.num_nodes():
        return []
    matcher = GraphMatcher(host.to_networkx(), pattern.to_networkx(),
                           node_match=_node_match, edge_match=_edge_match)

    seen: set = set()
    embeddings: list[list[float]] = []
    for mapping in matcher.subgraph_monomorphisms_iter():   # {host_node: pattern_node}
        pattern_to_host = {pn: hn for hn, pn in mapping.items()}
        host_edges = []
        matched: list[float] = []
        for (pu, pv), _edge_label in pattern.edge_items():
            hu, hv = pattern_to_host[pu], pattern_to_host[pv]
            host_edges.append((hu, hv) if hu < hv else (hv, hu))
            matched.append(host.weight(hu, hv))
        occurrence = (frozenset(mapping), frozenset(host_edges))
        if occurrence in seen:
            continue
        seen.add(occurrence)
        embeddings.append(matched)
    return embeddings


def match_database(pattern: Graph, hosts: list[Graph]) -> dict[int, list[list[float]]]:
    """Embeddings of ``pattern`` per host graph, as ``embeddings_by_graph`` for Evaluation.

    Only graphs with at least one embedding are included. The key is the host's
    ``graph_id`` (its index when the id is unset).
    """
    result: dict[int, list[list[float]]] = {}
    for index, host in enumerate(hosts):
        gid = host.graph_id if host.graph_id is not None else index
        embeddings = embedding_weights(pattern, host)
        if embeddings:
            result[gid] = embeddings
    return result
