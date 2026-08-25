"""A swarm particle: a candidate subgraph anchored at a pivot node.

A particle carries a pattern (structure and labels) plus a designated pivot node that
serves as the structural reference for guided expansion. The pivot is search metadata,
not part of the pattern's structural identity, so the canonical signature depends on the
pattern alone — two particles with the same structure but different pivots share a
signature.
"""

from __future__ import annotations

from dataclasses import dataclass

from .graph import Graph
from .signature import canonical_signature


@dataclass
class Particle:
    pattern: Graph
    pivot: int

    def __post_init__(self) -> None:
        if self.pivot not in self.pattern.node_labels:
            raise ValueError(f"pivot {self.pivot} is not a node of the pattern")

    def signature(self) -> str:
        return canonical_signature(self.pattern)
