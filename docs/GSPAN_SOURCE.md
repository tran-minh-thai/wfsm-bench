# gSpan as published — definitions transcribed before any code was written

Source: Xifeng Yan and Jiawei Han, *gSpan: Graph-Based Substructure Pattern Mining*, ICDM
2002. Read from the PDF pages themselves, not from a text extraction: this paper's formulas
are set in embedded math fonts that `pdftotext` renders as noise, so a transcription taken
from that output would have been a guess.

This file exists because of a specific earlier mistake in this project: a baseline variant
was named `tkus-published` and annotated with definition numbers from a paper nobody had
opened, and reading the paper afterwards showed it implemented no published algorithm at
all. So the rule here is that the definitions are copied first and the code is written
against them, and the file records what is **omitted** as loudly as what is implemented.

## Definitions, verbatim

**Edge order.** Given `G_T`, forward edges are the DFS tree edges, backward edges are the
rest. `(i, j)` is an ordered pair; if `i < j` it is a forward edge, otherwise a backward
edge. A linear order `≺_T` is built with `e_1 = (i_1, j_1)`, `e_2 = (i_2, j_2)`:

> (i) if `i_1 = i_2` and `j_1 < j_2`, `e_1 ≺_T e_2`;
> (ii) if `i_1 < j_1` and `j_1 = i_2`, `e_1 ≺_T e_2`;
> (iii) if `e_1 ≺_T e_2` and `e_2 ≺_T e_3`, `e_1 ≺_T e_3`.

**Definition 1 (DFS Code).** "Given a DFS tree `T` for a graph `G`, an edge sequence `(e_i)`
can be constructed based on `≺_T`, such that `e_i ≺_T e_{i+1}`, where `i = 0, ..., |E|-1`.
`(e_i)` is called a DFS code, denoted as `code(G, T)`."

An edge is the 5-tuple `(i, j, l_i, l_{(i,j)}, l_j)`, with `l_i`, `l_j` the labels of `v_i`
and `v_j` and `l_{(i,j)}` the edge label.

**Definition 2 (DFS Lexicographic Order).** With `α = code(G_α, T_α) = (a_0, ..., a_m)` and
`β = code(G_β, T_β) = (b_0, ..., b_n)`, `α ≤ β` iff either:

> (i) `∃t, 0 ≤ t ≤ min(m, n)`, `a_k = b_k` for `k < t`, `a_t ≺_e b_t`
> (ii) `a_k = b_k` for `0 ≤ k ≤ m`, and `n ≥ m`.

`≺_e` is the lexicographic combination of `≺_T` with the label order `≺_L`.

**Definition 3 (Minimum DFS Code).** `Z(G) = {code(G, T) | T is a DFS tree of G}`; the
minimum under DFS lexicographic order, `min(Z(G))`, is the **minimum DFS code** of `G`, and
is a canonical label.

**Theorem 1.** "Given two graphs `G` and `G'`, `G` is isomorphic to `G'` if and only if
`min(G) = min(G')`." The paper states *(proof omitted)*.

**Rightmost extension.** "To construct a valid DFS code `β`, `b` must be an edge which only
grows from the vertices on the rightmost path." Backward edges grow **only from the rightmost
vertex**; forward edges grow from **any vertex on the rightmost path**. The rightmost path is
the straight path from `v_0` to `v_n`, the rightmost vertex.

## The algorithm, verbatim

**Algorithm 1 `GraphSet_Projection(D, S)`**

```
1: sort the labels in D by their frequency;
2: remove infrequent vertices and edges;
3: relabel the remaining vertices and edges;
4: S1 <- all frequent 1-edge graphs in D;
5: sort S1 in DFS lexicographic order;
6: S <- S1;
7: for each edge e in S1 do
8:     initialize s with e, set s.D by graphs which contains e;
9:     Subgraph_Mining(D, S, s);
10:    D <- D - e;
11:    if |D| < minSup;
12:        break;
```

**Subprocedure 1 `Subgraph_Mining(D, S, s)`**

```
1: if s != min(s)
2:     return;
3: S <- S union {s};
4: enumerate s in each graph in D and count its children;
5: for each c, c is s' child do
6:     if support(c) >= minSup
7:         s <- c;
8:         Subgraph_Mining(D_s, S, s);
```

## What the paper does NOT contain, and what that means here

**The correctness proof of the pruning is not in this paper.** Line 1-2 of the subprocedure
discards every non-minimal code, and the justification given is a forward reference: "if in
Fig. 3 the darken nodes contain the same graph but different DFS codes, then `s'` is not the
minimum code (proved in [7]). Therefore, the whole sub-branch of `s'` can be pruned since it
will not contain any minimum DFS code." Reference [7] is the UIUC technical report
`UIUCDCS-R-2002-2296`, which this library does not hold -- see `CAN_TAI.md`.

So the pruning is implemented **as specified** and its completeness is verified here by
measurement rather than by citation: the mined set is compared against the exhaustive
enumeration this project already has, which is independent of DFS codes entirely. If the
pruning were unsound the two sets would differ.

**Theorem 1 is also stated with the proof omitted**, so the same applies to canonicity.

## Deliberate differences from the paper, each with its reason

- **Lines 1-3 of Algorithm 1 (relabelling by frequency) are not implemented.** They are a
  constant-factor optimisation that changes the label order and therefore which code is
  minimal; they do not change the mined set. Implementing them would make the produced codes
  incomparable with the ones in the paper's examples without buying anything measurable here.
- **A size bound is added.** Every comparison in this project is at a fixed maximum pattern
  size, and an unbounded gSpan would be answering a different question.
- **Multigraphs.** This project's collections carry parallel edges. gSpan as published assumes
  a simple labelled graph. Where a pair of vertices carries several edges, they differ by edge
  label, so the 5-tuple still distinguishes them; this is recorded as an assumption to check
  rather than a claim that the published algorithm covers the case.
