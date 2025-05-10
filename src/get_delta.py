

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List

import math
import networkx as nx
import numpy as np


def _log_bin(degree: int, base: int = 2) -> int:
    if degree <= 1:
        return degree  
    return int(math.floor(math.log(degree, base))) + 1


def _degree_histogram_L(
    G,          
    node: int,
    L: int,
    base: int
) -> List[Counter]:
    hists: List[Counter] = [Counter() for _ in range(L + 1)]
    frontier = {node}
    visited  = {node}

    for l in range(1, L + 1):
        next_frontier = set()
        for u in frontier:
            nbrs = G.neighbors(u) if hasattr(G, "neighbors") else G[u]
            for v in nbrs:
                if v not in visited:
                    next_frontier.add(v)
                    visited.add(v)

        if not next_frontier:
            break  
        for v in next_frontier:
            deg = G.degree(v) if hasattr(G, "degree") else len(G[v])
            if deg <= 0:
                continue
            bin_id = _log_bin(deg, base)
            hists[l][bin_id] += 1

        frontier = next_frontier

    return hists


def _hist_l1(h1: Counter, h2: Counter) -> int:
    keys = set(h1) | set(h2)
    return sum(abs(h1[k] - h2[k]) for k in keys)


def compute_deltas(
    G_prev: nx.Graph,
    G_curr: nx.Graph,
    L: int = 2,
    degree_bin_base: int = 2,
) -> Dict[int, float]:
    deltas: Dict[int, float] = {}
    hists_curr: Dict[int, List[Counter]] = {}
    for n in G_curr.nodes():
        hists_curr[n] = _degree_histogram_L(G_curr, n, L, degree_bin_base)
    hists_prev: Dict[int, List[Counter]] = {}

    for n in G_curr.nodes():
        if n not in G_prev:  
            deltas[n] = float("inf")
            continue

        if n not in hists_prev:
            hists_prev[n] = _degree_histogram_L(G_prev, n, L, degree_bin_base)

        delta = 0.0
        for l in range(1, L + 1):
            delta += _hist_l1(hists_prev[n][l], hists_curr[n][l])
        deltas[n] = delta

    return deltas
if __name__ == "__main__":
    import argparse
    import pathlib

    parser = argparse.ArgumentParser("Compute Δ_i between two graph snapshots.")
    parser.add_argument("--g_prev", required=True, help="Edge list of snapshot t‑1")
    parser.add_argument("--g_curr", required=True, help="Edge list of snapshot t")
    parser.add_argument("-L", type=int, default=2, help="Max BFS layer (<= until‑layer)")
    args = parser.parse_args()

    def load_graph(path: str) -> nx.Graph:
        G = nx.read_edgelist(path, nodetype=int, data=(("w", float),), create_using=nx.Graph())
        return G

    Gp = load_graph(args.g_prev)
    Gc = load_graph(args.g_curr)

    Δ = compute_deltas(Gp, Gc, L=args.L)
    print(f"Computed Δ for {len(Δ)} nodes.  Top‑10 by Δ:")
    for n, d in sorted(Δ.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(n, d)
