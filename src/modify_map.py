

from __future__ import annotations

from collections import namedtuple
from typing import Dict, List, Set, Tuple

RemapInfo = namedtuple(
    "RemapInfo", [
        "frozen_old", "modifiable_old", "new_nodes", "remap_dict", "inverse_dict"
    ]
)


def partition_and_remap(
    nodes_prev: Set[int],
    nodes_curr: Set[int],
    delta_scores: Dict[int, float],
    topk: int,
    do_remap: bool = True,
) -> RemapInfo:

    new_nodes: List[int] = sorted(nodes_curr - nodes_prev)

    old_nodes: List[int] = sorted(nodes_curr & nodes_prev)
    old_nodes_sorted = sorted(old_nodes, key=lambda n: delta_scores.get(n, 0.0), reverse=True)
    modifiable_old: List[int] = old_nodes_sorted[:topk]
    frozen_old: List[int] = old_nodes_sorted[topk:]

    if not do_remap:
        remap = {n: n for n in nodes_curr}
        inv = remap.copy()
        return RemapInfo(frozen_old, modifiable_old, new_nodes, remap, inv)

    new_order: List[int] = frozen_old + modifiable_old + new_nodes
    remap: Dict[int, int] = {old_id: new_idx for new_idx, old_id in enumerate(new_order)}
    inv: Dict[int, int] = {v: k for k, v in remap.items()}

    return RemapInfo(frozen_old, modifiable_old, new_nodes, remap, inv)

def remap_edgelist(edges: List[Tuple[int, int]], mapping: Dict[int, int]) -> List[Tuple[int, int]]:
    """Return a new edge list with both endpoints mapped through *mapping*."""
    return [(mapping[u], mapping[v]) for u, v in edges if u in mapping and v in mapping]

if __name__ == "__main__":
    import argparse, random

    parser = argparse.ArgumentParser("Partition nodes and build remapping table")
    parser.add_argument("--prev", type=int, default=10, help="|V_prev|")
    parser.add_argument("--curr", type=int, default=15, help="|V_curr|")
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()

    prev_nodes = set(range(args.prev))
    curr_nodes = set(range(args.curr))
    Δ = {n: random.random() for n in curr_nodes}
    info = partition_and_remap(prev_nodes, curr_nodes, Δ, args.topk, do_remap=True)
    print("frozen_old   =", info.frozen_old)
    print("modifiable_old=", info.modifiable_old)
    print("new_nodes    =", info.new_nodes)
    print("remap sample =", {k: info.remap_dict[k] for k in sorted(info.remap_dict)[:5]})
