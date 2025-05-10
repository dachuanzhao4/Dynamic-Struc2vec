# local_walk_sampling.py
# -----------------------------------------------------------------------------
# Localised random‑walk generator used during incremental warm‑start.
# -----------------------------------------------------------------------------
# Design goals
#   1) Reuse **exactly the same walk engine** implemented inside
#      sebkaz/struc2vec.Graph.walk(), so the statistical properties of walks
#      (layer hopping probability, alias sampling tables, etc.) are preserved –
#      ensuring result comparability with vanilla struc2vec.
#   2) Limit computational cost by starting walks **only** from the set
#      (modifiable_old ∪ new_nodes).  This is analogous to DNE's
#      enhance_joint_sampling.py, which biases samples towards perturbed/new
#      regions but still lets walks wander in the full graph – keeping rich
#      structural context.
#   3) Provide an optional `allowed_nodes` filter that – if enabled – forces the
#      walk to stay inside a user‑supplied node set.  In practice we leave it
#      *disabled* by default because hard‑clipping can disconnect structural
#      layers and harm convergence; but the switch is here for future tuning.
# -----------------------------------------------------------------------------
# Public API
#   generate_local_walks(s2v_graph: struc2vec.Graph,
#                        target_nodes: List[int],
#                        num_walks: int,
#                        walk_length: int,
#                        stay_local: bool = False,
#                        allowed_radius: int = 1) -> List[List[int]]
# -----------------------------------------------------------------------------

from __future__ import annotations

import random
from typing import Iterable, List, Set

# The Graph class type hint (we import lazily to avoid circular import in tests)
from struc2vec import Graph as S2VGraph
import networkx as nx

# -----------------------------------------------------------------------------

def _build_allowed_set(G_orig: nx.Graph, seed_nodes: Iterable[int], radius: int = 1) -> Set[int]:
    """Return the set of nodes within *radius* (in original graph) from *seed_nodes*."""
    allowed = set(seed_nodes)
    if radius <= 0:
        return allowed
    frontier = set(seed_nodes)
    for _ in range(radius):
        nxt = set()
        for u in frontier:
            nxt.update(G_orig.neighbors(u))
        nxt -= allowed
        if not nxt:
            break
        allowed.update(nxt)
        frontier = nxt
    return allowed

# -----------------------------------------------------------------------------


def generate_local_walks(graph, seeds, num_walks, walk_length, stay_local=False, allowed_radius=None):
    walks = []
    for seed in seeds:
        allowed_nodes = None
        if stay_local and allowed_radius is not None:
            allowed_nodes = {seed}
            queue = [(seed, 0)]
            while queue:
                node, dist = queue.pop(0)
                if dist < allowed_radius:
                    for neighbor in graph.G.get(node, []):
                        if neighbor not in allowed_nodes:
                            allowed_nodes.add(neighbor)
                            queue.append((neighbor, dist+1))
        for i in range(num_walks):
            cur_node = seed
            cur_layer = 0
            walk = [cur_node]
            for step in range(1, walk_length):
                next_node, next_layer = graph._choose_next(cur_node, cur_layer)
                if allowed_nodes is not None and next_node not in allowed_nodes:
                    break
                walk.append(next_node)
                cur_node = next_node
                cur_layer = next_layer
            walks.append(walk)
    return walks

