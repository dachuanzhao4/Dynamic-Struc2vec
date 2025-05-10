
import argparse
import math
import os
from collections import defaultdict
from typing import Dict, Tuple, List, Union

import numpy as np
from scipy.stats import pearsonr, spearmanr

from utils import restoreVariableFromDisk, load_layer_weights

Pair = Tuple[int, int]
Vec  = np.ndarray


def read_embedding(path: str, id_type: str = "str") -> Dict[Union[int, str], Vec]:
    embs: Dict[Union[int, str], Vec] = {}
    with open(path, "r", encoding="utf-8") as fh:
        header = fh.readline()
        if len(header.split()) != 2:
            fh.seek(0)
        for line in fh:
            toks = line.strip().split()
            if len(toks) < 2:
                continue
            raw_id, comps = toks[0], toks[1:]
            vec = np.asarray([float(x) for x in comps], dtype=np.float32)
            node_id = int(raw_id) if id_type == "int" else raw_id
            embs[node_id] = vec
    return embs


def build_distance_dict(
    prefix: str,
    layer_choice: Union[int, None],
    combine: str = "min",
) -> Dict[Pair, float]:
    path = f"pickles/{prefix}/distances_nets_graphs.pickle"
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    layer_adj: Dict[int, Dict[int, List[int]]] = restoreVariableFromDisk(
        f"{prefix}/distances_nets_graphs"
    )

    pair2layers: Dict[Pair, List[float]] = defaultdict(list)
    layers = [layer_choice] if layer_choice is not None else sorted(layer_adj.keys())

    for L in layers:
        if L not in layer_adj:
            continue
        w_table = load_layer_weights(prefix, L)
        nbrs = layer_adj[L]
        for u, vs in nbrs.items():
            w_list = w_table[u]
            for idx, v in enumerate(vs):
                w = w_list[idx]
                if w <= 0.0 or math.isnan(w):
                    continue
                a, b = (u, v) if u < v else (v, u)
                d = -math.log(w)
                pair2layers[(a, b)].append(d)

    pair2dist: Dict[Pair, float] = {}
    for pair, dlist in pair2layers.items():
        if not dlist:
            continue
        if layer_choice is not None:
            pair2dist[pair] = dlist[0]
        else:
            if combine == "min":
                pair2dist[pair] = min(dlist)
            elif combine == "mean":
                pair2dist[pair] = sum(dlist) / len(dlist)
            else:
                raise ValueError(f"Unknown combine: {combine}")
    return pair2dist


def to_sample_list(dist_dict: Dict[Pair, float]) -> List[Tuple[int, int, float]]:
    return [(u, v, d) for (u, v), d in dist_dict.items()]


def compute_and_print(dist_prefix, emb_file, idt, layer, combine):
    dist_dict = build_distance_dict(dist_prefix, layer, combine)
    embs = read_embedding(emb_file, idt)
    samples = to_sample_list(dist_dict)
    struct_d, emb_d = [], []
    for u, v, d in samples:
        uid = int(u) if idt == "int" else u
        vid = int(v) if idt == "int" else v
        if uid in embs and vid in embs:
            struct_d.append(d)
            emb_d.append(np.linalg.norm(embs[uid] - embs[vid]))
    if len(struct_d) < 2:
        print(f"[WARN] Not enough pairs for layer={layer}, combine={combine}")
        return

    pr, pp = pearsonr(struct_d, emb_d)
    sr, sp = spearmanr(struct_d, emb_d)
    tag = f"layer={layer}" if layer is not None else f"combine={combine}"
    print(f"[RESULT] {tag} Pearson r={pr:.4f}  Spearman ρ={sr:.4f}")


def main():
    ap = argparse.ArgumentParser("Correlation test between structure & embedding.")
    ap.add_argument("--dist-prefix", required=True,
                    help="Sub-folder under pickles/")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--layer", type=int, default=None,
                     help="Test exactly this layer")
    grp.add_argument("--combine", choices=["min", "mean"], default=None,
                     help="Aggregate across all layers")
    ap.add_argument("--embedding-file", required=True)
    ap.add_argument("--id-type", choices=["int","str"], default="str")
    args = ap.parse_args()

    graphs = restoreVariableFromDisk(f"{args.dist_prefix}/distances_nets_graphs")
    layers = sorted(graphs.keys())

    if args.layer is not None:
        compute_and_print(args.dist_prefix, args.embedding_file,
                          args.id_type, args.layer, None)
    elif args.combine is not None:
        compute_and_print(args.dist_prefix, args.embedding_file,
                          args.id_type, None, args.combine)
    else:
        for L in layers:
            compute_and_print(args.dist_prefix, args.embedding_file,
                              args.id_type, L, None)

    if args.layer is None and args.combine is None:
        pearsons, spearmans = [], []
        for L in layers:
            if L == 0:
                continue

            dist_dict = build_distance_dict(args.dist_prefix, L, 'min')
            embs = read_embedding(args.embedding_file, args.id_type)
            samples = to_sample_list(dist_dict)
            struct_d, emb_d = [], []
            for u, v, d in samples:
                uid = int(u) if args.id_type == "int" else u
                vid = int(v) if args.id_type == "int" else v
                if uid in embs and vid in embs:
                    struct_d.append(d)
                    emb_d.append(np.linalg.norm(embs[uid] - embs[vid]))
            if len(struct_d) >= 2:
                pr, _ = pearsonr(struct_d, emb_d)
                sr, _ = spearmanr(struct_d, emb_d)
                pearsons.append(pr)
                spearmans.append(sr)
        if pearsons:
            avg_pr = sum(pearsons) / len(pearsons)
            avg_sr = sum(spearmans) / len(spearmans)
            print(f"[SUMMARY] layers=1–{layers[-1]}  "
                  f"avg Pearson r={avg_pr:.4f}  avg Spearman ρ={avg_sr:.4f}")



if __name__ == "__main__":
    main()
