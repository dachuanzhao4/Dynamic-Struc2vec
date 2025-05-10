
import argparse
import logging
import pathlib
import shutil
import sys
import time as time_module
import yaml
import shlex
import glob
import os

import networkx as nx
import numpy as np
from gensim.models import Word2Vec

from get_delta import compute_deltas
from local_walk_sampling import generate_local_walks
from joint_optimize import incremental_train
from struc2vec import Graph as S2VGraph, write_walks_to_disk
from graph import from_networkx

from algorithms_distances import (
    generate_distances_network_part1,
    generate_distances_network_part2,
    splitDegreeList,
    calc_distances,
    generate_distances_network_part3,
    generate_distances_network_part4,
    generate_distances_network_part5,
    generate_distances_network_part6,
)

logger = logging.getLogger("sd_loop")
logging.basicConfig(
    format="[%(levelname)s] %(asctime)s %(name)s: %(message)s",
    level=logging.INFO
)


def _load_snapshot_pickles(snapshot_dir: pathlib.Path, pickles_dir: pathlib.Path):
    if not snapshot_dir.exists():
        logger.error("Snapshot pickles dir %s not found", snapshot_dir)
        sys.exit(1)
    for p in glob.glob(str(snapshot_dir / "*.pickle")):
        dest = pickles_dir / os.path.basename(p)
        shutil.copy2(p, dest)
    logger.info("Loaded all pickles from %s into %s", snapshot_dir, pickles_dir)


def run_increment(conf_path: pathlib.Path, step_id: str, use_warm: bool):
    start_time = time_module.time()
    cmd = " ".join(shlex.quote(x) for x in sys.argv)
    with open(conf_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    sequence = {snap["id"]: snap["edgelist"] for snap in cfg["sequence"]}
    ids = [snap["id"] for snap in cfg["sequence"]]
    if step_id not in sequence:
        logger.error("step_id %s not found", step_id)
        sys.exit(1)
    idx = ids.index(step_id)
    if idx == 0:
        logger.error("step_id %s is the first snapshot", step_id)
        sys.exit(1)
    prev_id = ids[idx - 1]
    prev_dir = f"{prev_id}_warm" if use_warm else prev_id
    g_prev_nx = nx.read_edgelist(sequence[prev_id], nodetype=int, data=(("w", float),), create_using=nx.Graph())
    g_curr_nx = nx.read_edgelist(sequence[step_id], nodetype=int, data=(("w", float),), create_using=nx.Graph())
    logger.info("Loaded |V_prev|=%d, |V_curr|=%d",
                g_prev_nx.number_of_nodes(), g_curr_nx.number_of_nodes())
    until_layer = cfg["params"].get("until_layer", None)
    delta_scores = compute_deltas(g_prev_nx, g_curr_nx, L=until_layer)
    topk = cfg["params"].get("topk_modify", 20)
    from modify_map import partition_and_remap
    info = partition_and_remap(
        set(g_prev_nx.nodes()),
        set(g_curr_nx.nodes()),
        delta_scores,
        topk,
        do_remap=False,       
    )
    logger.info("Partitioned: frozen=%d, modifiable=%d, new=%d",
                len(info.frozen_old), len(info.modifiable_old), len(info.new_nodes))

    pickles_dir = pathlib.Path("pickles")
    prev_snap = pickles_dir / prev_dir
    warm_snap = pickles_dir / f"{step_id}_warm"
    if warm_snap.exists():
        shutil.rmtree(warm_snap)
    shutil.copytree(prev_snap, warm_snap)
    logger.info("Copied pickles %s → %s", prev_snap, warm_snap)

    custom_G = from_networkx(g_curr_nx, undirected=True)
    workers = cfg["params"].get("workers", 1)
    s2v_graph = S2VGraph(custom_G, is_directed=False,
                         workers=workers, untilLayer=until_layer)

    _load_snapshot_pickles(prev_snap, pickles_dir)

    seeds = info.modifiable_old + info.new_nodes
    prev_pickles = pickles_dir / prev_dir

    if (prev_pickles / "compactDegreeList.pickle").exists():
        compactDegree = True
    elif (prev_pickles / "degreeList.pickle").exists():
        compactDegree = False
    else:
        raise RuntimeError(f"No degreeList or compactDegreeList found in {prev_pickles}")

    if until_layer is None:
        logger.error("until_layer must be set for incremental update")
        sys.exit(1)

    for layer in range(1, until_layer + 1):
        logger.info("Incremental recompute for layer %d …", layer)
        splitDegreeList(
            part=1,
            seeds=seeds,
            G=s2v_graph.G,
            compactDegree=compactDegree,
            prefix=prev_dir
        )
        calc_distances(1, compactDegree=compactDegree, prefix=prev_dir)

    generate_distances_network_part1(workers, prefix=prev_dir, seeds=seeds)
    generate_distances_network_part2(workers, prefix=prev_dir, seeds=seeds)
    generate_distances_network_part3()
    generate_distances_network_part4()
    generate_distances_network_part5()
    generate_distances_network_part6()
    logger.info("Incremental structural distance network updated for %d nodes", len(seeds))

    s2v_graph.load_walk_graph(prefix="")

    s2v_graph.preprocess_parameters_random_walk()
    logger.info("Recomputed random-walk parameters (amount_neighbours, average_weight).")

    num_walks = cfg["params"].get("num_walks", 20)
    walk_length = cfg["params"].get("walk_length", 80)
    walks = generate_local_walks(s2v_graph, seeds, num_walks, walk_length)
    write_walks_to_disk(walks, warm_snap / "random_walks.txt")
    logger.info("Generated %d local walks", len(walks))

    emb_dir = pathlib.Path("emb")
    prev_model = emb_dir / f"{prev_dir}.model"
    prev_emb   = emb_dir / f"{prev_dir}.emb"
    if prev_model.exists():
        model = Word2Vec.load(str(prev_model))
        model.wv.vectors_lockf = np.ones(len(model.wv.key_to_index), dtype=np.float32)
    elif prev_emb.exists():
        from gensim.models import KeyedVectors
        kv = KeyedVectors.load_word2vec_format(str(prev_emb))
        model = Word2Vec(vector_size=kv.vector_size, min_count=0)
        model.build_vocab_from_freq({k:1 for k in kv.key_to_index}, update=False)
        model.wv = kv
        model.wv.vectors_lockf = np.ones(len(kv.key_to_index), dtype=np.float32)
    else:
        logger.error("No previous embedding found for %s", prev_id)
        sys.exit(1)

    sentences = [[str(n) for n in w] for w in walks]
    model = incremental_train(
        prev_model_path = prev_model,
        sentences       = sentences,
        frozen_nodes    = info.frozen_old,
        iter_inc        = cfg["params"].get("iter_inc", 2),
        alpha           = cfg["params"].get("alpha_inc", 0.005),
        lambda_reg      = cfg["params"].get("lambda_reg", 0.0),
        save_model_path = emb_dir / f"{step_id}_warm.model",
        save_emb_path   = emb_dir / f"{step_id}_warm.emb",
    )

    for root_pickle in glob.glob(str(pickles_dir / '*.pickle')):
        shutil.move(root_pickle, warm_snap / os.path.basename(root_pickle))
    logger.info("Moved all intermediate pickles into %s", warm_snap)

    elapsed = time_module.time() - start_time
    (pathlib.Path("time") / f"{step_id}_warm").write_text(f"{cmd}\n{elapsed:.3f}\n")
    logger.info("Incremental %s done in %.3f s", step_id, elapsed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run struc2vec incremental warm-start")
    parser.add_argument("--conf", required=True, type=pathlib.Path, help="YAML timeline")
    parser.add_argument("--step", required=True, help="Snapshot ID to process")
    parser.add_argument("--use-warm", action="store_true",
                        help="If set, load pickles and embedding from <prev_id>_warm instead of <prev_id>")
    args = parser.parse_args()
    run_increment(args.conf, args.step, args.use_warm)