#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse, logging
import numpy as np
import struc2vec
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from time import time
import os
import glob
import shutil
import pathlib
import graph
import shlex
import sys

logging.basicConfig(
    filename='struc2vec.log',
    filemode='w',
    level=logging.DEBUG,
    format='%(asctime)s %(message)s'
)

def parse_args():
    '''
    Parses the struc2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Run struc2vec.")

    parser.add_argument(
        '--input', nargs='?', default='graph/karate.edgelist',
        help='Input graph path'
    )

    parser.add_argument(
        '--output', nargs='?', default='emb/karate.emb',
        help='Embeddings path (.emb). Binary .model will share the same basename.'
    )
    parser.add_argument(
        '--prefix', default=None,
        help=(
            "Sub‐folder under pickles/ in which to stash all .pickle files. "
            "If unset, uses the basename of --output (e.g. 'karate' for emb/karate.emb)."
        )
    )

    parser.add_argument(
        '--dimensions', type=int, default=128,
        help='Number of dimensions. Default is 128.'
    )

    parser.add_argument(
        '--walk-length', type=int, default=80,
        help='Length of walk per source. Default is 80.'
    )

    parser.add_argument(
        '--num-walks', type=int, default=10,
        help='Number of walks per source. Default is 10.'
    )

    parser.add_argument(
        '--window-size', type=int, default=10,
        help='Context size for optimization. Default is 10.'
    )

    parser.add_argument(
        '--until-layer', type=int, default=None,
        help='Calculation until the layer.'
    )

    parser.add_argument(
        '--iter', default=5, type=int,
        help='Number of epochs in SGD'
    )

    parser.add_argument(
        '--workers', type=int, default=4,
        help='Number of parallel workers. Default is 4.'
    )

    parser.add_argument(
        '--weighted', dest='weighted', action='store_true',
        help='Boolean specifying weighted vs. unweighted. Default is unweighted.'
    )
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)

    parser.add_argument(
        '--directed', dest='directed', action='store_true',
        help='Graph is directed. Default is undirected.'
    )
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)

    parser.add_argument(
        '--OPT1', default=False, type=bool,
        help='optimization 1'
    )
    parser.add_argument(
        '--OPT2', default=False, type=bool,
        help='optimization 2'
    )
    parser.add_argument(
        '--OPT3', default=False, type=bool,
        help='optimization 3'
    )

    return parser.parse_args()

def read_graph():
    '''
    Reads the input network.
    '''
    logging.info(" - Loading graph...")
    G = graph.load_edgelist(args.input, undirected=not args.directed)
    logging.info(" - Graph loaded.")
    return G

def learn_embeddings():
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    logging.info("Initializing creation of the representations...")
    walks = LineSentence('random_walks.txt')
    # 1) 训练 Word2Vec 模型
    model = Word2Vec(
        walks,
        vector_size=args.dimensions,
        window=args.window_size,
        min_count=0,
        hs=1,
        sg=1,
        workers=args.workers,
        epochs=args.iter
    )
    # 2) 保存文本格式的嵌入（.emb）
    model.wv.save_word2vec_format(args.output)
    logging.info("Text embeddings saved to %s", args.output)

    # 3) 同名二进制模型（.model），以便增量脚本加载
    base, _ = os.path.splitext(args.output)
    bin_path = base + ".model"
    model.save(bin_path)
    logging.info("Binary model saved to %s", bin_path)

    return model

def exec_struc2vec(args):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''
    until_layer = args.until_layer if args.OPT3 else None

    G_nx = read_graph()
    G = struc2vec.Graph(G_nx, args.directed, args.workers, untilLayer=until_layer)

    if args.OPT1:
        G.preprocess_neighbors_with_bfs_compact()
    else:
        G.preprocess_neighbors_with_bfs()

    if args.OPT2:
        G.create_vectors()
        G.calc_distances(compactDegree=args.OPT1)
    else:
        G.calc_distances_all_vertices(compactDegree=args.OPT1)

    G.create_distances_network()
    G.preprocess_parameters_random_walk()
    G.simulate_walks(args.num_walks, args.walk_length)

    return G

def main(args):
    start = time()
    cmd = " ".join(shlex.quote(x) for x in sys.argv)

    G = exec_struc2vec(args)

    learn_embeddings()
    G.create_vectors()
    G.preprocess_degree_lists()

    suffix = pathlib.Path(args.output).stem
    target_prefix = args.prefix if args.prefix else suffix
    out_dir = pathlib.Path("pickles") / target_prefix
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in glob.glob("pickles/*.pickle"):
        shutil.move(p, str(out_dir))
    logging.info("All pickles moved to pickles/%s/", target_prefix)

    os.makedirs("time", exist_ok=True)
    elapsed = time() - start
    with open(f"time/{suffix}", "w") as f:
        f.write(cmd + "\n")
        f.write(f"{elapsed:.3f}\n")

    logging.info("Finished in %.3f s", elapsed)

if __name__ == "__main__":
    args = parse_args()
    main(args)
