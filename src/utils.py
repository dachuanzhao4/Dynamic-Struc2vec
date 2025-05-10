# -*- coding: utf-8 -*-
from time import time
import logging,inspect
import pickle as pickle
from itertools import islice
import os.path
import os, pickle
import math

dir_f = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
folder_pickles = dir_f+"/../pickles/"


def prob_moveup(num_neighbors: int) -> float:
    x = math.log(num_neighbors + math.e)
    return x / (x + 1)



def write_walks_to_disk(walks, out_path):
    out_path = str(out_path)
    with open(out_path, "w") as f:
        for walk in walks:
            f.write(" ".join(map(str, walk)) + "\n")
    return


def load_layer_weights(prefix: str, layer: int):

    fname = f"{prefix}/distances_nets_weights-layer-{layer}"
    return restoreVariableFromDisk(fname)

def returnPathStruc2vec():
    return dir_f

def isPickle(fname):
    return os.path.isfile(dir_f+'/../pickles/'+fname+'.pickle')

def chunks(data, SIZE=10000):
    it = iter(data)
    for i in range(0, len(data), SIZE):
        yield {k:data[k] for k in islice(it, SIZE)}

def partition(lst, n):
    division = len(lst) / float(n)
    return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n) ]

def restoreVariableFromDisk(name, prefix: str = ""):
    if prefix:
        path = os.path.join("pickles", prefix, f"{name}.pickle")
    else:
        path = os.path.join("pickles", f"{name}.pickle")

    if not os.path.isfile(path):    
        raise FileNotFoundError(f"[restore] '{path}' not found.")
    with open(path, "rb") as handle:
        return pickle.load(handle)

def saveVariableOnDisk(f,name):
    logging.info('Saving variable on disk...')
    t0 = time()
    with open(folder_pickles + name + '.pickle', 'wb') as handle:
        pickle.dump(f, handle, protocol=pickle.HIGHEST_PROTOCOL)
    t1 = time()
    logging.info('Variable saved. Time: {}m'.format((t1-t0)/60))

    return





