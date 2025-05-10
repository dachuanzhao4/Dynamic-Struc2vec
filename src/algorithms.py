from time import time
from collections import deque
import numpy as np
import math,random,logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from collections import defaultdict

from utils import *

__all__ = [
    "generate_parameters_random_walk",
    "generate_random_walks",
    "generate_random_walks_large_graphs",
    "exec_random_walk",
    "alias_draw",
    "prob_moveup",
]


def generate_parameters_random_walk():
    logging.info('Loading distances_nets from disk...')
    sum_weights = {}
    amount_edges = {}
    layer = 0

    while isPickle('distances_nets_weights-layer-' + str(layer)):
        logging.info('Executing layer {}...'.format(layer))
        weights = restoreVariableFromDisk('distances_nets_weights-layer-' + str(layer))
        for v, w_list in weights.items():
            sum_weights.setdefault(layer, 0.0)
            amount_edges.setdefault(layer, 0)
            for w in w_list:
                sum_weights[layer] += w
                amount_edges[layer] += 1
        logging.info('Layer {} executed.'.format(layer))
        layer += 1
    average_weight = {}
    for layer, total in sum_weights.items():
        count = amount_edges.get(layer, 0)
        if count > 0:
            average_weight[layer] = total / count
        else:
            average_weight[layer] = 0.0

    logging.info("Saving average_weight on disk...")
    saveVariableOnDisk(average_weight, 'average_weight')
    amount_neighbours = {}
    layer = 0
    while isPickle('distances_nets_weights-layer-' + str(layer)):
        logging.info('Executing layer {}...'.format(layer))
        weights = restoreVariableFromDisk('distances_nets_weights-layer-' + str(layer))
        amount_neighbours[layer] = {}
        for v, w_list in weights.items():
            cnt = 0
            thresh = average_weight.get(layer, 0.0)
            for w in w_list:
                if w > thresh:
                    cnt += 1
            amount_neighbours[layer][v] = cnt
        logging.info('Layer {} executed.'.format(layer))
        layer += 1

    logging.info("Saving amount_neighbours on disk...")
    saveVariableOnDisk(amount_neighbours, 'amount_neighbours')



def chooseNeighbor(v,graphs,alias_method_j,alias_method_q,layer):
    nbrs = graphs[layer].get(v, [])
    if not nbrs:
        return v

    J_table = alias_method_j[layer].get(v, None)
    q_table = alias_method_q[layer].get(v, None)
    if J_table is None or q_table is None or len(J_table) == 0:
        return v

    idx = alias_draw(J_table, q_table)
    return nbrs[idx]


def exec_random_walk(graphs,alias_method_j,alias_method_q,v,walk_length,amount_neighbours):
    original_v = v
    t0 = time()
    initialLayer = 0
    layer = initialLayer


    path = deque()
    path.append(v)

    while len(path) < walk_length:
        r = random.random()

        if(r < 0.3):
                v = chooseNeighbor(v,graphs,alias_method_j,alias_method_q,layer)
                path.append(v)

        else:
            r = random.random()
            limiar_moveup = prob_moveup(amount_neighbours[layer][v])
            if(r > limiar_moveup):
                if(layer > initialLayer):
                    layer = layer - 1           
            else:
                if((layer + 1) in graphs and v in graphs[layer + 1]):
                    layer = layer + 1

    t1 = time()
    logging.info('RW - vertex {}. Time : {}s'.format(original_v,(t1-t0)))

    return path


def exec_ramdom_walks_for_chunck(vertices,graphs,alias_method_j,alias_method_q,walk_length,amount_neighbours):
    walks = deque()
    for v in vertices:
        walks.append(exec_random_walk(graphs,alias_method_j,alias_method_q,v,walk_length,amount_neighbours))
    return walks

def generate_random_walks_large_graphs(num_walks,walk_length,workers,vertices):

    logging.info('Loading distances_nets from disk...')

    graphs = restoreVariableFromDisk('distances_nets_graphs')
    alias_method_j = restoreVariableFromDisk('nets_weights_alias_method_j')
    alias_method_q = restoreVariableFromDisk('nets_weights_alias_method_q')
    amount_neighbours = restoreVariableFromDisk('amount_neighbours')

    logging.info('Creating RWs...')
    t0 = time()
    
    walks = deque()
    initialLayer = 0

    parts = workers

    with ProcessPoolExecutor(max_workers=workers) as executor:

        for walk_iter in range(num_walks):
            random.shuffle(vertices)
            logging.info("Execution iteration {} ...".format(walk_iter))
            walk = exec_ramdom_walks_for_chunck(vertices,graphs,alias_method_j,alias_method_q,walk_length,amount_neighbours)
            walks.extend(walk)
            logging.info("Iteration {} executed.".format(walk_iter))



    t1 = time()
    logging.info('RWs created. Time : {}m'.format((t1-t0)/60))
    logging.info("Saving Random Walks on disk...")
    save_random_walks(walks)

def generate_random_walks(num_walks,walk_length,workers,vertices):

    logging.info('Loading distances_nets on disk...')

    graphs = restoreVariableFromDisk('distances_nets_graphs')
    alias_method_j = restoreVariableFromDisk('nets_weights_alias_method_j')
    alias_method_q = restoreVariableFromDisk('nets_weights_alias_method_q')
    amount_neighbours = restoreVariableFromDisk('amount_neighbours')

    logging.info('Creating RWs...')
    t0 = time()
    
    walks = deque()
    initialLayer = 0

    if(workers > num_walks):
        workers = num_walks

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for walk_iter in range(num_walks):
            random.shuffle(vertices)
            job = executor.submit(exec_ramdom_walks_for_chunck,vertices,graphs,alias_method_j,alias_method_q,walk_length,amount_neighbours)
            futures[job] = walk_iter
            #part += 1
        logging.info("Receiving results...")
        for job in as_completed(futures):
            walk = job.result()
            r = futures[job]
            logging.info("Iteration {} executed.".format(r))
            walks.extend(walk)
            del futures[job]


    t1 = time()
    logging.info('RWs created. Time: {}m'.format((t1-t0)/60))
    logging.info("Saving Random Walks on disk...")
    save_random_walks(walks)

def save_random_walks(walks):
    with open('random_walks.txt', 'w') as file:
        for walk in walks:
            line = ''
            for v in walk:
                line += str(v)+' '
            line += '\n'
            file.write(line)
    return

def prob_moveup(amount_neighbours):
    x = math.log(amount_neighbours + math.e)
    p = (x / ( x + 1))
    return p



def alias_draw(J, q):
    K = len(J)

    kk = int(np.floor(np.random.rand()*K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]