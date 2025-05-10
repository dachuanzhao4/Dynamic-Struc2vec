
from time import time
from collections import deque
import numpy as np
import math,logging
from fastdtw import fastdtw
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from utils import *
import os, logging
from typing import Iterable
from utils import restoreVariableFromDisk, saveVariableOnDisk, folder_pickles

limiteDist = 20

def getDegreeListsVertices(g,vertices,calcUntilLayer):
    degreeList = {}

    for v in vertices:
        degreeList[v] = getDegreeLists(g,v,calcUntilLayer)

    return degreeList

def getCompactDegreeListsVertices(g,vertices,maxDegree,calcUntilLayer):
    degreeList = {}

    for v in vertices:
        degreeList[v] = getCompactDegreeLists(g,v,maxDegree,calcUntilLayer)

    return degreeList


def getCompactDegreeLists(g, root, maxDegree,calcUntilLayer):
    t0 = time()

    listas = {}
    vetor_marcacao = [0] * (max(g) + 1)

    queue = deque()
    queue.append(root)
    vetor_marcacao[root] = 1
    l = {}
    depth = 0
    pendingDepthIncrease = 0
    timeToDepthIncrease = 1

    while queue:
        vertex = queue.popleft()
        timeToDepthIncrease -= 1

        d = len(g[vertex])
        if(d not in l):
            l[d] = 0
        l[d] += 1

        for v in g[vertex]:
            if(vetor_marcacao[v] == 0):
                vetor_marcacao[v] = 1
                queue.append(v)
                pendingDepthIncrease += 1    

        if(timeToDepthIncrease == 0):

            list_d = []
            for degree,freq in l.items():
                list_d.append((degree,freq))
            list_d.sort(key=lambda x: x[0])
            listas[depth] = np.array(list_d,dtype=np.int32)

            l = {}

            if(calcUntilLayer == depth):
                break

            depth += 1
            timeToDepthIncrease = pendingDepthIncrease
            pendingDepthIncrease = 0


    t1 = time()
    logging.info('BFS vertex {}. Time: {}s'.format(root,(t1-t0)))

    return listas


def getDegreeLists(g, root, calcUntilLayer):
    t0 = time()

    listas = {}
    vetor_marcacao = [0] * (max(g) + 1)

    queue = deque()
    queue.append(root)
    vetor_marcacao[root] = 1
    

    l = deque()
    depth = 0
    pendingDepthIncrease = 0
    timeToDepthIncrease = 1

    while queue:
        vertex = queue.popleft()
        timeToDepthIncrease -= 1

        l.append(len(g[vertex]))

        for v in g[vertex]:
            if(vetor_marcacao[v] == 0):
                vetor_marcacao[v] = 1
                queue.append(v)
                pendingDepthIncrease += 1    

        if(timeToDepthIncrease == 0):

            lp = np.array(l,dtype='float')
            lp = np.sort(lp)
            listas[depth] = lp
            l = deque()

            if(calcUntilLayer == depth):
                break

            depth += 1
            timeToDepthIncrease = pendingDepthIncrease
            pendingDepthIncrease = 0


    t1 = time()
    logging.info('BFS vertex {}. Time: {}s'.format(root,(t1-t0)))


    return listas

def cost(a, b):
    ep = 0.5
    m  = max(a, b) + ep
    mi = min(a, b) + ep
    return (m / mi) - 1

def cost_min(a, b):
    ep = 0.5
    m  = max(a[0], b[0]) + ep
    mi = min(a[0], b[0]) + ep
    return ((m / mi) - 1) * min(a[1], b[1])

def cost_max(a, b):

    ep = 0.5
    m  = max(a[0], b[0]) + ep
    mi = min(a[0], b[0]) + ep
    return ((m / mi) - 1) * max(a[1], b[1])


def preprocess_degreeLists():

    logging.info("Recovering degreeList from disk...")
    try:
        degreeList = restoreVariableFromDisk('degreeList')
    except FileNotFoundError:
        logging.info(
            "'pickles/degreeList.pickle' not found; "
            "skipping creation of compactDegreeList."
        )
        return

    logging.info("Creating compactDegreeList...")
    dFrequency: Dict[int, Dict[int, Dict[int, int]]] = {}
    for v, layers in degreeList.items():
        dFrequency[v] = {}
        for layer, degreeListLayer in layers.items():
            freq: Dict[int, int] = {}
            for degree in degreeListLayer:
                freq[degree] = freq.get(degree, 0) + 1
            dFrequency[v][layer] = freq

    dList: Dict[int, Dict[int, np.ndarray]] = {}
    for v, layers in dFrequency.items():
        dList[v] = {}
        for layer, freq_map in layers.items():
            items = sorted(freq_map.items(), key=lambda x: x[0])
            dList[v][layer] = np.array(items, dtype='float')

    saveVariableOnDisk(dList, 'compactDegreeList')
    logging.info("compactDegreeList.pickle has been (re)created.")


def verifyDegrees(degrees,degree_v_root,degree_a,degree_b):

    if(degree_b == -1):
        degree_now = degree_a
    elif(degree_a == -1):
        degree_now = degree_b
    elif(abs(degree_b - degree_v_root) < abs(degree_a - degree_v_root)):
        degree_now = degree_b
    else:
        degree_now = degree_a

    return degree_now 

def get_vertices(v,degree_v,degrees,a_vertices):
    a_vertices_selected = 2 * math.log(a_vertices,2)
    vertices = deque()

    try:
        c_v = 0  

        for v2 in degrees[degree_v]['vertices']:
            if(v != v2):
                vertices.append(v2)
                c_v += 1
                if(c_v > a_vertices_selected):
                    raise StopIteration

        if('before' not in degrees[degree_v]):
            degree_b = -1
        else:
            degree_b = degrees[degree_v]['before']
        if('after' not in degrees[degree_v]):
            degree_a = -1
        else:
            degree_a = degrees[degree_v]['after']
        if(degree_b == -1 and degree_a == -1):
            raise StopIteration
        degree_now = verifyDegrees(degrees,degree_v,degree_a,degree_b)

        while True:
            for v2 in degrees[degree_now]['vertices']:
                if(v != v2):
                    vertices.append(v2)
                    c_v += 1
                    if(c_v > a_vertices_selected):
                        raise StopIteration

            if(degree_now == degree_b):
                if('before' not in degrees[degree_b]):
                    degree_b = -1
                else:
                    degree_b = degrees[degree_b]['before']
            else:
                if('after' not in degrees[degree_a]):
                    degree_a = -1
                else:
                    degree_a = degrees[degree_a]['after']
            
            if(degree_b == -1 and degree_a == -1):
                raise StopIteration

            degree_now = verifyDegrees(degrees,degree_v,degree_a,degree_b)

    except StopIteration:
        return list(vertices)

    return list(vertices)


def splitDegreeList(part, seeds, G, compactDegree, prefix=None):
    if compactDegree:
        logging.info("Recovering compactDegreeList from pickles/%s...", prefix)
        degreeList = restoreVariableFromDisk('compactDegreeList', prefix)
    else:
        logging.info("Recovering degreeList from pickles/%s...", prefix)
        degreeList = restoreVariableFromDisk('degreeList', prefix)

    logging.info("Recovering degrees_vector from pickles/%s...", prefix)
    degrees = restoreVariableFromDisk('degrees_vector', prefix)
    all_layers = []
    for layers_map in degreeList.values():
        all_layers.extend(layers_map.keys())
    calcUntil = max(all_layers) if all_layers else 0

    degreeListsSelected = {}
    vertices = {}
    total_nodes = len(G)

    for v in seeds:
        if v not in degreeList:
            logging.info("New node %s detected; running BFS up to layer %d", v, calcUntil)
            if compactDegree:
                single = getCompactDegreeLists(G, v, max(G.keys()), calcUntil)
            else:
                single = getDegreeLists(G, v, calcUntil)
            degreeList[v] = single

        try:
            nbs = get_vertices(v, len(G[v]), degrees, total_nodes)
        except Exception as e:
            logging.warning("get_vertices failed for %s: %s", v, e)
            nbs = []

        vertices[v] = nbs
        degreeListsSelected[v] = degreeList[v]
        for n in nbs:
            if n in degreeList:
                degreeListsSelected[n] = degreeList[n]

    saveVariableOnDisk(vertices, f'split-vertices-{part}')
    saveVariableOnDisk(degreeListsSelected, f'split-degreeList-{part}')




def calc_distances(part, compactDegree=False, prefix=""):
    vertices = restoreVariableFromDisk(f"split-vertices-{part}", prefix)
    degreeList = restoreVariableFromDisk(f"split-degreeList-{part}", prefix)

    distances = {}

    if compactDegree:
        dist_func = cost_max
    else:
        dist_func = cost

    for v1, nbs in vertices.items():
        lists_v1 = degreeList[v1]

        for v2 in nbs:
            t00 = time()
            lists_v2 = degreeList[v2]

            max_layer = min(len(lists_v1), len(lists_v2))
            distances[(v1, v2)] = {}

            for layer in range(max_layer):
                dist, path = fastdtw(
                    lists_v1[layer],
                    lists_v2[layer],
                    radius=1,
                    dist=dist_func
                )
                distances[(v1, v2)][layer] = dist

            t11 = time()
            logging.info(
                'fastDTW between vertices (%d, %d). Time: %.2fs',
                v1, v2, (t11 - t00)
            )

    preprocess_consolides_distances(distances)
    saveVariableOnDisk(distances, f"distances-{part}")

def calc_distances_all(vertices,list_vertices,degreeList,part, compactDegree = False):

    distances = {}
    cont = 0

    if compactDegree:
        dist_func = cost_max
    else:
        dist_func = cost

    for v1 in vertices:
        lists_v1 = degreeList[v1]

        for v2 in list_vertices[cont]:
            lists_v2 = degreeList[v2]
            
            max_layer = min(len(lists_v1),len(lists_v2))
            distances[v1,v2] = {}

            for layer in range(0,max_layer):
                #t0 = time()
                dist, path = fastdtw(lists_v1[layer],lists_v2[layer],radius=1,dist=dist_func)
                #t1 = time()
                distances[v1,v2][layer] = dist
                

        cont += 1

    preprocess_consolides_distances(distances)
    saveVariableOnDisk(distances,'distances-'+str(part))
    return


def selectVertices(layer,fractionCalcDists):
    previousLayer = layer - 1

    logging.info("Recovering distances from disk...")
    distances = restoreVariableFromDisk('distances')

    threshold = calcThresholdDistance(previousLayer,distances,fractionCalcDists)

    logging.info('Selecting vertices...')

    vertices_selected = deque()

    for vertices,layers in distances.items():
        if(previousLayer not in layers):
            continue
        if(layers[previousLayer] <= threshold):
            vertices_selected.append(vertices)

    distances = {}

    logging.info('Vertices selected.')

    return vertices_selected


def preprocess_consolides_distances(distances, startLayer = 1):

    logging.info('Consolidating distances...')

    for vertices,layers in distances.items():
        keys_layers = list(sorted(layers.keys()))
        startLayer = min(len(keys_layers),startLayer)
        for layer in range(0,startLayer):
            keys_layers.pop(0)


        for layer in keys_layers:
            layers[layer] += layers[layer - 1]

    logging.info('Distances consolidated.')


def exec_bfs_compact(G,workers,calcUntilLayer):

    futures = {}
    degreeList = {}

    t0 = time()
    vertices = list(G.keys())
    parts = workers
    chunks = partition(vertices,parts)

    logging.info('Capturing larger degree...')
    maxDegree = 0
    for v in vertices:
        if(len(G[v]) > maxDegree):
            maxDegree = len(G[v])
    logging.info('Larger degree captured')

    with ProcessPoolExecutor(max_workers=workers) as executor:

        part = 1
        for c in chunks:
            job = executor.submit(getCompactDegreeListsVertices,G,c,maxDegree,calcUntilLayer)
            futures[job] = part
            part += 1

        for job in as_completed(futures):
            dl = job.result()
            v = futures[job]
            degreeList.update(dl)

    logging.info("Saving degreeList on disk...")
    saveVariableOnDisk(degreeList,'compactDegreeList')
    t1 = time()
    logging.info('Execution time - BFS: {}m'.format((t1-t0)/60))


    return

def exec_bfs(G,workers,calcUntilLayer):

    futures = {}
    degreeList = {}

    t0 = time()
    vertices = list(G.keys())
    parts = workers
    chunks = partition(vertices,parts)

    with ProcessPoolExecutor(max_workers=workers) as executor:

        part = 1
        for c in chunks:
            job = executor.submit(getDegreeListsVertices,G,c,calcUntilLayer)
            futures[job] = part
            part += 1

        for job in as_completed(futures):
            dl = job.result()
            v = futures[job]
            degreeList.update(dl)

    logging.info("Saving degreeList on disk...")
    saveVariableOnDisk(degreeList,'degreeList')
    t1 = time()
    logging.info('Execution time - BFS: {}m'.format((t1-t0)/60))


    return





def generate_distances_network_part1(
    workers: int,
    prefix: str = "",
    seeds: Iterable[int] = None
):
    if prefix and seeds is not None:
        logging.info("Part1 (incremental): loading incremental distances from pickles/%s...", prefix)
        new_dist = restoreVariableFromDisk("distances-1", prefix)
        layers = { L for layer_map in new_dist.values() for L in layer_map }
        for L in layers:
            fname = f"weights_distances-layer-{L}"
            try:
                old_w = restoreVariableFromDisk(fname)
            except FileNotFoundError:
                old_w = {}
            for (u, v), layer_map in new_dist.items():
                if L in layer_map and (u in seeds or v in seeds):
                    old_w[(u, v)] = layer_map[L]
            saveVariableOnDisk(old_w, fname)
            logging.info("Part1 (incremental): merged layer %d into %s.pickle", L, fname)
        logging.info("Part1 (incremental) complete.")
    else:
        logging.info("Part1 (full): merging distances from all %d parts...", workers)
        weights: dict[int, dict[tuple[int,int], float]] = {}
        for part in range(1, workers + 1):
            logging.info("Part %d (full) loading distances-%d.pickle...", part, part)
            distances = restoreVariableFromDisk(f"distances-{part}")
            for (u, v), layer_map in distances.items():
                for L, dist in layer_map.items():
                    weights.setdefault(L, {})[(u, v)] = dist
        for L, wmap in weights.items():
            saveVariableOnDisk(wmap, f"weights_distances-layer-{L}")
            logging.info("Part1 (full): wrote weights_distances-layer-%d.pickle", L)
        logging.info("Part1 (full) complete.")



def generate_distances_network_part2(workers=1, prefix="", seeds=None):
    layer = 0
    while isPickle(f"{prefix}weights_distances-layer-{layer}"):
        wmap = restoreVariableFromDisk(f"weights_distances-layer-{layer}", prefix)

        if seeds:
            graphs = restoreVariableFromDisk(f"graphs-layer-{layer}", prefix)
        else:
            from collections import defaultdict
            graphs = defaultdict(list)
            for (u, v), dist in wmap.items():
                graphs[u].append(v)
                graphs[v].append(u)
        if seeds:
            for v_new in seeds:
                graphs.setdefault(v_new, [])
            for (u, v), dist in wmap.items():
                if u in seeds or v in seeds:
                    if v not in graphs[u]:
                        graphs[u].append(v)
                    if u not in graphs[v]:
                        graphs[v].append(u)
        saveVariableOnDisk(graphs, f"graphs-layer-{layer}")
        logging.info(f"Part2 {'incremental' if seeds else 'full'}: wrote graphs-layer-{layer}.pickle")
        layer += 1

    logging.info("Part2 complete.")






def generate_distances_network_part3():
    layer = 0
    while isPickle(f'graphs-layer-{layer}'):
        graphs = restoreVariableFromDisk(f'graphs-layer-{layer}')
        weights_distances = restoreVariableFromDisk(f'weights_distances-layer-{layer}')
        logging.info(f'Executing layer {layer}...')
        alias_method_j = {}
        alias_method_q = {}
        weights = {}
        for v, neighbors in graphs.items():  
            e_list = []
            sum_w = 0.0
            for n in neighbors:
                if (v, n) in weights_distances:
                    wd = weights_distances[(v, n)]
                else:
                    wd = weights_distances[(n, v)]
                w = math.exp(-float(wd))
                e_list.append(w)
                sum_w += w
            if e_list:
                if sum_w > 0.0:
                    e_list = [x / sum_w for x in e_list]
                else:
                    n = len(e_list)
                    e_list = [1.0/n] * n
            weights[v] = e_list
            J, q = alias_setup(e_list)
            alias_method_j[v] = J
            alias_method_q[v] = q
        saveVariableOnDisk(weights, f'distances_nets_weights-layer-{layer}')
        saveVariableOnDisk(alias_method_j, f'alias_method_j-layer-{layer}')
        saveVariableOnDisk(alias_method_q, f'alias_method_q-layer-{layer}')
        logging.info(f'Layer {layer} executed.')
        layer += 1
    logging.info('Weights and alias tables created.')



def generate_distances_network_part4():
    logging.info('Consolidating graphs...')
    graphs_c = {}
    layer = 0
    while(isPickle('graphs-layer-'+str(layer))):
        logging.info('Executing layer {}...'.format(layer))
        graphs = restoreVariableFromDisk('graphs-layer-'+str(layer))
        graphs_c[layer] = graphs
        logging.info('Layer {} executed.'.format(layer))
        layer += 1


    logging.info("Saving distancesNets on disk...")
    saveVariableOnDisk(graphs_c,'distances_nets_graphs')
    logging.info('Graphs consolidated.')
    return

def generate_distances_network_part5():
    alias_method_j_c = {}
    layer = 0
    while(isPickle('alias_method_j-layer-'+str(layer))):
        logging.info('Executing layer {}...'.format(layer))          
        alias_method_j = restoreVariableFromDisk('alias_method_j-layer-'+str(layer))
        alias_method_j_c[layer] = alias_method_j
        logging.info('Layer {} executed.'.format(layer))
        layer += 1

    logging.info("Saving nets_weights_alias_method_j on disk...")
    saveVariableOnDisk(alias_method_j_c,'nets_weights_alias_method_j')

    return

def generate_distances_network_part6():
    alias_method_q_c = {}
    layer = 0
    while(isPickle('alias_method_q-layer-'+str(layer))):
        logging.info('Executing layer {}...'.format(layer))          
        alias_method_q = restoreVariableFromDisk('alias_method_q-layer-'+str(layer))
        alias_method_q_c[layer] = alias_method_q
        logging.info('Layer {} executed.'.format(layer))
        layer += 1

    logging.info("Saving nets_weights_alias_method_q on disk...")
    saveVariableOnDisk(alias_method_q_c,'nets_weights_alias_method_q')

    return

def generate_distances_network(workers):
    t0 = time()
    logging.info('Creating distance network...')

    os.system("rm "+returnPathStruc2vec()+"/../pickles/weights_distances-layer-*.pickle")
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generate_distances_network_part1,workers)
        job.result()
    t1 = time()
    t = t1-t0
    logging.info('- Time - part 1: {}s'.format(t))

    t0 = time()
    os.system("rm "+returnPathStruc2vec()+"/../pickles/graphs-layer-*.pickle")
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generate_distances_network_part2,workers)
        job.result()
    t1 = time()
    t = t1-t0
    logging.info('- Time - part 2: {}s'.format(t))
    logging.info('distance network created.')

    logging.info('Transforming distances into weights...')

    t0 = time()
    os.system("rm "+returnPathStruc2vec()+"/../pickles/distances_nets_weights-layer-*.pickle")
    os.system("rm "+returnPathStruc2vec()+"/../pickles/alias_method_j-layer-*.pickle")
    os.system("rm "+returnPathStruc2vec()+"/../pickles/alias_method_q-layer-*.pickle")
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generate_distances_network_part3)
        job.result()
    t1 = time()
    t = t1-t0
    logging.info('- Time - part 3: {}s'.format(t))

    t0 = time()
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generate_distances_network_part4)
        job.result()
    t1 = time()
    t = t1-t0
    logging.info('- Time - part 4: {}s'.format(t))

    t0 = time()
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generate_distances_network_part5)
        job.result()
    t1 = time()
    t = t1-t0
    logging.info('- Time - part 5: {}s'.format(t))

    t0 = time()
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generate_distances_network_part6)
        job.result()
    t1 = time()
    t = t1-t0
    logging.info('- Time - part 6: {}s'.format(t))
 
    return


def alias_setup(probs):
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q