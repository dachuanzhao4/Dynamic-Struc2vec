
import numpy as np
import random,sys,logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
from time import time
from collections import deque

from utils import *
from algorithms import *
from algorithms_distances import *
import graph
import pickle, re, glob, os



class Graph():
	def __init__(self, g, is_directed, workers, untilLayer = None):

		logging.info(" - Converting graph to dict...")
		self.G = g.gToDict()
		logging.info("Graph converted.")

		self.num_vertices = g.number_of_nodes()
		self.num_edges = g.number_of_edges()
		self.is_directed = is_directed
		self.workers = workers
		self.calcUntilLayer = untilLayer
		logging.info('Graph - Number of vertices: {}'.format(self.num_vertices))
		logging.info('Graph - Number of edges: {}'.format(self.num_edges))


	def preprocess_neighbors_with_bfs(self):

		with ProcessPoolExecutor(max_workers=self.workers) as executor:
			job = executor.submit(exec_bfs,self.G,self.workers,self.calcUntilLayer)
			
			job.result()

		return

	def preprocess_neighbors_with_bfs_compact(self):

		with ProcessPoolExecutor(max_workers=self.workers) as executor:
			job = executor.submit(exec_bfs_compact,self.G,self.workers,self.calcUntilLayer)
			
			job.result()

		return

	def preprocess_degree_lists(self):

		with ProcessPoolExecutor(max_workers=self.workers) as executor:
			job = executor.submit(preprocess_degreeLists)
			
			job.result()

		return


	def create_vectors(self):
		logging.info("Creating degree vectors...")
		degrees = {}
		degrees_sorted = set()
		G = self.G
		for v in list(G.keys()):
			degree = len(G[v])
			degrees_sorted.add(degree)
			if(degree not in degrees):
				degrees[degree] = {}
				degrees[degree]['vertices'] = deque() 
			degrees[degree]['vertices'].append(v)
		degrees_sorted = np.array(list(degrees_sorted),dtype='int')
		degrees_sorted = np.sort(degrees_sorted)

		l = len(degrees_sorted)
		for index, degree in enumerate(degrees_sorted):
			if(index > 0):
				degrees[degree]['before'] = degrees_sorted[index - 1]
			if(index < (l - 1)):
				degrees[degree]['after'] = degrees_sorted[index + 1]
		logging.info("Degree vectors created.")
		logging.info("Saving degree vectors...")
		saveVariableOnDisk(degrees,'degrees_vector')


	def calc_distances_all_vertices(self,compactDegree = False):

		logging.info("Using compactDegree: {}".format(compactDegree))
		if(self.calcUntilLayer):
			logging.info("Calculations until layer: {}".format(self.calcUntilLayer))

		futures = {}

		count_calc = 0

		vertices = list(reversed(sorted(self.G.keys())))

		if(compactDegree):
		    logging.info("Recovering degreeList from disk...")
		    degreeList = restoreVariableFromDisk('compactDegreeList')
		else:
		    logging.info("Recovering compactDegreeList from disk...")
		    degreeList = restoreVariableFromDisk('degreeList')

		parts = self.workers
		chunks = partition(vertices,parts)

		t0 = time()

		with ProcessPoolExecutor(max_workers = self.workers) as executor:

			part = 1
			for c in chunks:
				logging.info("Executing part {}...".format(part))
				list_v = []
				for v in c:
					list_v.append([vd for vd in list(degreeList.keys()) if vd > v])
				job = executor.submit(calc_distances_all, c, list_v, degreeList,part, compactDegree = compactDegree)
				futures[job] = part
				part += 1


			logging.info("Receiving results...")

			for job in as_completed(futures):
				job.result()
				r = futures[job]
				logging.info("Part {} Completed.".format(r))
		
		logging.info('Distances calculated.')
		t1 = time()
		logging.info('Time : {}m'.format((t1-t0)/60))
		
		return


	def calc_distances(self, compactDegree = False):

		logging.info("Using compactDegree: {}".format(compactDegree))
		if(self.calcUntilLayer):
			logging.info("Calculations until layer: {}".format(self.calcUntilLayer))

		futures = {}
		#distances = {}

		count_calc = 0

		G = self.G
		vertices = list(G.keys())

		parts = self.workers
		chunks = partition(vertices,parts)

		with ProcessPoolExecutor(max_workers = 1) as executor:

			logging.info("Split degree List...")
			part = 1
			for c in chunks:
				job = executor.submit(splitDegreeList,part,c,G,compactDegree)
				job.result()
				logging.info("degreeList {} completed.".format(part))
				part += 1

		
		with ProcessPoolExecutor(max_workers = self.workers) as executor:

			part = 1
			for c in chunks:
				logging.info("Executing part {}...".format(part))
				job = executor.submit(calc_distances, part, compactDegree = compactDegree)
				futures[job] = part
				part += 1

			logging.info("Receiving results...")
			for job in as_completed(futures):
				job.result()
				r = futures[job]
				logging.info("Part {} completed.".format(r))


		return

	def consolide_distances(self):

		distances = {}

		parts = self.workers
		for part in range(1,parts + 1):
			d = restoreVariableFromDisk('distances-'+str(part))
			preprocess_consolides_distances(distances)
			distances.update(d)


		preprocess_consolides_distances(distances)
		saveVariableOnDisk(distances,'distances')


	def create_distances_network(self):

		with ProcessPoolExecutor(max_workers=1) as executor:
			job = executor.submit(generate_distances_network,self.workers)

			job.result()

		return

	def preprocess_parameters_random_walk(self):

		with ProcessPoolExecutor(max_workers=1) as executor:
			job = executor.submit(generate_parameters_random_walk)

			job.result()

		return


	def simulate_walks(self,num_walks,walk_length):

		# for large graphs, it is serially executed, because of memory use.
		if(len(self.G) > 500000):

			with ProcessPoolExecutor(max_workers=1) as executor:
				job = executor.submit(generate_random_walks_large_graphs,num_walks,walk_length,self.workers,list(self.G.keys()))

				job.result()

		else:

			with ProcessPoolExecutor(max_workers=1) as executor:
				job = executor.submit(generate_random_walks,num_walks,walk_length,self.workers,list(self.G.keys()))

				job.result()


		return	

	def preprocess_components(self, NUM_LAYERS=5):
		self.preprocess_neighbors_with_bfs_compact()
		self.preprocess_degree_lists()
		self.create_vectors()
		self.calc_distances_all_vertices()        
		self.create_distances_network()        
		self.preprocess_parameters_random_walk()
		return

	def save_pickle_files(self, snapshot_dir):
		import shutil, pathlib, glob, os
		snapshot_dir = pathlib.Path(snapshot_dir)
		snapshot_dir.mkdir(parents=True, exist_ok=True)
		for p in glob.glob(f"{folder_pickles}/*.pickle"):
			shutil.copy2(p, snapshot_dir / os.path.basename(p))
		return
	
	def load_walk_graph(self, prefix: str):
		self.graphs = restoreVariableFromDisk("distances_nets_graphs", prefix)
		self.alias_method_j = restoreVariableFromDisk("nets_weights_alias_method_j", prefix)
		self.alias_method_q = restoreVariableFromDisk("nets_weights_alias_method_q", prefix)
		logging.info("Graph walk structures loaded into memory from prefix '%s'", prefix)
	def load_pickle_files(self, snapshot_dir: str):
		import os, glob, shutil
		from utils import folder_pickles
		os.makedirs(folder_pickles, exist_ok=True)

		for f in glob.glob(os.path.join(folder_pickles, "*.pickle")):
			os.remove(f)

		for f in glob.glob(os.path.join(snapshot_dir, "*.pickle")):
			shutil.copy2(f, os.path.join(folder_pickles, os.path.basename(f)))


	def _choose_next(self, cur_node, cur_layer):
		max_layer = max(self.alias_method_j.keys())
		r = np.random.rand()
		if cur_layer == 0:
			if r < 0.5:
				neighbors = self.graphs[cur_layer].get(cur_node, [])
				if neighbors:
					idx = alias_draw(self.alias_method_j[cur_layer][cur_node],
									self.alias_method_q[cur_layer][cur_node])
					next_node = neighbors[idx]
					next_layer = cur_layer
				else:
					next_node = cur_node
					next_layer = cur_layer + 1
			else:
				next_node = cur_node
				next_layer = cur_layer + 1
		elif cur_layer == max_layer:
			if r < 0.5:
				neighbors = self.graphs[cur_layer].get(cur_node, [])
				if neighbors:
					idx = alias_draw(self.alias_method_j[cur_layer][cur_node],
									self.alias_method_q[cur_layer][cur_node])
					next_node = neighbors[idx]
					next_layer = cur_layer
				else:
					next_node = cur_node
					next_layer = cur_layer - 1
			else:
				next_node = cur_node
				next_layer = cur_layer - 1
		else:
			if r < 0.3333:
				next_node = cur_node
				next_layer = cur_layer - 1
			elif r < 0.6666:
				next_node = cur_node
				next_layer = cur_layer + 1
			else:
				neighbors = self.graphs[cur_layer].get(cur_node, [])
				if neighbors:
					idx = alias_draw(self.alias_method_j[cur_layer][cur_node],
									self.alias_method_q[cur_layer][cur_node])
					next_node = neighbors[idx]
					next_layer = cur_layer
				else:
					if np.random.rand() < 0.5:
						next_node = cur_node
						next_layer = cur_layer - 1
					else:
						next_node = cur_node
						next_layer = cur_layer + 1
		return next_node, next_layer




	








		

      	


