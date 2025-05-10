def count_nodes_in_edgelist(path):
    nodes = set()
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            u, v = parts[0], parts[1]
            nodes.add(u)
            nodes.add(v)
    return len(nodes)

if __name__ == "__main__":
    filepath = "graph/edgelist2010.edgelist"
    num_nodes = count_nodes_in_edgelist(filepath)
    print(f"Total nodes: {num_nodes}")
