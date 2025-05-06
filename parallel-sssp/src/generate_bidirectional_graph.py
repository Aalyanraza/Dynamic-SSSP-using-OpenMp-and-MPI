import random

def generate_undirected_graph(num_nodes, num_edges, weight_range=(1, 10), output_file="graph_data.txt"):
    edges = set()
    max_possible_edges = num_nodes * (num_nodes - 1) // 2
    num_edges = min(num_edges, max_possible_edges)

    while len(edges) < num_edges:
        u = random.randint(0, num_nodes - 1)
        v = random.randint(0, num_nodes - 1)
        if u != v:
            # Sort nodes to avoid both (u,v) and (v,u)
            edge = tuple(sorted((u, v)))
            if edge not in edges:
                w = random.randint(*weight_range)
                edges.add((edge[0], edge[1], w))

    with open(output_file, "w") as f:
        for u, v, w in edges:
            f.write(f"{u} {v} {w}\n")

    print(f"✅ Undirected graph with {num_nodes} nodes and {len(edges)} edges saved to '{output_file}'.")

# Example usage
if __name__ == "__main__":
    generate_undirected_graph(num_nodes=9, num_edges=13)
