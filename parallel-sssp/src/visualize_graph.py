import networkx as nx
import matplotlib.pyplot as plt

def read_bidirectional_graph(filename):
    G = nx.Graph()  # Undirected since graph_data.txt includes both directions
    with open(filename, 'r') as f:
        for line in f:
            u, v, w = map(int, line.strip().split())
            G.add_edge(u, v, weight=w)
    return G

def draw_graph(G):
    pos = nx.spring_layout(G, seed=42)  # For consistent layout
    edge_labels = nx.get_edge_attributes(G, 'weight')
    
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_size=700, node_color='skyblue', edge_color='gray', font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title("Bidirectional Graph Visualization")
    plt.show()

if __name__ == "__main__":
    G = read_bidirectional_graph("graph_data.txt")
    draw_graph(G)
