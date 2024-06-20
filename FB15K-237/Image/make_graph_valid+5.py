import matplotlib.pyplot as plt
import networkx as nx

# Define the path to the file
file_path = 'valid+5.txt'

# Initialize a directed graph
G = nx.DiGraph()

# Read and parse the file
with open(file_path, 'r') as file:
    i=0
    for line in file:
        i+=1
        parts = line.strip().split('\t')
        if len(parts) == 3:
            node_dep, edge_name, node_arr = parts
            G.add_edge(node_dep, node_arr, label=edge_name)

# Plot the graph
plt.figure(figsize=(50, 50), dpi=300)  # Larger figure size and higher dpi for better resolution
pos = nx.spring_layout(G, k=0.1, iterations=10)  # More iterations for a more spaced-out layout
                                                 # Adjust k to scale the distance between nodes

# Draw nodes and edges
nx.draw_networkx_nodes(G, pos, node_size=100, node_color='skyblue', alpha=0.7)
nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=15, edge_color='gray', width=1.5, alpha=0.6)
# nx.draw_networkx_labels(G, pos, font_size=8, font_color='black', font_weight='bold')


# Optionally, draw edge labels
# edge_labels = nx.get_edge_attributes(G, 'label')
# nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=5)

plt.title("Graph Visualization")
plt.savefig('graph_valid+5.png')
plt.show()
