import networkx as nx
import matplotlib.pyplot as plt
import random

# Create a random graph with n users and p probability of connection
def create_random_graph(n, p, graph_name=None):
    # Generate the graph
    G = nx.erdos_renyi_graph(n, p)

    if graph_name != None:
        # Draw the graph
        nx.draw(G, with_labels=True)
        plt.savefig(graph_name)
    return G

# Create a randomized ring graph with n users arranged in a ring
def create_random_ring_graph(n, graph_name=None):
    # Create a list of nodes
    nodes = list(range(n))
    
    # Shuffle the nodes to randomize connections
    random.shuffle(nodes)
    
    # Create an empty graph
    G = nx.Graph()
    
    # Add nodes to the graph
    G.add_nodes_from(range(n))

    # Connect nodes in a ring topology
    for i in range(n):
        G.add_edge(nodes[i], nodes[(i+1) % n])
    return G

# Create a regular graph with n users and d degree of each node
def create_regular_graph(n, d, graph_name=None):
    # Generate the graph
    G = nx.random_regular_graph(d, n)
    
    if graph_name != None:
        # Draw the graph
        nx.draw(G, with_labels=True)
        plt.savefig(graph_name)
    return G