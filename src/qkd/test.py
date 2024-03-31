import matplotlib.pyplot as plt
import networkx as nx


# Function to draw the network diagram
def draw_network_diagram():
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes with the node name as the key
    nodes = ['Alice', 'Node1', 'Node2', 'Node3', 'Node4', 'Bob']
    G.add_nodes_from(nodes)

    # Add edges between nodes with direction
    edges = [('Alice', 'Node1'), ('Node1', 'Node2'), ('Node2', 'Node3'), ('Node3', 'Node4'), ('Node4', 'Bob')]
    G.add_edges_from(edges)

    # Draw the network
    pos = nx.shell_layout(G)  # Positions nodes in concentric circles
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='skyblue')
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, edge_color='gray')
    nx.draw_networkx_labels(G, pos, font_size=10)

    # Label the edges
    edge_labels = dict([((u, v,), f'Quantum Channel\n{u} to {v}') for u, v in edges])
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.5, font_size=9)

    # Set the size of the plot
    plt.figure(figsize=(12, 8))

    # Remove the axes
    plt.axis('off')

    # Save the plot to a file
    plt.savefig('network_diagram.png', format='PNG')
    plt.show()


# Call the function to draw the diagram
draw_network_diagram()
