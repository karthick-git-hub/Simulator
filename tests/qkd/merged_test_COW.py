
def generate_adjacency_matrix(topology, m):
    n = m;
    size = n * n

    if (topology == 'Grid') or (topology == 'grid'):
        adj_matrix = np.zeros((size, size))
        for row in range(n):
            for col in range(n):
                index = row * n + col

                # Connect to the right
                if col < n - 1:
                    adj_matrix[index][index + 1] = 1
                    adj_matrix[index + 1][index] = 1

                # Connect downwards
                if row < n - 1:
                    adj_matrix[index][index + n] = 1
                    adj_matrix[index + n][index] = 1
        return adj_matrix

    elif (topology == 'ring') or (topology == 'Ring'):
        adjacency_matrix = np.zeros((n, n))
        for i in range(n):
            adjacency_matrix[i][(i - 1) % n] = 1  # Connect to previous node
            adjacency_matrix[i][(i + 1) % n] = 1  # Connect to next node
        return adjacency_matrix

    elif (topology == 'star') or (topology == 'Star'):
        adjacency_matrix = np.zeros((n + 1, n + 1))
        for i in range(1, n + 1):
            adjacency_matrix[i][0] = adjacency_matrix[0][i] = 1
        return adjacency_matrix

    elif (topology == 'torus') or (topology == 'Torus'):
        totalNodes = size
        totalNodesX = n
        totalNodesY = n
        adjacencyMatrix = np.zeros((totalNodes, totalNodes))

        for i in range(totalNodesX):
            for j in range(totalNodesY):
                node = i * totalNodesY + j

                # Calculate the left neighbor
                if j == 0:
                    leftNeighbor = node + (totalNodesY - 1)
                else:
                    leftNeighbor = node - 1

                # Calculate the upper neighbor
                if i == 0:
                    upperNeighbor = node + (totalNodesX - 1) * totalNodesY
                else:
                    upperNeighbor = node - totalNodesY

                # Update the adjacency matrix
                adjacencyMatrix[node, leftNeighbor] = 1
                adjacencyMatrix[leftNeighbor, node] = 1
                adjacencyMatrix[node, upperNeighbor] = 1
                adjacencyMatrix[upperNeighbor, node] = 1

        return adjacencyMatrix

def get_nodes(topology, n):
    topology = topology.lower()
    if topology in ['ring', 'grid', 'torus']:
        if topology == 'ring':
            adj_matrix = generate_adjacency_matrix('Ring', n)
            start_node = 0
            end_node = n // 2
        elif topology == 'grid':
            adj_matrix = generate_adjacency_matrix('Grid', n)
            start_node = 0
            end_node = n * n - 1
        elif topology == 'torus':
            adj_matrix = generate_adjacency_matrix('Torus', n)
            start_node = 0
            end_node = n * n - 1
        nodes_in_between = shortest_path_length_bfs(adj_matrix, start_node, end_node)
        return nodes_in_between
    else:
        raise ValueError('Error in topology: use either of ring, grid, or torus topology')

def shortest_path_length_bfs(adjacency_matrix, start_node, end_node):
    """
    Find the length of the shortest path in an unweighted graph using BFS.

    Args:
    adjacency_matrix (numpy.ndarray): The adjacency matrix of the graph.
    start_node (int): The starting node index.
    end_node (int): The ending node index.

    Returns:
    int: The number of nodes in between the start_node and end_node in the shortest path.
    """
    num_nodes = adjacency_matrix.shape[0]
    visited = [False] * num_nodes
    prev = [None] * num_nodes

    # BFS
    queue = deque([start_node])
    visited[start_node] = True

    while queue:
        node = queue.popleft()

        # Visit the neighbors
        for neighbor, connected in enumerate(adjacency_matrix[node]):
            if connected and not visited[neighbor]:
                queue.append(neighbor)
                visited[neighbor] = True
                prev[neighbor] = node

    # Reconstruct the path
    path = []
    at = end_node
    while at is not None:
        path.append(at)
        at = prev[at]
    path.reverse()

    # Return the number of nodes in between start and end, excluding start and end
    nonodes = len(path) - 2 if path[0] == start_node and len(path) > 1 else 0
    return nonodes


import random

import pytest
from src.qkd.COW import pair_cow_protocols
from src.components.optical_channel import QuantumChannel, ClassicalChannel
from src.kernel.timeline import Timeline
from src.topology.node import QKDNode
from src.protocol import Protocol
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend to prevent Tkinter errors
import matplotlib.pyplot as plt
import networkx as nx

class Parent(Protocol):
    def __init__(self, own, length, name):
        super().__init__(own, name)
        self.upper_protocols = []
        self.lower_protocols = []
        self.length = length

    def received_message(self):
        pass

def clear_file_contents(file_name):
    with open(file_name, 'w') as file:
        file.write('')

@pytest.fixture(scope="module", autouse=True)
def clear_files():
    clear_file_contents('result.txt')


def draw_network_diagram(nodes, edges, title, file_name):
    # Create a directed graph
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # Position the nodes using the shell layout for better spacing
    pos = nx.shell_layout(G)

    # Set figure size to accommodate the graph
    plt.figure(figsize=(12, 8))

    # Draw the nodes
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='skyblue', alpha=0.6)

    # Draw the edges with arrows
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrowstyle='->', arrowsize=20)

    # Draw the node labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif")

    # Define and draw the edge labels
    edge_labels = {edge: 'Quantum Channel' for edge in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.5, font_size=9)

    # Set the title of the plot
    plt.title(title)

    # Remove the axes
    plt.axis('off')

    # Save the diagram to a file
    plt.savefig(file_name, format='PNG', bbox_inches='tight')
    plt.close()


def reset_qc_names(qc_alice_node1, qc_node1_node2, qc_node2_node3, qc_node3_node4, qc_node4_bob):
    qc_alice_node1.name = "qc_alice_node1"
    qc_node1_node2.name = "qc_node1_node2"
    qc_node2_node3.name = "qc_node2_node3"
    qc_node3_node4.name = "qc_node3_node4"
    qc_node4_bob.name = "qc_node4_bob"
    pass


@pytest.mark.parametrize("distance", range(1, 52, 5))
def test_cow_protocol(distance):
    clear_file_contents('round_details.txt')
    num_rounds = 1000
    num_of_bits = 200
    tl = Timeline(1e12)
    tl.seed(1)

    # Initialize nodes
    alice = QKDNode("Alice", tl, stack_size=3)
    node1 = QKDNode("Node1", tl, stack_size=3)
    node2 = QKDNode("Node2", tl, stack_size=3)
    node3 = QKDNode("Node3", tl, stack_size=3)
    node4 = QKDNode("Node4", tl, stack_size=3)
    bob = QKDNode("Bob", tl, stack_size=3)

    # Include the additional nodes in the diagram and topology
    nodes = [alice.name, node1.name, node2.name, node3.name, node4.name, bob.name]
    edges = [(alice.name, node1.name), (node1.name, node2.name), (node2.name, node3.name),
             (node3.name, node4.name), (node4.name, bob.name)]
    diagram_file_name = 'network_diagram.png'
    draw_network_diagram(nodes, edges, "Network Diagram for QKD System using COW Protocol", diagram_file_name)

    alice.set_seed(0)
    node1.set_seed(1)
    bob.set_seed(2)

    pair_cow_protocols(alice.protocol_stack[0], node1.protocol_stack[0])
    pair_cow_protocols(node1.protocol_stack[0], node2.protocol_stack[0])
    pair_cow_protocols(node2.protocol_stack[0], node3.protocol_stack[0])
    pair_cow_protocols(node3.protocol_stack[0], node4.protocol_stack[0])
    pair_cow_protocols(node4.protocol_stack[0], bob.protocol_stack[0])

    # Set up quantum channels in a ring topology
    qc_alice_node1 = QuantumChannel("qc_alice_node1", tl, distance=distance, attenuation=0.1, polarization_fidelity=0.8)
    qc_node1_node2 = QuantumChannel("qc_node1_node2", tl, distance=distance, attenuation=0.1, polarization_fidelity=0.8)
    qc_node2_node3 = QuantumChannel("qc_node2_node3", tl, distance=distance, attenuation=0.1, polarization_fidelity=0.8)
    qc_node3_node4 = QuantumChannel("qc_node3_node4", tl, distance=distance, attenuation=0.1, polarization_fidelity=0.8)
    qc_node4_bob = QuantumChannel("qc_node4_bob", tl, distance=distance, attenuation=0.1, polarization_fidelity=0.8)

    if random.random() < 0.75 and random.random() < 0.75:
        qc_alice_node1 = tl.get_entity_by_name("qc_alice_node1")
        qc_alice_node1.name = "qc_alice_qr_node1"
        qc_alice_node1.__setattr__("attenuation", 0.0)
    if random.random() < 0.75 and random.random() < 0.75:
        qc_node1_node2 = tl.get_entity_by_name("qc_node1_node2")
        qc_node1_node2.name = "qc_node1_qr_node2"
        qc_node1_node2.__setattr__("attenuation", 0.0)
    if random.random() < 0.75 and random.random() < 0.75:
        qc_node2_node3 = tl.get_entity_by_name("qc_node2_node3")
        qc_node2_node3.name = "qc_node2_qr_node3"
        qc_node2_node3.__setattr__("attenuation", 0.0)
    if random.random() < 0.75 and random.random() < 0.75:
        qc_node1_node2 = tl.get_entity_by_name("qc_node3_node4")
        qc_node3_node4.name = "qc_node3_qr_node4"
        qc_node3_node4.__setattr__("attenuation", 0.0)
    if random.random() < 0.75 and random.random() < 0.75:
        qc_node1_node2 = tl.get_entity_by_name("qc_node4_bob")
        qc_node4_bob.name = "qc_node4_qr_bob"
        qc_node4_bob.__setattr__("attenuation", 0.0)
    qc_alice_node1.set_ends(alice, node1.name)
    qc_node1_node2.set_ends(node1, node2.name)
    qc_node2_node3.set_ends(node2, node3.name)
    qc_node3_node4.set_ends(node3, node4.name)
    qc_node4_bob.set_ends(node4, bob.name)

    pa = Parent(alice, 128, "parent_Alice")
    pnode_1 = Parent(node1, 128, "parent_Node_1")
    pnode_2 = Parent(node2, 128, "parent_Node_2")
    pnode_3 = Parent(node3, 128, "parent_Node_3")
    pnode_4 = Parent(node4, 128, "parent_Node_4")

    alice.protocols[0].upper_protocols.append(pa)
    pa.lower_protocols.append(alice.protocol_stack[0])

    node1.protocols[0].upper_protocols.append(pnode_1)
    pnode_1.lower_protocols.append(node1.protocol_stack[0])

    node2.protocols[0].upper_protocols.append(pnode_2)
    pnode_2.lower_protocols.append(node2.protocol_stack[0])

    node3.protocols[0].upper_protocols.append(pnode_3)
    pnode_3.lower_protocols.append(node3.protocol_stack[0])

    node4.protocols[0].upper_protocols.append(pnode_4)
    pnode_4.lower_protocols.append(node4.protocol_stack[0])

    tl.init()
    alice.protocols[0].generate_sequences(num_of_bits, num_rounds)
    alice.protocols[0].send_pulse()

    for round in range(1, num_rounds + 1):
        print(f"Round {round} in progress")
        alice.protocols[0].push(1, round)
        tl.run()
        while not tl.events.isempty():
            tl.run()
        if "qr" in qc_alice_node1.name:
            print(f"Quantum Repeaters are attached between Alice and Node1 in {round}")
        print(f" Alice run done")
        node1.protocols[0].push(1, round)
        tl.run()
        while not tl.events.isempty():
            tl.run()
        if "qr" in qc_node1_node2.name:
            print(f"Quantum Repeaters are attached between Node1 and Node2 in {round}")
        print(f" Node1 run done")
        node2.protocols[0].push(1, round)
        tl.run()
        while not tl.events.isempty():
            tl.run()
        if "qr" in qc_node2_node3.name:
            print(f"Quantum Repeaters are attached between Node2 and Node3 in {round}")
        print(f" Node2 run done")

        node3.protocols[0].push(1, round)
        tl.run()
        while not tl.events.isempty():
            tl.run()
        if "qr" in qc_node3_node4.name:
            print(f"Quantum Repeaters are attached between Node3 and Node4 in {round}")
        print(f" Node3 run done")

        node4.protocols[0].push(1, round)
        tl.run()
        while not tl.events.isempty():
            tl.run()
        if "qr" in qc_node4_bob.name:
            print(f"Quantum Repeaters are attached between Node4 and Bob in {round}")
        print(f" Node4 run done")
        reset_qc_names(qc_alice_node1, qc_node1_node2, qc_node2_node3, qc_node3_node4, qc_node4_bob)

        node4.protocols[0].begin_classical_communication()

    node4.protocols[0].end_of_round(distance * 5, num_rounds)


# Example usage of the generated nodes
topology = 'grid'
matrix_size = 4  # Assuming a 4x4 grid for simplicity
adjacency_matrix = generate_adjacency_matrix(topology, matrix_size)
nodes = get_nodes(adjacency_matrix)

# Here we can continue with the logic of test_COW.py, using the nodes as needed
