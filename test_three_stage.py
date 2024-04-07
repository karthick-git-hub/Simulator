
import pytest
from src.components.optical_channel import QuantumChannel, ClassicalChannel

from src.kernel.timeline import Timeline
from src.qkd.three_stage import pair_3stage_protocols
from src.protocol import Protocol
from src.topology.node import QKDNode

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
    clear_file_contents('result_3stage.txt')


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


@pytest.mark.parametrize("distance", range(1, 52, 5))
def test_3stage_protocol(distance):
    clear_file_contents('round_details_3stage.txt')
    num_rounds = 200
    num_of_bits = 200
    tl = Timeline(1e12)
    tl.seed(1)

    # Initialize nodes
    alice = QKDNode("Alice", tl, stack_size=4)
    node1 = QKDNode("Node1", tl, stack_size=4)
    bob = QKDNode("Bob", tl, stack_size=4)

    alice.set_seed(0)
    node1.set_seed(1)
    bob.set_seed(2)

    pair_3stage_protocols(alice.protocol_stack[0], node1.protocol_stack[0])
    pair_3stage_protocols(node1.protocol_stack[0], bob.protocol_stack[0])
    pair_3stage_protocols(bob.protocol_stack[0], node1.protocol_stack[0])
    pair_3stage_protocols(node1.protocol_stack[0], alice.protocol_stack[0])
    pair_3stage_protocols(alice.protocol_stack[0], node1.protocol_stack[0])
    pair_3stage_protocols(node1.protocol_stack[0], bob.protocol_stack[0])

    qc_alice_node1 = QuantumChannel("qc_alice_node1", tl, distance=distance, attenuation=0.1, polarization_fidelity=0.8)
    qc_node1_bob = QuantumChannel("qc_node1_bob", tl, distance=distance, attenuation=0.1, polarization_fidelity=0.8)
    qc_bob_node1 = QuantumChannel("qc_bob_node1", tl, distance=distance, attenuation=0.1, polarization_fidelity=0.8)
    qc_node1_alice = QuantumChannel("qc_node1_alice", tl, distance=distance, attenuation=0.1, polarization_fidelity=0.8)
    qc_alice_node1_1 = QuantumChannel("qc_alice_node1_1", tl, distance=distance, attenuation=0.1, polarization_fidelity=0.8)
    qc_node1_bob_1 = QuantumChannel("qc_node1_bob_1", tl, distance=distance, attenuation=0.1, polarization_fidelity=0.8)

    qc_alice_node1.set_ends(alice, node1.name)
    qc_node1_bob.set_ends(node1, bob.name)
    qc_bob_node1.set_ends(bob, node1.name)
    qc_node1_alice.set_ends(node1, alice.name)
    qc_alice_node1_1.set_ends(alice, node1.name)
    qc_node1_bob_1.set_ends(node1, bob.name)

    # Parent
    pa = Parent(alice, 128, "alice")
    pnode1 = Parent(node1, 128, "node1")
    pb = Parent(bob, 128, "bob")
    alice.protocol_stack[0].upper_protocols.append(pa)
    pa.lower_protocols.append(alice.protocol_stack[0])
    node1.protocol_stack[0].upper_protocols.append(pnode1)
    pnode1.lower_protocols.append(node1.protocol_stack[0])
    bob.protocol_stack[0].upper_protocols.append(pb)
    pb.lower_protocols.append(bob.protocol_stack[0])

    tl.init()
    alice.protocols[0].generate_sequences(num_of_bits, num_rounds)
    alice.protocols[0].send_pulse()

    for round in range(1, num_rounds + 1):
        print(f"Round {round} in progress")
        alice.protocols[0].push(1, round)
        tl.run()
        while not tl.events.isempty():
            tl.run()
        print(f"Alice done")

        node1.protocols[0].push(1, round)
        tl.run()
        while not tl.events.isempty():
            tl.run()
        print(f"Node 1 done")

        bob.protocols[0].push(1, round)
        tl.run()
        while not tl.events.isempty():
            tl.run()

        print(f"Bob done")

        node1.protocols[0].push(1, round)
        tl.run()
        while not tl.events.isempty():
            tl.run()
        print(f"Node 1 done 2nd time")

        alice.protocols[0].push(1, round)
        tl.run()
        while not tl.events.isempty():
            tl.run()
        print(f"Alice done 2nd time")

        node1.protocols[0].push(1, round)
        tl.run()
        while not tl.events.isempty():
            tl.run()
        print(f"Node 1 done 3rd time")

        bob.protocols[0].push(1, round)
        tl.run()
        while not tl.events.isempty():
            tl.run()

        print(f"Bob done 2nd time")

        alice.protocols[0].decoding()
        alice.protocols[0].begin_classical_communication()

    alice.protocols[0].end_of_round(distance * 6, num_rounds)
