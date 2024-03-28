import pytest
import numpy as np
from src.qkd.COW import pair_cow_protocols
from src.components.optical_channel import QuantumChannel, ClassicalChannel
from src.kernel.timeline import Timeline
from src.topology.node import QKDNode
from src.protocol import Protocol

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
    # Clear files before starting the test suite
    clear_file_contents('result.txt')

@pytest.mark.parametrize("distance", range(1, 9, 10))
def test_cow_protocol(distance):
    clear_file_contents('round_details.txt')
    num_rounds = 100
    num_of_bits = 100
    tl = Timeline(1e12)
    tl.seed(1)

    alice = QKDNode("Alice", tl)
    node1 = QKDNode("Node1", tl)
    bob = QKDNode("Bob", tl)

    alice.set_seed(0)
    node1.set_seed(1)
    bob.set_seed(2)

    # Pair protocols for communication
    pair_cow_protocols(alice.protocol_stack[0], node1.protocol_stack[0])
    pair_cow_protocols(node1.protocol_stack[0], bob.protocol_stack[0])

    # Set up quantum channels between the nodes
    qc_alice_node1 = QuantumChannel("qc_alice_node1", tl, distance=distance, attenuation=0.1, polarization_fidelity=0.1)
    qc_node1_bob = QuantumChannel("qc_node1_bob", tl, distance=distance, attenuation=0.1, polarization_fidelity=0.1)

    qc_alice_node1.set_ends(alice, node1.name)
    qc_node1_bob.set_ends(node1, bob.name)

    pa = Parent(alice, 128, "parent_Alice")
    pnode = Parent(node1, 128, "parent_Node")

    alice.protocols[0].upper_protocols.append(pa)
    pa.lower_protocols.append(alice.protocol_stack[0])

    node1.protocols[0].upper_protocols.append(pnode)
    pnode.lower_protocols.append(node1.protocol_stack[0])

    tl.init()
    alice.protocols[0].generate_sequences(num_of_bits, num_rounds)
    alice.protocols[0].send_pulse()

    for round in range(1, num_rounds + 1):
        print(f"Round {round} in progress")
        # Alice sends to Node1
        alice.protocols[0].push(1, round)
        tl.run()  # Run the timeline to process Alice's push to Node1
        # Wait until all events are processed before starting the next push
        while not tl.events.isempty():
            tl.run()
        # Now start Node1 to Bob communication
        node1.protocols[0].push(1, round)
        tl.run()  # Run the timeline to process Node1's push to Bob
        # Wait until all events are processed before ending the round
        while not tl.events.isempty():
            tl.run()
        node1.protocols[0].begin_classical_communication()

    node1.protocols[0].end_of_round(distance, num_rounds)
