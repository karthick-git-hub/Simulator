import pytest
from src.components.optical_channel import QuantumChannel, ClassicalChannel

from src.kernel.timeline import Timeline
from src.qkd.three_stage import pair_3stage_protocols
from src.protocol import Protocol
from src.topology.node import QKDNode


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

@pytest.mark.parametrize("distance", range(0, 1, 10))
def test_cow_protocol(distance):
    clear_file_contents('round_details_3stage.txt')
    num_rounds = 10
    num_of_bits = 10
    tl = Timeline(1e12)
    tl.seed(1)

    # Initialize nodes
    alice = QKDNode("Alice", tl, stack_size=4)
    bob = QKDNode("Bob", tl, stack_size=4)

    alice.set_seed(0)
    bob.set_seed(2)

    pair_3stage_protocols(alice.protocol_stack[0], bob.protocol_stack[0])
    pair_3stage_protocols(bob.protocol_stack[0], alice.protocol_stack[0])
    pair_3stage_protocols(alice.protocol_stack[0], bob.protocol_stack[0])

    qc_alice_bob = QuantumChannel("qc_alice_bob", tl, distance=distance, attenuation=0.1, polarization_fidelity=0.8)
    qc_bob_alice = QuantumChannel("qc_bob_alice", tl, distance=distance, attenuation=0.1, polarization_fidelity=0.8)
    qc_alice_bob_1 = QuantumChannel("qc_alice_bob_1", tl, distance=distance, attenuation=0.1, polarization_fidelity=0.8)

    qc_alice_bob.set_ends(alice, bob.name)
    qc_bob_alice.set_ends(bob, alice.name)
    qc_alice_bob_1.set_ends(alice, bob.name)

    # Parent
    pa = Parent(alice, 128, "alice")
    pb = Parent(bob, 128, "bob")
    alice.protocol_stack[0].upper_protocols.append(pa)
    pa.lower_protocols.append(alice.protocol_stack[0])
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
        print(f"push 1 done")
        bob.protocols[0].push(1, round)
        tl.run()
        while not tl.events.isempty():
            tl.run()

        print(f"push 2 done")
        alice.protocols[0].push(1, round)
        tl.run()
        while not tl.events.isempty():
            tl.run()


        print(f"push 3 done")
        alice.protocols[0].begin_classical_communication()

    alice.protocols[0].end_of_round(distance * 3, num_rounds)
