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

@pytest.mark.parametrize("distance", range(21, 29, 10))
def test_cow_protocol(distance):
    num_rounds = 100
    num_of_bits = 100
    tl = Timeline(1e12)
    tl.seed(1)

    alice = QKDNode("Alice", tl, stack_size=3)
    bob = QKDNode("Bob", tl, stack_size=3)

    alice.set_seed(0)
    bob.set_seed(1)
    clear_file_contents('round_details.txt')

    pair_cow_protocols(alice.protocol_stack[0], bob.protocol_stack[0])

    qc = QuantumChannel("qc", tl, attenuation=0.10, distance=distance, polarization_fidelity=0.1)
    cc = ClassicalChannel("cc", tl, distance=distance)

    qc.set_ends(alice, bob.name)
    cc.set_ends(alice, bob.name)

    pa = Parent(alice, 128, "parent_Alice")

    alice.protocols[0].upper_protocols.append(pa)
    pa.lower_protocols.append(alice.protocol_stack[0])

    tl.init()
    alice.protocols[0].generate_sequences(num_of_bits, num_rounds)
    alice.protocols[0].send_pulse()
    for round in range(1, num_rounds + 1):
        print(f"Round in progress - {round}")
        alice.protocols[0].push(1, round)
        tl.run()
    alice.protocols[0].end_of_round(distance, num_rounds)
