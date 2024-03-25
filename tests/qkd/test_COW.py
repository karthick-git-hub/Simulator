import numpy as np
from sequence.qkd.COW import pair_cow_protocols
from sequence.components.optical_channel import QuantumChannel, ClassicalChannel
from sequence.kernel.timeline import Timeline
from sequence.topology.node import QKDNode
from sequence.protocol import Protocol

class Parent(Protocol):
    def __init__(self, own, length, name):
        super().__init__(own, name)
        self.upper_protocols = []
        self.lower_protocols = []
        self.length = length

    def received_message(self):
        pass

def test_cow_protocol():
    num_rounds = 10  # or however many rounds you want to simulate
    num_of_bits = 10  # or however many rounds you want to simulate
    distance = 100
    # Create a timeline for the simulation
    tl = Timeline(1e12)  # 1 second simulation
    tl.seed(1)

    # Create QKD nodes for Alice and Bob
    alice = QKDNode("Alice", tl, stack_size=3)
    bob = QKDNode("Bob", tl, stack_size=3)

    alice.set_seed(0)
    bob.set_seed(1)
    clear_file_contents()

    pair_cow_protocols(alice.protocol_stack[0], bob.protocol_stack[0])

    # Create quantum and classical channels
    qc = QuantumChannel("qc", tl, attenuation=0.10, distance=distance)
    cc = ClassicalChannel("cc", tl, distance=10)

    # Set the ends of the channels
    qc.set_ends(alice, bob.name)
    cc.set_ends(alice, bob.name)

    # Set parent protocols to manage the QKD process
    pa = Parent(alice, 128, "parent_Alice")

    # Append parent protocols as upper protocols
    alice.protocols[0].upper_protocols.append(pa)
    pa.lower_protocols.append(alice.protocol_stack[0])

    # Start the simulation
    tl.init()
    alice.protocols[0].generate_sequences(num_of_bits, num_rounds)
    alice.protocols[0].send_pulse()
    for round in range(num_rounds):
        print(f"Round in progress - {round}")
        alice.protocols[0].push(1, round)  # Push some parameters for the COW protocol to start
        tl.run()
    alice.protocols[0].end_of_round(distance)

def clear_file_contents(file_name='round_details.txt'):
    with open(file_name, 'w') as file:
        file.write('')  # Writing an empty string will clear the file

    # Here, you can check the results of the protocol, such as key generation rates, error rates, etc.

if __name__ == "__main__":
    test_cow_protocol()
