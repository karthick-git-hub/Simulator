import json
import random
from copy import deepcopy
from datetime import datetime, timedelta

import numpy as np
from qiskit import Aer, execute
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate

from src.protocol import StackProtocol


def pair_3stage_protocols(sender: "ThreeStageProtocol", receiver: "ThreeStageProtocol") -> None:
    sender.another = receiver
    receiver.another = sender
    sender.role = 0
    receiver.role = 1


class ThreeStageProtocol(StackProtocol):
    U_A = None
    U_B = None
    key_rate = 0
    total_rounds = 0
    alice_bits = []
    alice_bits_encoded = []
    channel_list = []
    detection_data = []
    iterations_passed = 0
    bits_bob_received = []
    sifting_percentage = []
    round_number = 0
    round_details = {}

    def __init__(self, own: "QKDNode", name, lightsource, qsdetector):
        super().__init__(own, name)
        self.lightsource = lightsource
        self.qsdetector = qsdetector
        self.alice_bits = []
        self.alice_bits_encoded = []
        self.qc_0 = QuantumCircuit(1)  # Circuit for 0 degree
        self.qc_180 = QuantumCircuit(1)  # Circuit for 180 degree
        self.qc_180.x(0)  # Applying an X gate for 180 degree

    def generate_sequences(self, bit_length, total_rounds):
        ThreeStageProtocol.total_rounds = total_rounds
        self.alice_bits = np.random.choice([0, 1], bit_length)
        ThreeStageProtocol.U_A = self.generate_unitary()
        ThreeStageProtocol.U_B = self.generate_unitary()

    def send_pulse(self):
        current_time = datetime.now()
        for bit_value in self.alice_bits:
            timestamp = current_time.strftime('%d-%b-%Y %H:%M:%S.%f')[:-3]
            encoded_sequence = self.encode_bit(bit_value)
            self.alice_bits_encoded.append((encoded_sequence, timestamp))
            ThreeStageProtocol.alice_bits.append((bit_value, timestamp))
            current_time += timedelta(milliseconds=1)
        ThreeStageProtocol.alice_bits_encoded = self.alice_bits_encoded

    def encode_bit(self, bit):
        if bit == 0:
            return self.qc_0
        elif bit == 1:
            return self.qc_180

    def generate_unitary(self):
        """Generate a 2x2 unitary diagonal matrix."""
        angle = np.random.rand() * 2 * np.pi
        return np.array([[np.exp(1j * angle), 0],
                         [0, np.exp(-1j * angle)]])

    def is_unitary(self, m):
        """Check if the matrix is unitary."""
        return np.allclose(np.eye(m.shape[0]), m @ m.conj().T)

    def pop(self, detector_index: int, time: int):
        # Implement the logic to process detection events
        pass

    def push(self, key_num, rounds, run_time=np.inf):
        new_alice_bits_with_vacuum = []
        ThreeStageProtocol.iterations_passed += 1
        self.own.destination = self.another.own.name
        self.attach_to_detector()
        if ThreeStageProtocol.iterations_passed == 1:
            alice_bits_encoded_copy = deepcopy(ThreeStageProtocol.alice_bits_encoded)
            U = ThreeStageProtocol.U_A
        else:
            alice_bits_encoded_copy = deepcopy(sorted(ThreeStageProtocol.detection_data, key=lambda x: x[1]))
            U = ThreeStageProtocol.U_B
            if ThreeStageProtocol.iterations_passed == 3:
                U = np.conjugate(ThreeStageProtocol.U_A).T
        print(f"length -- {len(alice_bits_encoded_copy)} alice_bits_encoded_copy- {alice_bits_encoded_copy}")
        ThreeStageProtocol.detection_data = []
        for i, alice_bit in enumerate(alice_bits_encoded_copy):
            original_bit, timestamp = alice_bit
            circuit_copy = deepcopy(original_bit)
            circuit_copy.append(UnitaryGate(U, label=f'U_transmission_{i}'), [0])
            new_alice_bits_with_vacuum.append((circuit_copy, timestamp))
        alice_bits_encoded_copy = new_alice_bits_with_vacuum
        lightsource = self.own.components[self.lightsource]
        lightsource.custom_emit(alice_bits_encoded_copy)

    def attach_to_detector(self):
        """Attach this protocol instance as an observer to a detector."""
        if self.qsdetector in self.own.components:
            detector = self.own.components[self.qsdetector]
            if hasattr(detector, 'attach_observer'):
                detector.attach_observer(self)
            else:
                raise AttributeError(f"Detector {self.qsdetector} does not have an attach_observer method")
        else:
            raise AttributeError(f"Detector {self.qsdetector} not found in QKDNode components")

    def received_message(self, info):
        ThreeStageProtocol.detection_data.append(info['photon'])

    def decoding(self):
        self.decoding_bits(ThreeStageProtocol.detection_data)
        ThreeStageProtocol.iterations_passed = 0

    def decoding_bits(self, detection_data):
        ThreeStageProtocol.bits_bob_received = []
        for i, detection_data_bit in enumerate(detection_data):
            original_circuit, timestamp = detection_data_bit
            circuit_copy = deepcopy(original_circuit)
            U = np.conjugate(ThreeStageProtocol.U_B).T
            circuit_copy.append(UnitaryGate(U, label=f'U_transmission_{i}'), [0])
            ThreeStageProtocol.bits_bob_received.append((self.measure(circuit_copy), timestamp))
        ThreeStageProtocol.bits_bob_received = sorted(ThreeStageProtocol.bits_bob_received, key=lambda x: x[1])
        print(
            f"length -- {len(ThreeStageProtocol.bits_bob_received)}  bits_bob_received {ThreeStageProtocol.bits_bob_received} \n, "
            f"length -- {len(ThreeStageProtocol.alice_bits)}  ThreeStageProtocol.alice_bits {ThreeStageProtocol.alice_bits}")

    def measure(self, photon):
        bit_0 = self.calculateBitValue(photon)
        return bit_0

    def calculateBitValue(self, photon):
        if isinstance(photon, QuantumCircuit):
            photon.measure_all()
            simulator = Aer.get_backend('qasm_simulator')
            job = execute(photon, simulator, shots=1)
            result = job.result()
            counts = result.get_counts(photon)
            measured_bit = list(counts.keys())[0][-1]  # Get the last character of the result key
            return int(measured_bit)  # Convert to integer (0 or 1)

    def generate_random_bits(self, length):
        self.random_bits = 0
        if not isinstance(length, int):
            raise TypeError("Length must be an integer")
        if length <= 0:
            return []
        self.random_bits = max(1, round(length * 0.10))  # Calculate 10% of the length, ensuring at least 1 segment
        print(f"generating random bits {self.random_bits} {length}")
        random_bits = random.sample(range(length), self.random_bits)  # Generate unique random indices
        return sorted(random_bits)  # Return the sorted list of random_bits

    def begin_classical_communication(self):
        ThreeStageProtocol.sifting_percentage = []
        security_percentage = 0.0

        sorted_alice_entries = sorted(ThreeStageProtocol.alice_bits, key=lambda x: x[1])
        sorted_bob_entries = sorted(ThreeStageProtocol.bits_bob_received, key=lambda x: x[1])
        print(f"length -- {len(sorted_alice_entries)} sorted_alice_entries {sorted_alice_entries} \n "
              f"length -- {len(sorted_bob_entries)} sorted_bob_entries {sorted_bob_entries}")
        # Create a set of Bob's timestamps for quick lookup
        bob_timestamps = {bit[1] for bit in sorted_bob_entries}

        # Filter Alice's bits to include only those with timestamps matching Bob's
        raw_key_alice = [(bit, timestamp) for bit, timestamp in sorted_alice_entries if
                         timestamp in bob_timestamps]

        # Continue with the sifting process...
        random_bits = self.generate_random_bits(len(sorted_bob_entries))
        print("\nRandom bits for sifting:", random_bits)
        sameFlagValue = 0
        differentFlagValue = 0
        if len(random_bits) > 0:
            # Perform the sifting process
            for index in random_bits:
                if index < len(sorted_bob_entries):
                    bob_bit, bob_timestamp = sorted_bob_entries[index]
                    # Find the corresponding Alice bit using the timestamp
                    alice_bit = next((bit for bit, time in raw_key_alice if time == bob_timestamp), None)
                    print(f"Alice's bit: {alice_bit}, Bob's bit: {bob_bit} at {bob_timestamp}")
                    if alice_bit == bob_bit:
                        sameFlagValue += 1
                    else:
                        differentFlagValue += 1

            print(f"Same: {sameFlagValue}, Different: {differentFlagValue}")
            security_percentage = (sameFlagValue / len(random_bits)) * 100 if sameFlagValue > 0 else 0
            print(f"security_percentage: {security_percentage}")

        ThreeStageProtocol.sifting_percentage.append(security_percentage)
        self.discard_bits_post_sifting(random_bits, raw_key_alice, sorted_bob_entries)

    def discard_bits_post_sifting(self, random_bits, raw_key_alice, raw_key_bob):
        sorted_random_bits = sorted(random_bits, reverse=True)
        print("\nBefore discarding")
        print("\n length -- ", len(raw_key_alice), "self.raw_key_alice", raw_key_alice)
        print("\n length -- ", len(raw_key_bob), "self.raw_key_bob", raw_key_bob)
        for index in sorted_random_bits:
            if index < len(raw_key_alice):  # Check to avoid index out of range
                raw_key_alice.pop(index)

        for index in sorted_random_bits:
            if index < len(raw_key_bob):  # Check to avoid index out of range
                raw_key_bob.pop(index)

        print("\nAfter discarding")
        print("\n length -- ", len(raw_key_alice), "self.raw_key_alice", raw_key_alice)
        print("\n length -- ", len(raw_key_bob), "self.raw_key_bob", raw_key_bob)
        self.parity_check(raw_key_alice, raw_key_bob)

    def parity_check(self, raw_key_alice, raw_key_bob):
        alice_parity_list = []
        bob_parity_list = []
        matched_alice_bits = []
        matched_bob_bits = []
        ThreeStageProtocol.key_rate = 0
        ThreeStageProtocol.round_number += 1
        print(f"raw_key_alice {raw_key_alice} ")
        print(f"raw_key_bob {raw_key_bob} ")

        def calculate_parity(bits):
            return "even" if sum(bits) % 2 == 0 else "odd"

        # Calculate parity for Alice's bits in blocks of 3
        for i in range(0, len(raw_key_alice), 3):
            alice_bits = [bit for bit, _ in raw_key_alice[i:i + 3]]
            alice_parity = calculate_parity(alice_bits.copy())
            alice_parity_list.append(alice_parity)
        # Calculate parity for Bob's bits in blocks of 3
        for i in range(0, len(raw_key_bob), 3):
            bob_bits = [bit for bit, _ in raw_key_bob[i:i + 3]]
            bob_parity = calculate_parity(bob_bits.copy())
            bob_parity_list.append(bob_parity)

        # Initialize match count
        matches = 0
        # Compare the parity lists directly and count matches
        for i, (a_parity, b_parity) in enumerate(zip(alice_parity_list, bob_parity_list)):
            if a_parity == b_parity:
                matches += 1
                # Calculate start and end indices for actual bits (without padding)
                start_idx = i * 3
                end_idx = start_idx + 3
                # Store the actual bits, excluding the runtime appended 0s
                matched_alice_bits.append(
                    [raw_key_alice[j][0] for j in range(start_idx, min(end_idx, len(raw_key_alice)))])
                matched_bob_bits.append(
                    [raw_key_bob[j][0] for j in range(start_idx, min(end_idx, len(raw_key_bob)))])
        print("\n Matches: ", matches)
        ThreeStageProtocol.key_rate = (self.count_elements(matched_bob_bits) / (
                    len(ThreeStageProtocol.alice_bits_encoded) + 0.1 * len(
                ThreeStageProtocol.alice_bits_encoded))) if len(ThreeStageProtocol.alice_bits_encoded) > 0 else 0

        print(f"Explicit parity check success rate: {ThreeStageProtocol.key_rate}%")
        print(f"Length: {len(matched_alice_bits)} Matched Alice bits: {matched_alice_bits}")
        print(f"Length: {len(matched_bob_bits)} Matched Bob bits:  {matched_bob_bits}")
        self.print_details()

    def count_elements(self, nested_list):
        count = 0
        for element in nested_list:
            if isinstance(element, list):
                count += self.count_elements(element)
            else:
                count += 1
        return count

    def print_details(self, file_name='round_details_3stage.txt'):
        round_number = ThreeStageProtocol.round_number
        details = {
            'sifting_percentage': ThreeStageProtocol.sifting_percentage[-1],
            'key_rate': ThreeStageProtocol.key_rate,
            'time': datetime.now().strftime("%d-%b-%Y %H:%M:%S.%f")[:-3]
        }

        ThreeStageProtocol.round_details[round_number] = details

        with open(file_name, 'a') as file:
            file.write(f"Round {round_number}: ")
            file.write(json.dumps(details) + '\n')

    def end_of_round(self, distance, num_rounds, attenuation, file_name='round_details_3stage.txt', output_file='result_3stage.txt'):
        total_sifting_percentage = 0
        total_key_rate = 0
        # Read and process each line in the input file
        with open(file_name, 'r') as file:
            for line in file:
                try:
                    json_part = line.split(': ', 1)[1]
                    detail_dict = json.loads(json_part)
                    total_sifting_percentage += detail_dict['sifting_percentage']
                    total_key_rate += detail_dict['key_rate']
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")

        # Calculate averages
        if num_rounds > 0:
            average_sifting_percentage = total_sifting_percentage / num_rounds
            average_key_rate = total_key_rate / num_rounds
        else:
            average_sifting_percentage = 0
            average_key_rate = 0

        # Load the existing results and update or add the new entry for the given distance
        try:
            with open(output_file, 'r') as outfile:
                existing_results = json.load(outfile)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_results = {}

        # If the attenuation key doesn't exist, create it
        if str(attenuation) not in existing_results:
            existing_results[str(attenuation)] = {}

        existing_results[str(attenuation)][str(distance)] = {
            'average_sifting_percentage': average_sifting_percentage,
            'average_key_rate': average_key_rate
        }

        # Write the updated results back to the file
        with open(output_file, 'w') as outfile:
            json.dump(existing_results, outfile, indent=4)

        print(f"Average Sifting Percentage: {average_sifting_percentage} for attenuation {attenuation}")
        print(f"Average Key Rate: {average_key_rate} for attenuation {attenuation}")
