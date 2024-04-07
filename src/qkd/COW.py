import random
from copy import deepcopy
from typing import TYPE_CHECKING
from datetime import datetime, timedelta
import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from ..protocol import StackProtocol

if TYPE_CHECKING:
    from ..topology.node import QKDNode

import json

def pair_cow_protocols(sender: "COWProtocol", receiver: "COWProtocol") -> None:
    sender.another = receiver
    receiver.another = sender
    sender.role = 0
    receiver.role = 1

class COWProtocol(StackProtocol):
    decoy_sequence = ['decoy', 'decoy', 'decoy']
    alice_bits_with_vacuum = []
    decoy_indices = []
    actual_alice_bits = []
    sifting_percentage = []
    key_rate = 0
    total_rounds = 0
    round_details = {}
    round_number = 0
    channel_list = []
    detector_line_mapping = {
        'detector1': 'dataline',
        'detector2': 'DM1',
        'detector3': 'DM2'
    }

    # Initialize data structures to store information for each line
    detection_data = {
        'dataline': [],
        'DM1': [],
        'DM2': []
    }

    def __init__(self, own: "QKDNode", name, lightsource, qsdetector):
        super().__init__(own, name)
        self.lightsource = lightsource
        self.qsdetector = qsdetector
        self.bit_sequence = []
        self.decoy_indices = []
        self.alice_bits_with_vacuum = []
        self.message_counter = 0  # Counter for received messages

        self.qc_0 = QuantumCircuit(1)  # Circuit for 0 degree
        self.qc_180 = QuantumCircuit(1)  # Circuit for 180 degree
        self.qc_180.x(0)  # Applying an X gate for 180 degree


    def generate_sequences(self, bit_length, total_rounds, decoy_rate=0.1):
        COWProtocol.total_rounds = total_rounds
        self.bit_sequence = np.random.choice([0, 1], bit_length)
        num_decoys = int(bit_length * decoy_rate)
        total_length = bit_length + num_decoys
        self.decoy_indices = np.random.choice(range(total_length), num_decoys, replace=False)

    def send_pulse(self):
        current_time = datetime.now()
        for bit_value in self.bit_sequence:
            encoded_sequence = self.encode_bit(bit_value)
            timestamp = current_time.strftime("%d-%b-%Y %H:%M:%S.%f")[:-3]
            self.alice_bits_with_vacuum.append((encoded_sequence, timestamp))
            current_time += timedelta(milliseconds=1)
        self.insert_decoy_pulses()

    def insert_decoy_pulses(self):
        total_length = len(self.bit_sequence) + len(self.decoy_indices)
        new_alice_bits_with_vacuum = []
        original_bit_index = 0
        current_time = datetime.now()

        for i in range(total_length):
            timestamp = current_time.strftime('%d-%b-%Y %H:%M:%S.%f')[:-3]
            if i in self.decoy_indices:
                new_alice_bits_with_vacuum.append((COWProtocol.decoy_sequence, timestamp))
            else:
                encoded_sequence = self.encode_bit(self.bit_sequence[original_bit_index])
                new_alice_bits_with_vacuum.append((encoded_sequence, timestamp))
                original_bit_index += 1
            current_time += timedelta(milliseconds=1)
        self.alice_bits_with_vacuum = new_alice_bits_with_vacuum
        COWProtocol.alice_bits_with_vacuum = self.alice_bits_with_vacuum

    def encode_bit(self, bit):
        VACUUM = 'decoy'
        if bit == 0:
            return [self.qc_0, VACUUM, self.qc_0]
        elif bit == 1:
            return [self.qc_180, VACUUM, self.qc_180]

    def push(self, key_num, rounds, run_time=np.inf):
        self.message_counter = 0
        COWProtocol.actual_alice_bits = []
        self.own.destination = self.another.own.name
        COWProtocol.detection_data = {'dataline': [], 'DM1': [], 'DM2': []}
        self.attach_to_detector()
        lightsource = self.own.components[self.lightsource]
        alice_bits_with_vacuum_copy = deepcopy(COWProtocol.alice_bits_with_vacuum)
        if len(COWProtocol.channel_list) != 0:
            alice_bits_with_vacuum_copy = deepcopy(COWProtocol.channel_list)
        print(f"Total with vacuum: {len(alice_bits_with_vacuum_copy)} alice_bits_with_vacuum --  {alice_bits_with_vacuum_copy}  round -- {rounds}")
        COWProtocol.channel_list = []
        lightsource.custom_emit(alice_bits_with_vacuum_copy)
        for bit in alice_bits_with_vacuum_copy:
            if bit[0] != COWProtocol.decoy_sequence:
                COWProtocol.actual_alice_bits.append((self.measure(bit[0]), bit[1]))


    def pop(self, detector_index: int, time: int):
        # Implement the logic to process detection events
        pass

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

    def receive_node_messages(self, info):
        COWProtocol.channel_list.append((info['photon'], info['time']))
        print(f"length {len(COWProtocol.channel_list)} COWProtocol.channel_list {COWProtocol.channel_list}")

    def received_message(self, info):
        # Handle the detection time update from the detector
        detection_time = info['time']
        detector = info['detector']
        photon = info['photon']
        if photon != COWProtocol.decoy_sequence:
            if not (detector == 'detector2' and photon == None):
                self.message_counter += 1

        # Determine the line based on the detector
        line = COWProtocol.detector_line_mapping.get(detector)
        if line:
            # Store the detection information in the respective line's data
            COWProtocol.detection_data[line].append({
                'time': detection_time,
                'photon': self.measure(photon),
                'detector': detector
            })


    def measure(self, photon):
        if (len(photon) == 3 and photon[1] == 'decoy'):
            bit_0 = self.calculateBitValue(photon[0])
            bit_1 = self.calculateBitValue(photon[2])
            if bit_0 == bit_1:
                return bit_0
            else:
                if self.get_generator().random() < 0.5:
                    return bit_0
                else:
                    return bit_1

    def calculateBitValue(self, photon):
        if isinstance(photon, QuantumCircuit):
            photon.measure_all()
            simulator = Aer.get_backend('qasm_simulator')
            job = execute(photon, simulator, shots=1)
            result = job.result()
            counts = result.get_counts(photon)
            measured_bit = list(counts.keys())[0][-1]  # Get the last character of the result key
            return int(measured_bit)  # Convert to integer (0 or 1)

    def begin_classical_communication(self):
        self.sifting_process(COWProtocol.detection_data)

    def sifting_process(self, detection_data):
        COWProtocol.sifting_percentage = []
        dataline_entries = detection_data['dataline']
        security_percentage = 0.0
        # Remove duplicates based on a specific key in each dictionary
        unique_entries = []
        unique_keys = set()
        for entry in dataline_entries:
            key = entry['time']  # Assuming 'timestamp' is the key used for comparison
            if key not in unique_keys:
                unique_keys.add(key)
                unique_entries.append(entry)
        # Sort the unique entries based on the timestamp
        sorted_dataline_entries = sorted(unique_entries, key=lambda x: x['time'])
        print(f"length -- {len(sorted_dataline_entries)} dataline_entries {sorted_dataline_entries} ")
        raw_bob_bits = [(entry['photon'], entry['time']) for entry in sorted_dataline_entries]
        raw_key_bob = sorted(raw_bob_bits, key=lambda x: x[1])

        # Create a set of Bob's timestamps for quick lookup
        bob_timestamps = {bit[1] for bit in raw_key_bob}

        # Filter Alice's bits to include only those with timestamps matching Bob's
        raw_key_alice = [(bit, timestamp) for bit, timestamp in COWProtocol.actual_alice_bits if
                              timestamp in bob_timestamps]

        # Continue with the sifting process...
        random_bits = self.generate_random_bits(len(raw_key_bob))
        print("\nRandom bits for sifting:", random_bits)
        sameFlagValue = 0
        differentFlagValue = 0
        if len(random_bits) > 0:
            # Perform the sifting process
            for index in random_bits:
                if index < len(raw_key_bob):
                    bob_bit, bob_timestamp = raw_key_bob[index]
                    # Find the corresponding Alice bit using the timestamp
                    alice_bit = next((bit for bit, time in raw_key_alice if time == bob_timestamp), None)
                    print(f"Alice's bit: {alice_bit}, Bob's bit: {bob_bit} at {bob_timestamp}")
                    if alice_bit == bob_bit:
                        sameFlagValue += 1
                    else:
                        differentFlagValue += 1

            print(f"Same: {sameFlagValue}, Different: {differentFlagValue}")
            security_percentage  = (sameFlagValue / len(random_bits)) * 100 if sameFlagValue > 0 else 0
            print(f"security_percentage: {security_percentage}")

        COWProtocol.sifting_percentage.append(security_percentage)
        self.discard_bits_post_sifting(random_bits, raw_key_alice, raw_key_bob)


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
        COWProtocol.key_rate = 0
        COWProtocol.round_number += 1
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
        COWProtocol.key_rate = (self.count_elements(matched_bob_bits) /(len(COWProtocol.actual_alice_bits) + 0.1 * len(COWProtocol.actual_alice_bits)))  if len(COWProtocol.actual_alice_bits) > 0 else 0

        # COWProtocol.key_rate = (self.count_elements(matched_bob_bits) / (len(COWProtocol.actual_alice_bits) + 0.1 * len(COWProtocol.actual_alice_bits))) if len(COWProtocol.actual_alice_bits) > 0 else 0
        print(f"Explicit parity check success rate: {COWProtocol.key_rate}%")
        print(f"Length: {len(matched_alice_bits)} Matched Alice bits: {matched_alice_bits}")
        print(f"Length: {len(matched_bob_bits)} Matched Bob bits:  {matched_bob_bits}")
        self.print_details()


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


    def count_elements(self, nested_list):
        count = 0
        for element in nested_list:
            if isinstance(element, list):
                count += self.count_elements(element)
            else:
                count += 1
        return count


    def print_details(self, file_name='round_details.txt'):
        round_number = COWProtocol.round_number
        details = {
            'sifting_percentage': COWProtocol.sifting_percentage[-1],
            'key_rate': COWProtocol.key_rate,
            'time': datetime.now().strftime("%d-%b-%Y %H:%M:%S.%f")[:-3]
        }

        COWProtocol.round_details[round_number] = details

        with open(file_name, 'a') as file:
            file.write(f"Round {round_number}: ")
            file.write(json.dumps(details) + '\n')

    def end_of_round(self, distance, num_rounds, attenuation, file_name='round_details.txt', output_file='result.json'):
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

        # Load the existing results and update or add the new entry for the given distance and attenuation
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
