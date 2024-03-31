import json
from copy import deepcopy
from datetime import datetime, timedelta

import numpy as np
from qiskit import Aer, execute
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate

from protocol import StackProtocol


def pair_3stage_protocols(sender: "ThreeStageProtocol", receiver: "ThreeStageProtocol") -> None:
    sender.another = receiver
    receiver.another = sender
    sender.role = 0
    receiver.role = 1


class ThreeStageProtocol(StackProtocol):
    key_rate = 0
    total_rounds = 0
    alice_bits = []
    alice_bits_encoded = []
    channel_list = []
    detection_data = []
    iterations_passed = 0
    U_A = None
    U_B = None

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
        # Handle the detection time update from the detector
        print(f"inside received_message {info}")
        ThreeStageProtocol.detection_data.append(info['photon'])


    def begin_classical_communication(self):
        self.sifting_process(ThreeStageProtocol.detection_data)

    def sifting_process(self, detection_data):
        print(f" inside sifting_process")
        bits_bob_received = []
        for i, detection_data_bit in enumerate(detection_data):
            original_circuit, timestamp = detection_data_bit
            circuit_copy = deepcopy(original_circuit)
            U = np.conjugate(ThreeStageProtocol.U_B).T
            circuit_copy.append(UnitaryGate(U, label=f'U_transmission_{i}'), [0])
            bits_bob_received.append((self.measure(circuit_copy), timestamp))
        bits_bob_received = sorted(bits_bob_received, key=lambda x: x[1])
        print(f"bits_bob_received {bits_bob_received} , ThreeStageProtocol.alice_bits {ThreeStageProtocol.alice_bits}")


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

    def end_of_round(self, distance, num_rounds, file_name='round_details_3stage.txt', output_file='result_3stage.txt'):
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

        existing_results[str(distance)] = {
            'average_sifting_percentage': average_sifting_percentage,
            'average_key_rate': average_key_rate
        }

        # Write the updated results back to the file
        with open(output_file, 'w') as outfile:
            json.dump(existing_results, outfile, indent=4)

        print(f"Average Sifting Percentage: {average_sifting_percentage}")
        print(f"Average Key Rate: {average_key_rate}")