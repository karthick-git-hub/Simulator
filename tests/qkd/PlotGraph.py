import matplotlib.pyplot as plt
import json


def plot_graphs(file_path):
    # Open and read the file
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Extracting data from the JSON object
    distances = sorted(data.keys(), key=int)
    average_sifting_percentages = [data[distance]['average_sifting_percentage'] for distance in distances]
    average_key_rates = [data[distance]['average_key_rate'] for distance in distances]
    qber = [(100 - sp) / 100 for sp in average_sifting_percentages]

    # Plot for sifting percentage over distance
    plt.figure(figsize=(10, 5))
    plt.plot(distances, average_sifting_percentages, marker='o', linestyle='-', label='Average Sifting Percentage')
    plt.title('Sifting Success Rate over Distance for Alice and Bob')
    plt.xlabel('Distance A-B')
    plt.ylabel('Sifting Success Rate')
    plt.grid(True)
    plt.ylim(0, 100)
    plt.legend()
    plt.savefig('sifting_percentage_over_distance.png')  # Save the plot as a PNG file
    plt.show()  # This might not display in non-interactive environments

    # Plot for QBER over distance
    plt.figure(figsize=(10, 5))
    plt.plot(distances, qber, marker='o', linestyle='-', label='QBER Graph')
    plt.title('QBER over Distance for Alice and Bob')
    plt.xlabel('Distance A-B')
    plt.ylabel('QBER (%)')
    plt.grid(True)
    plt.ylim(0, 1)  # QBER ranges from 0 to 1
    plt.legend()
    plt.savefig('qber_percentage_over_distance.png')
    plt.show()

    # Plot for key rate over distance
    plt.figure(figsize=(10, 5))
    plt.plot(distances, average_key_rates, marker='o', linestyle='-', color='green', label='Average Key Rate')
    plt.title('Average Key Rate over Distance')
    plt.xlabel('Distance')
    plt.ylabel('Average Key Rate')
    plt.grid(True)
    plt.ylim(0, 0.7)
    plt.legend()
    plt.savefig('key_rate_over_distance.png')  # Save the plot as a PNG file
    plt.show()  # This might not display in non-interactive environments


# Usage example
file_path = 'C:\\Karthick\\Temp\\test\\tests\\qkd\\Images\\100_bits_100_rounds\\3-stage\\0.05-attenuation\\result_3stage.txt'  # Path to the result file
plot_graphs(file_path)
