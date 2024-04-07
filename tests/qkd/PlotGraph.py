import matplotlib.pyplot as plt
import json


def plot_graphs(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    plt.figure(figsize=(10, 5))
    for attenuation, results in data.items():
        distances = sorted(results.keys(), key=float)
        average_sifting_percentages = [results[distance]['average_sifting_percentage'] for distance in distances]
        plt.plot(distances, average_sifting_percentages, marker='o', linestyle='-', label=f'Attenuation {attenuation}')

    plt.title('Sifting Success Rate over Distance')
    plt.xlabel('Distance A-B')
    plt.ylabel('Sifting Success Rate (%)')
    plt.grid(True)
    plt.ylim(0, 100)
    plt.legend()
    plt.savefig('sifting_percentage_over_distance.png')
    plt.show()

    plt.figure(figsize=(10, 5))
    for attenuation, results in data.items():
        distances = sorted(results.keys(), key=float)
        average_sifting_percentages = [results[distance]['average_sifting_percentage'] for distance in distances]
        qber = [(100 - sp) / 100 for sp in average_sifting_percentages]
        plt.plot(distances, qber, marker='o', linestyle='-', label=f'Attenuation {attenuation}')

    plt.title('QBER over Distance')
    plt.xlabel('Distance A-B')
    plt.ylabel('QBER (%)')
    plt.grid(True)
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig('qber_percentage_over_distance.png')
    plt.show()

    plt.figure(figsize=(10, 5))
    for attenuation, results in data.items():
        distances = sorted(results.keys(), key=float)
        average_key_rates = [results[distance]['average_key_rate'] for distance in distances]
        plt.plot(distances, average_key_rates, marker='o', linestyle='-', label=f'Attenuation {attenuation}')

    plt.title('Average Key Rate over Distance')
    plt.xlabel('Distance')
    plt.ylabel('Average Key Rate')
    plt.grid(True)
    plt.legend()
    plt.savefig('key_rate_over_distance.png')
    plt.show()


# Usage example
file_path = 'result.json'  # Path to the result file
plot_graphs(file_path)
