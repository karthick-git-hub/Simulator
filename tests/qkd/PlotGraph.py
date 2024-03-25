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

    # Plot for sifting percentage over distance
    plt.figure(figsize=(10, 5))
    plt.plot(distances, average_sifting_percentages, marker='o', linestyle='-', label='Average Sifting Percentage')
    plt.title('Average Sifting Percentage over Distance')
    plt.xlabel('Distance')
    plt.ylabel('Average Sifting Percentage')
    plt.grid(True)
    plt.legend()
    plt.savefig('sifting_percentage_over_distance.png')  # Save the plot as a PNG file
    plt.show()  # This might not display in non-interactive environments

    # Plot for key rate over distance
    plt.figure(figsize=(10, 5))
    plt.plot(distances, average_key_rates, marker='o', linestyle='-', color='green', label='Average Key Rate')
    plt.title('Average Key Rate over Distance')
    plt.xlabel('Distance')
    plt.ylabel('Average Key Rate')
    plt.grid(True)
    plt.legend()
    plt.savefig('key_rate_over_distance.png')  # Save the plot as a PNG file
    plt.show()  # This might not display in non-interactive environments


# Usage example
file_path = 'result.txt'  # Path to the result file
plot_graphs(file_path)
