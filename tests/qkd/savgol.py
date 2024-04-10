import pandas as pd
import json
from scipy.signal import savgol_filter

# Path to the JSON-formatted text file
file_path = 'Images/100_bits_100_rounds/cow/0.1-attenuation/result.txt'

# Load data from the text file
with open(file_path, 'r') as file:
    data = json.load(file)

# Convert the loaded data into a DataFrame
df = pd.DataFrame.from_dict(data, orient='index')

# Apply Savitzky-Golay filter
window_size = 5  # Window size for Savitzky-Golay filter, must be odd
poly_order = 2  # Polynomial order for Savitzky-Golay filter

# Ensure the window size is less than the number of data points and is an odd number
if window_size > len(df):
    window_size = len(df) // 2 * 2 + 1  # Make sure it's odd
if window_size <= poly_order:
    poly_order = window_size - 1  # Adjust poly_order if necessary

df['sg_average_sifting_percentage'] = savgol_filter(df['average_sifting_percentage'], window_size, poly_order)
df['sg_average_key_rate'] = savgol_filter(df['average_key_rate'], window_size, poly_order)

# Apply Running Average
df['ra_average_sifting_percentage'] = df['average_sifting_percentage'].rolling(window=3, min_periods=1).mean()
df['ra_average_key_rate'] = df['average_key_rate'].rolling(window=3, min_periods=1).mean()

# Prepare the processed data for JSON output
processed_data = {
    key: {
        'sg_average_sifting_percentage': row['sg_average_sifting_percentage'],
        'sg_average_key_rate': row['sg_average_key_rate'],
        'ra_average_sifting_percentage': row['ra_average_sifting_percentage'],
        'ra_average_key_rate': row['ra_average_key_rate']
    } for key, row in df.iterrows()
}

# Save the processed data to a JSON file
output_file_path = 'Images/100_bits_100_rounds/cow/0.1-attenuation/processed_3stage.json'
with open(output_file_path, 'w') as file:
    json.dump(processed_data, file, indent=4)

print(f"Processed data saved to: {output_file_path}")
