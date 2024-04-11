import pandas as pd


# Assuming your CSV file is named 'your_file.csv'
file_path = 'ml-100k-csv/u_data.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Sort the DataFrame by 'user_id' and then 'item_id'
df_sorted = df.sort_values(by=['item_id', 'user_id'])

# Save the sorted DataFrame back to a CSV file
df_sorted.to_csv('ud_data_item_sorted.csv', index=False)