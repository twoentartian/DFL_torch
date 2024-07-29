import os
import pandas as pd
import matplotlib.pyplot as plt

samples_per_df = 200

# Function to create and save a plot for a specific column
def plot_column(data, column_name):
    plt.figure(figsize=(10, 6))

    for folder, df in data.items():
        row = len(df.index)
        down_sample_rate = (row - 1) // samples_per_df + 1
        df_down_sampled = df.iloc[::down_sample_rate, :]
        plt.plot(df_down_sampled['tick'], df_down_sampled[column_name], label=f"{folder}")

    plt.xlabel('Tick')
    plt.ylabel(column_name)
    plt.title(f'{column_name} Curves from Multiple Folders')
    plt.grid(True)

    # Save the figure
    plt.savefig(f'diff_{column_name}.png', dpi=400)
    plt.savefig(f'diff_{column_name}.pdf')
    plt.close()


# Dictionary to store dataframes from each folder
data = {}

# Walk through the current directory and its subdirectories
for root, dirs, files in os.walk('..'):
    for file in files:
        if file == "weight_difference_l2.csv":
            file_path = os.path.join(root, file)

            # Read the CSV file
            df = pd.read_csv(file_path)

            # Store the dataframe in the dictionary with folder name as key
            folder_name = os.path.basename(root)
            data[folder_name] = df

# Determine columns to plot based on the sum of their values
columns_to_plot = set()
for folder, df in data.items():
    for column in df.columns:
        if column != 'tick' and df[column].sum() != 0:
            columns_to_plot.add(column)
print(columns_to_plot)

# Create and save plots for each column with non-zero sum
for column in columns_to_plot:
    plot_column(data, column)
