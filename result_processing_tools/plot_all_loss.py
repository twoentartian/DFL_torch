import os
import pandas as pd
import matplotlib.pyplot as plt

# Create a large figure
plt.figure(figsize=(15, 10))

# Walk through the current directory and its subdirectories
for root, dirs, files in os.walk('.'):
    for file in files:
        if file == "loss.csv":
            file_path = os.path.join(root, file)

            # Read the CSV file
            df = pd.read_csv(file_path)

            # Plot "1" and "2" as y values and "tick" as x values
            plt.plot(df['tick'], df['1'], label=f"{root}_1", alpha=0.5)
            plt.plot(df['tick'], df['2'], label=f"{root}_2", alpha=0.5)

# Add labels and title
plt.xlabel('Tick')
plt.ylabel('Values')
plt.title('Loss Curves from Multiple Folders')
# plt.legend()
plt.grid(True)
plt.yscale("log")

# Show the plot
plt.savefig("all_loss.pdf")
plt.savefig("all_loss.png", dpi=400)