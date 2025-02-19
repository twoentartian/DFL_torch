import os
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # Create a large figure
    plt.figure(figsize=(15, 10))

    samples_per_df = 200

    high_loss_threshold = 1

    # Walk through the current directory and its subdirectories
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file == "loss.csv":
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                high_loss_count = (df['0'] > high_loss_threshold).sum()
                if high_loss_count > 0:
                    print(f"warning: in {file_path}, counter of loss>{high_loss_threshold} is {high_loss_count}")
                row = len(df.index)
                down_sample_rate = (row-1) // samples_per_df + 1
                df_down_sampled = df.iloc[::down_sample_rate, :]
                plt.plot(df_down_sampled['tick'], df_down_sampled['0'], label=f"{root}_1", alpha=0.5)

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
    plt.close()

