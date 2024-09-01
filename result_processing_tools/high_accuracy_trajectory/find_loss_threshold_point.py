import pandas as pd


if __name__ == "__main__":
    df = pd.read_csv('loss.csv', index_col='tick')
    df_weight_diff = pd.read_csv('weight_difference_l2.csv', index_col='tick')

    # Calculate the rolling mean of column "0"
    rolling_mean = df['0'].rolling(window=10).mean()

    # Find the ticks where the rolling mean exceeds 0.1
    exceed_ticks = rolling_mean[rolling_mean > 0.1].index

    first_exceed_tick = exceed_ticks[0]
    conv1_weight = df_weight_diff.loc[first_exceed_tick, 'conv1.weight']
    bn1_weight = df_weight_diff.loc[first_exceed_tick, 'bn1.weight']
    print(f'Tick: {first_exceed_tick}, Rolling Mean: {rolling_mean[first_exceed_tick]:.4f}, conv1.weight: {conv1_weight}, bn1.weight: {bn1_weight}')
