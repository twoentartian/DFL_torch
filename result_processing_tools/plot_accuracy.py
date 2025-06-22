import os.path

import pandas
import matplotlib.pyplot as plt
import numpy

draw_only_first_node = False
enable_draw_every_tick = False
draw_every_tick = 500

__ignore_layer_list = ["num_batches_tracked", "running_mean", "running_var"]
# __ignore_layer_list = []


def is_ignored_layer(layer_name):
    output = False
    for i in __ignore_layer_list:
        if i in layer_name:
            output = True
            break
    return output


def calculate_herd_effect_delay(arg_accuracy_df: pandas.DataFrame):
    average_accuracy: pandas.Series = arg_accuracy_df.mean(axis=1)
    average_accuracy_diff = average_accuracy.diff()
    average_accuracy_diff.dropna(inplace=True)
    largest_diff = average_accuracy_diff.nlargest(10)
    largest_indexes = largest_diff.index
    for i in largest_indexes:
        if i > 20:
            return i


if __name__ == '__main__':
    accuracy_file_path = 'xxx'
    weight_diff_file_path = 'xxx'
    loss_file_path = 'xxx'
    training_loss_file_path = 'xxx'
    weight_change_file_path = 'xxx'

    accuracy_file_path = 'accuracy.csv'
    weight_diff_file_path = 'weight_difference_l2.csv'
    loss_file_path = 'loss.csv'
    training_loss_file_path = 'training_loss.csv'
    weight_change_file_path = "weight_change_l2.csv"

    # other_files_to_plot_candidate = {'0__distance_to_origin_l1.csv': ["conv1.weight", "bn1.weight"]}
    other_files_to_plot_candidate = {}

    num_of_plots = 0
    if os.path.exists(accuracy_file_path):
        draw_accuracy = True
        num_of_plots += 1
    else:
        draw_accuracy = False

    if os.path.exists(weight_diff_file_path):
        draw_weight_diff = True
        num_of_plots += 1
    else:
        draw_weight_diff = False

    if os.path.exists(loss_file_path):
        draw_loss = True
        num_of_plots += 1
    else:
        draw_loss = False

    if os.path.exists(training_loss_file_path):
        draw_training_loss = True
        num_of_plots += 1
    else:
        draw_training_loss = False

    if os.path.exists(weight_change_file_path):
        draw_weight_change = True
        num_of_plots += 1
    else:
        draw_weight_change = False

    other_files_to_plot = {}
    for other_file, col_to_plot in other_files_to_plot_candidate.items():
        if os.path.exists(other_file):
            other_files_to_plot[other_file] = col_to_plot
            num_of_plots += 1

    fig, axs = plt.subplots(num_of_plots, figsize=(10, 3*num_of_plots))
    plot_index = 0

    herd_effect_delay = None
    if draw_accuracy:
        accuracy_df = pandas.read_csv(accuracy_file_path, index_col=0, header=0)
        accuracy_df.drop(columns=["phase"], inplace=True)
        if enable_draw_every_tick:
            accuracy_df = accuracy_df[accuracy_df.index % draw_every_tick ==0]
        print(accuracy_df)
        accuracy_x = accuracy_df.index
        accuracy_df_len = len(accuracy_df)

        herd_effect_delay = calculate_herd_effect_delay(accuracy_df)

        ###################### accuracy
        axs[plot_index].axvline(x=herd_effect_delay, color='r', label=f'herd effect delay={herd_effect_delay}')
        for col in accuracy_df.columns:
            if draw_only_first_node:
                if col == "0":
                    axs[plot_index].plot(accuracy_x, accuracy_df[col], label=col, alpha=0.75)
            else:
                axs[plot_index].plot(accuracy_x, accuracy_df[col], label=col, alpha=0.75)

        axs[plot_index].grid()
        axs[plot_index].legend(ncol=5)
        axs[plot_index].set_title('accuracy')
        axs[plot_index].set_xlabel('time (tick)')
        axs[plot_index].set_ylabel('accuracy (0-1)')
        axs[plot_index].set_xlim([0, accuracy_df.index[accuracy_df_len-1]])
        if len(accuracy_df.columns) > 10:
            axs[plot_index].legend().remove()
        plot_index = plot_index + 1

    if draw_loss:
        loss_df = pandas.read_csv(loss_file_path, index_col=0, header=0)
        if enable_draw_every_tick:
            loss_df = loss_df[loss_df.index % draw_every_tick ==0]
        print(loss_df)
        loss_x = loss_df.index
        loss_df_len = len(loss_df)
        for col in loss_df.columns:
            axs[plot_index].plot(loss_x, loss_df[col], label=col, alpha=0.75)
        axs[plot_index].grid()
        axs[plot_index].legend(ncol=5)
        axs[plot_index].set_title('test loss')
        axs[plot_index].set_xlabel('time (tick)')
        axs[plot_index].set_ylabel('test loss')
        axs[plot_index].set_xlim([0, loss_df.index[loss_df_len - 1]])
        if len(loss_df.columns) > 10:
            axs[plot_index].legend().remove()
        plot_index = plot_index + 1

    if draw_training_loss:
        training_loss_df = pandas.read_csv(training_loss_file_path, index_col=0, header=0)
        if enable_draw_every_tick:
            training_loss_df = training_loss_df[training_loss_df.index % draw_every_tick ==0]
        print(training_loss_df)
        training_loss_x = training_loss_df.index
        training_loss_df_len = len(training_loss_df)
        for col in training_loss_df.columns:
            axs[plot_index].plot(training_loss_x, training_loss_df[col], label=col, alpha=0.75)
        axs[plot_index].grid()
        axs[plot_index].legend(ncol=5)
        axs[plot_index].set_title('training loss')
        axs[plot_index].set_xlabel('time (tick)')
        axs[plot_index].set_ylabel('training loss')
        axs[plot_index].set_xlim([0, training_loss_df.index[training_loss_df_len - 1]])
        if len(training_loss_df.columns) > 10:
            axs[plot_index].legend().remove()
        plot_index = plot_index + 1

    if draw_weight_diff:
        weight_diff_df = pandas.read_csv(weight_diff_file_path, index_col=0, header=0)
        if enable_draw_every_tick:
            weight_diff_df = weight_diff_df[weight_diff_df.index % draw_every_tick ==0]
        print(weight_diff_df)
        weight_diff_x = weight_diff_df.index
        weight_diff_df_len = len(weight_diff_df)

        if herd_effect_delay is not None:
            axs[plot_index].axvline(x=herd_effect_delay, color='r', label=f'herd effect delay={herd_effect_delay}')
        for col in weight_diff_df.columns:
            if numpy.sum(weight_diff_df[col]) == 0:
                continue
            if is_ignored_layer(col):
                continue
            axs[plot_index].plot(weight_diff_x, weight_diff_df[col], label=col)

        axs[plot_index].grid()
        axs[plot_index].legend()
        axs[plot_index].set_title('model weight diff')
        axs[plot_index].set_xlabel('time (tick)')
        axs[plot_index].set_ylabel('weight diff')
        axs[plot_index].set_yscale('log')
        axs[plot_index].set_xlim([0, weight_diff_df.index[weight_diff_df_len-1]])
        if len(weight_diff_df.columns) > 10:
            axs[plot_index].legend().remove()
        plot_index = plot_index + 1

    if draw_weight_change:
        weight_change_df = pandas.read_csv(weight_change_file_path, index_col=0, header=0)
        if enable_draw_every_tick:
            weight_change_df = weight_change_df[weight_change_df.index % draw_every_tick == 0]
        print(weight_change_df)
        weight_change_x = weight_change_df.index
        weight_change_df_len = len(weight_change_df)

        if herd_effect_delay is not None:
            axs[plot_index].axvline(x=herd_effect_delay, color='r', label=f'herd effect delay={herd_effect_delay}')
        for col in weight_change_df.columns:
            if numpy.sum(weight_change_df[col]) == 0:
                continue
            if is_ignored_layer(col):
                continue
            axs[plot_index].plot(weight_change_x, weight_change_df[col], label=col)

        axs[plot_index].grid()
        axs[plot_index].legend()
        axs[plot_index].set_title('weight change - L2 distance per tick')
        axs[plot_index].set_xlabel('time (tick)')
        axs[plot_index].set_ylabel('weight cahnge')
        axs[plot_index].set_yscale('log')
        axs[plot_index].set_xlim([0, weight_change_df.index[weight_change_df_len-1]])
        if len(weight_change_df.columns) > 10:
            axs[plot_index].legend().remove()
        plot_index = plot_index + 1

    for other_file, col_to_plot in other_files_to_plot.items():
        df = pandas.read_csv(other_file, index_col=0, header=0)
        print(df)
        df_x = df.index
        df_len = len(df)

        if herd_effect_delay is not None:
            axs[plot_index].axvline(x=herd_effect_delay, color='r', label=f'herd effect delay={herd_effect_delay}')
        for col in col_to_plot:
            axs[plot_index].plot(df_x, df[col], label=col)

        axs[plot_index].grid()
        axs[plot_index].legend()
        axs[plot_index].set_title(f'{other_file}')
        axs[plot_index].set_xlabel('time (tick)')
        axs[plot_index].set_ylabel('value')
        axs[plot_index].set_xlim([0, df.index[df_len-1]])
        if len(df.columns) > 10:
            axs[plot_index].legend().remove()
        plot_index = plot_index + 1

    plt.tight_layout()
    plt.savefig('accuracy_weight_diff_combine.pdf')
    plt.savefig('accuracy_weight_diff_combine.jpg', dpi=400)