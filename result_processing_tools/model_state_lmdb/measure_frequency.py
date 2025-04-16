import io

import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import os

import lmdb
import argparse
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

window_size = 100
step_size = 50
fps = 5

def measure_frequency_lmdb(db_path, output_path):
    device = torch.device('cpu')

    # Open the LMDB environment
    env = lmdb.open(db_path, readonly=True)

    with env.begin() as txn:
        # Get the total number of entries
        total_entries = txn.stat()['entries']
        print(f"Total number of models: {total_entries}")
        cursor = txn.cursor()
        keys = sorted(txn.cursor().iternext(values=False))
        all_ticks = []
        record_node_name = None
        for key in keys:
            items = key.decode("utf-8").split("/")
            node_name = items[0]
            tick = items[1].replace(".model.pt", "")
            if record_node_name is None:
                record_node_name = node_name
            else:
                assert record_node_name == node_name, f"node name changes: {record_node_name} -> {node_name}"
            all_ticks.append(int(tick))
        all_ticks = sorted(all_ticks)
        print(f"all ticks: {all_ticks}")

        amplitude_per_layer = {}

        for tick_index, tick in enumerate(all_ticks):
            key = f"{record_node_name}/{tick}.model.pt"
            print(f"Loading model: {key}")
            key_b = key.encode("utf-8")
            value = cursor.get(key_b)
            buffer = io.BytesIO(value)
            state_dict = torch.load(buffer, map_location=device)
            for layer_name, layer_weights in state_dict.items():
                if layer_name not in amplitude_per_layer:
                    amplitude_per_layer[layer_name] = np.zeros(len(all_ticks))
                amplitude_per_layer[layer_name][tick_index] = torch.norm(layer_weights)

    for layer_name, amplitudes in amplitude_per_layer.items():
        layer_output_dir = os.path.join(output_path, f"fft_{layer_name}")
        os.makedirs(layer_output_dir, exist_ok=True)
        all_files = []
        for start in range(0, len(amplitudes) - window_size + 1, step_size):
            window = amplitudes[start:start + window_size]
            freqs = np.fft.rfftfreq(window_size, d=1.0)  # Change `d=1.0` to your sample spacing
            spectrum = np.abs(np.fft.rfft(window))

            plt.figure()
            plt.plot(freqs, spectrum)
            plt.title(f"FFT Window {start}-{start + window_size}")
            plt.xlabel("Frequency")
            plt.ylabel("Amplitude")
            plt.ylim([0, 0.1])
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{layer_output_dir}/{start}.jpg")
            plt.close()

            all_files.append(f"{start}.jpg")

        print(f"generating video for layer {layer_name}")

        first_img = cv2.imread(f"{layer_output_dir}/{all_files[0]}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width, channel = first_img.shape
        video = cv2.VideoWriter(f'{output_path}/fft_{layer_name}.mp4', fourcc, fps, (width, height))
        for single_frame in all_files:
            img = cv2.imread(f"{layer_output_dir}/{single_frame}")
            video.write(img)
        video.release()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="measure the model test accuracy in a LMDB file")
    parser.add_argument("db_path", help="Path to the LMDB database")
    args = parser.parse_args()

    db_path = args.db_path
    output_path = os.path.dirname(db_path)
    measure_frequency_lmdb(db_path, output_path)
