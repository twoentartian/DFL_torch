import argparse
import pickle
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='input plt3d file')

    args = parser.parse_args()
    file = args.file
    fig = pickle.load(open(file, 'rb'))
    plt.show()
