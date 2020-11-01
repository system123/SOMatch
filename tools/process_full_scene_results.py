from argparse import ArgumentParser
import scipy.io as sio
import numpy as np
from skimage.io import imread, imsave
import matplotlib.pyplot as plt

import os

from utils import Experiment
from utils.factory import *
from utils.helpers import load_file_list

def _len_full_scene(full_scene, patch_size, stride):
    ny, nx = full_scene.shape[:2]

    # Compensate for edges, and stride to get the number of centers
    ncy = (ny - 2*(patch_size//2))//stride
    ncx = (nx - 2*(patch_size//2))//stride

    return nx, ny, ncx, ncy

def make_heatmap(data, normalize=False):
    cmap = plt.cm.jet

    if normalize:
        norm = plt.Normalize(vmin=data.min(), vmax=data.max())
        data = norm(data)

    image = cmap(data)
    return image

def main(config):
    results = sio.loadmat(config.results_file)['y_pred'][:,1]
    full_scene = imread(config["datasets"]["test"].full_scene)
    img_list = load_file_list(config["datasets"]["test"].data_path)

    nx, ny, ncx, ncy = _len_full_scene(full_scene, config["datasets"]["test"].patch_size, config["datasets"]["test"].stride)
    map_size = ncx*ncy

    n_map = len(results)//map_size

    maps = [np.reshape(results[i*map_size:i*map_size+map_size], (ncy, ncx)) for i in range(n_map)]

    map_dir = os.path.join(config.result_path, 'maps')
    os.makedirs(map_dir, exist_ok=True)

    for i in range(n_map):
        image = make_heatmap(maps[i], True)
        coords = np.unravel_index(maps[i].argmax(), (ncx, ncy))
        print("Max Point {} {:.5f}".format(coords, maps[i][coords]))
        plt.imsave(os.path.join(map_dir, "{}.png".format(os.path.basename(img_list[i]))), image)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', default=None, type=str, required=True, help='config file path (default: None)')
    parser.add_argument('-r', '--results', default=None, type=str, required=True, help='Results file to process')
    args = parser.parse_args()

    config = Experiment.load_from_path(args.config)

    config.results_file = args.results

    assert config, "Config could not be loaded."

    main(config)
