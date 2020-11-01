import os
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import argparse
import glob
import shutil
from tqdm import tqdm

from skimage import io

def get_image_collection(src_dir, ext=None):
    if ext is None:
        ext = ""
    else:
        ext = ".{}".format(ext)

    files = io.ImageCollection(os.path.join(src_dir, "*{}".format(ext)), conserve_memory=False)

    return(files)

def make_dataset_lists(df, dest, ltype, no_negs=False):
    # parent, _ = os.path.split(dest.rstrip(os.sep))
    parent = dest
    la_name = os.path.join(parent, "list.{}.opt.txt".format(ltype))
    lb_name = os.path.join(parent, "list.{}.sar.txt".format(ltype))

    f_a = df['files_a'].values.tolist()
    f_b = df['files_b'].values.tolist()
    f_c = df['files_c'].values.tolist()

    # Shuffle the file lists to make the dataset more diverse
    # f_a_shuf, f_c = f_a, f_c
#    f_mirror, f_orig = shuffle(f_mirror, f_orig)

    if no_negs:
        zip_a = zip(f_a)
        zip_b = zip(f_b)
    else:
        zip_a = zip(f_a, f_a)
        zip_b = zip(f_b, f_c)

    # Doing it this way ensures the lists stay balanced, 1 pos + 1 neg ...
    # Otherwise if we created the list and then shuffled the dataset could become unbalanced
    list_a = [val for pair in zip_a for val in pair]
    list_b = [val for pair in zip_b for val in pair]

    with open(la_name, "w") as f:
        list_a = map(lambda x: x + '\n', list_a)
        f.writelines(list_a)

    with open(lb_name, "w") as f:
        list_b = map(lambda x: x + '\n', list_b)
        f.writelines(list_b)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src_a", help="The optical dataset")
    parser.add_argument("src_b", help="The sar dataset")
    parser.add_argument("dest", help="Base directory to save file lists to")
    parser.add_argument("--src_c", default=None, help="Negative dataset if not created randomly")
    parser.add_argument("--type", default="train", help="Addtional file list identifier")
    parser.add_argument("--ext", default="png", help="File extension")
    parser.add_argument("--no_negs", action="store_true", help="Don't create negative pairs, just match the files as they are in the folders")
    args = parser.parse_args()

    files_a = get_image_collection(args.src_a, ext=args.ext).files
    files_b = get_image_collection(args.src_b, ext=args.ext).files
    print(len(files_b))
    df = pd.DataFrame.from_dict({'files_a': files_a})
    df['files_b'] = files_b

    if args.src_c:
        files_c = get_image_collection(args.src_c, ext=args.ext).files
        print(len(files_c))
        df['files_c'] = files_c
    else:
        df['files_c'] = shuffle(files_b)

    print("# matching negative items {}".format(len(df.loc[df['files_b'] == df['files_c']])))

    make_dataset_lists(df, args.dest, args.type, args.no_negs)
