import os
import logging
from argparse import ArgumentParser

import pandas as pd

import torch
from torch.utils.data import DataLoader

from ignite.engine.engine import Engine, State, Events
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite._utils import convert_tensor

from utils import Experiment
from utils.factory import *
from utils.helpers import static_vars

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='')
logger = logging.getLogger()

def save_image(tensor, fname, cmap=plt.cm.jet):
    data = tensor.to("cpu").numpy().squeeze(0).squeeze(0)
    plt.imsave(fname, data, cmap=cmap)

def save_numpy(tensor, fname):
    data = tensor.to("cpu").numpy()
    np.save(fname, data)

def main(config, dataset="test"):
    assert validate_config(config), "ERROR: Config file is invalid. Please see log for details."

    logger.info("INFO: {}".format(config.toDict()))

    if config.device == "cpu" and torch.cuda.is_available():
        logger.warning("WARNING: Not using the GPU")

    if "cuda" in config.device:
        config.device = "cuda"

    assert dataset in config.datasets, "ERROR: Not test dataset is specified in the config. Don't know how to proceed."

    logger.info("INFO: Creating datasets and dataloaders...")

    config.datasets[dataset].update({'shuffle': False, 'augment': False, 'workers': 1})
    config.datasets[dataset].update({'batch_size': 1, "named": True, "return_all": True})

    # Create the training dataset
    dset_test = create_dataset(config.datasets[dataset])

    loader_test = get_data_loader(dset_test, config.datasets[dataset])

    logger.info("INFO: Running inference on {} samples".format(len(dset_test)))

    cp_paths = None
    last_epoch = 0
    checkpoint_dir = config.result_dir
    if 'checkpoint' in config:
        checkpoint_dir = config.checkpoint_dir if 'checkpoint_dir' in config else config.result_path
        cp_paths, last_epoch = config.get_checkpoints(path=checkpoint_dir, tag=config.checkpoint)
        print(f"Found checkpoint {cp_paths} for epoch {last_epoch}")

    if "DetA" in cp_paths:
        del cp_paths["DetA"]
        del cp_paths["DetB"]
    
    if "Match" in cp_paths:
        del cp_paths["Match"]

    models = {}
    for name, model in config.model.items():
        if name in ["DetA", "DetB", "Match"]:
            continue

        logger.info("INFO: Building the {} model".format(name))
        models[name] = build_model(model)

        # Load the checkpoint
        if name in cp_paths:
            models[name].load_state_dict( torch.load( cp_paths[name] ) )
            logger.info("INFO: Loaded model {} checkpoint {}".format(name, cp_paths[name]))

        models[name].to(config.device)
        print(models[name])

        if 'debug' in config and config.debug is True:
            print("*********** {} ************".format(name))
            for name, param in models[name].named_parameters():
                if param.requires_grad:
                    print(name, param.data)

    losses = {}
    for name, fcns in config.loss.items():
        losses[name] = []
        for l in fcns:
            losses[name].append( get_loss(l) )
            assert losses[name][-1], "Loss function {} for {} could not be found, please check your config".format(l, name)

    exp_logger = None
    if 'logger' in config:
        logger.info("INFO: Initialising the experiment logger")
        exp_logger = get_experiment_logger(config.result_path, config.logger)

    logger.info("INFO: Creating training manager and configuring callbacks")
    trainer = get_trainer(models, None, losses, None, config)

    evaluator_engine = Engine(trainer.evaluate)

    trainer.attach("test_loader", loader_test)
    trainer.attach("evaluation_engine", evaluator_engine)

    logger.info("INFO: Starting inference...")

    results = []

    save_path = os.path.join(config.checkpoint_dir, f"extracted_{last_epoch}", dataset)
    os.makedirs(save_path, exist_ok=True)

    with torch.no_grad():
        for i, (xs, ys, names)  in enumerate(tqdm(loader_test)):
            batch = (xs, ys)
            try:
                entity = {
                    "wkt": names["WKT"][0],
                    "city": names["city"][0],
                    "shift_x": names["p_match"][0].to("cpu").numpy()[0],
                    "shift_y": names["p_match"][1].to("cpu").numpy()[0]
                    }
                    
                filename = "{}_{}".format(names["city"], names["WKT"])
            except:
                entity = {
                    "season": names["season"][0], 
                    "scene": names["scene"][0], 
                    "patch": names["patch"][0],
                    "shift_x": names["p_match"][0].to("cpu").numpy()[0],
                    "shift_y": names["p_match"][1].to("cpu").numpy()[0]
                }

                filename = "{}_{}_{}".format(names["season"], names["scene"], names["patch"])

            imgs, hms, y, fts, dets = trainer.infer_batch(batch)
            (search_img, template_img, template_hard, search_hard) = imgs
            (heatmap_neg, heatmap_neg_raw, heatmap_hneg, heatmap_hneg_raw) = hms
            (y_a, y_b, y_bhn) = fts

            d_l2 = trainer.l2_shift_loss(heatmap_hneg, y[0], device="cuda")

            d_target = trainer.weighted_binary_cross_entropy(heatmap_hneg.detach(), y[0], device="cuda", reduction="none")
            d_target = torch.mean(d_target, dim=[1,2,3])
            d_target = -torch.log(d_target)

            # Save heatmaps
            save_numpy( heatmap_hneg_raw, os.path.join(save_path, f"{filename}_hm.npy") )
            save_numpy( search_img, os.path.join(save_path, f"{filename}_sar.npy") )
            save_numpy( search_hard, os.path.join(save_path, f"{filename}_sar_crop.npy") )
            save_numpy( template_img, os.path.join(save_path, f"{filename}_opt.npy") )
            save_numpy( template_hard, os.path.join(save_path, f"{filename}_opt_crop.npy") )
            save_numpy( y[0], os.path.join(save_path, f"{filename}_gt.npy") )

            entity.update({
                "l2": d_l2.to("cpu").numpy()[0],
                "nlog_match_loss": d_target.to("cpu").numpy()[0]              
                })
            
            results.append(entity)

            if i % 1000 == 0:
                df = pd.DataFrame.from_dict(results)
                df.to_csv(os.path.join(config.checkpoint_dir, "checkpoint_{}_extracted_dset_{}.csv".format(last_epoch, dataset)) , index=None)           

    df = pd.DataFrame.from_dict(results)
    df.to_csv(os.path.join(config.checkpoint_dir, "checkpoint_{}_extracted_dset_{}.csv".format(last_epoch, dataset)) , index=None)
    config.save()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', default=None, type=str, required=True, help='config file path (default: None)')
    parser.add_argument('--checkpoint', default=None, type=str, help='Checkpoint tag to reload')    
    parser.add_argument('--checkpoint_dir', default=None, type=str, help='Checkpoint directory to reload')
    parser.add_argument('--dataset', default="test", type=str, help="Which dataset to test on")
    args = parser.parse_args()

    OVERLOADABLE = ['checkpoint', 'epochs', 'checkpoint_dir', 'resume_from']

    overloaded = {}
    for k, v in vars(args).items():
        if (k in OVERLOADABLE) and (v is not None):
            overloaded[k] = v

    config = Experiment.load_from_path(args.config, overloaded)

    print(config.checkpoint)

    assert config, "Config could not be loaded."

    main(config, args.dataset)
