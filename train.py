import os
import sys
import logging
import signal
import random
from argparse import ArgumentParser

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchsummary import summary

from ignite.engine.engine import Engine, State, Events
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite._utils import convert_tensor

from utils import Experiment
from utils.factory import *

logging.basicConfig(level=logging.INFO, format='')
logger = logging.getLogger()

torch.backends.cudnn.benchmark = True

def main(config):
    assert validate_config(config), "ERROR: Config file is invalid. Please see log for details."

    logger.info("INFO: {}".format(config.toDict()))

    # Set the random number generator seed for torch, as we use their dataloaders this will ensure shuffle is constant
    # Remeber to seed custom datasets etc with the same seed
    if config.seed > 0:
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(config.seed)
        torch.manual_seed(config.seed)
        random.seed(config.seed)
        np.random.seed(config.seed)

    if config.device == "cpu" and torch.cuda.is_available():
        logger.warning("WARNING: Not using the GPU")
    elif config.device == "cuda":
        config.device = f"cuda:{config.device_ids[0]}"

    config.nsave = config.nsave if "nsave" in config else 5

    logger.info("INFO: Creating datasets and dataloaders...")
    # Create the training dataset
    dset_train = create_dataset(config.datasets.train)

    # If the validation config has a parameter called split then we ask the training dset for the validation dataset
    # it should be noted that you shouldn't shuffle the dataset in the init of the train dataset if this is the case
    # as only on get_validation_split will we know how to split the data. Unless shuffling is deterministic.
    train_ids = None
    if 'validation' in config.datasets:
        # Ensure we have a full config for validation, this means we don't need t specify everything in the config file
        # only the differences
        config_val = config.datasets.train.copy()
        config_val.update(config.datasets.validation)

        dset_val = create_dataset(config_val)

        loader_val = get_data_loader(dset_val, config_val)
        print("Using validation dataset of {} samples or {} batches".format(len(dset_val), len(loader_val)))
    elif 'includes_validation' in config.datasets.train:
        train_ids, val_ids = dset_train.get_validation_split(config_val)
        loader_val = get_data_loader(dset_train, config.datasets.train, val_ids)
        print("Using validation dataset of {} samples or {} batches".format(len(val_ids), len(loader_val)))
    else:
        logger.warning("WARNING: No validation dataset was specified")
        dset_val = None
        loader_val = None

    loader_train = get_data_loader(dset_train, config.datasets.train, train_ids)
    dset_len = len(train_ids) if train_ids is not None else len(dset_train)
    print("Using training dataset of {} samples or {} batches".format(dset_len, len(loader_train)))

    cp_paths = None
    last_epoch = 0
    if 'checkpoint' in config:
        checkpoint_dir = config.checkpoint_dir if 'checkpoint_dir' in config else config.result_path
        cp_paths, last_epoch = config.get_checkpoints(path=checkpoint_dir, tag=config.checkpoint)
        print("Found checkpoint {} for Epoch {}".format(config.checkpoint, last_epoch))
        last_epoch = last_epoch if config.resume_from == -1 else config.resume_from
        # config.epochs = config.epochs - last_epoch if last_epoch else config.epochs

    models = {}
    for name, model in config.model.items():
        logger.info("INFO: Building the {} model".format(name))
        models[name] = build_model(model)

        # Load the checkpoint
        if name in cp_paths:
            models[name].load_state_dict( torch.load( cp_paths[name] ) )
            logger.info("INFO: Loaded model {} checkpoint {}".format(name, cp_paths[name]))

        if len(config.device_ids) > 1:
            models[name] = nn.DataParallel(models[name], device_ids=config.device_ids)

        models[name].to(config.device)
        print(models[name])

        if 'debug' in config and config.debug is True:
            print("*********** {} ************".format(name))
            for name, param in models[name].named_parameters():
                if param.requires_grad:
                    print(name, param.data)

    optimizers = {}
    for name, conf in config.optimizer.items():
        optim_conf = conf.copy()
        del optim_conf["models"]

        model_params = []
        for model_id in conf.models:
            model_params.extend( list(filter(lambda p: p.requires_grad, models[model_id].parameters())) )
        
        logger.info("INFO: Using {} Optimization for {}".format(list(optim_conf.keys())[0], name))
        optimizers[name] = get_optimizer(model_params, optim_conf)

        # Restoring the optimizer breaks because we do not include all parameters in the optimizer state. So if we aren't continuing training then just make a new optimizer
        if name in cp_paths and 'checkpoint_dir' not in config:
            optimizers[name].load_state_dict( torch.load( cp_paths[name] ) )
            logger.info("INFO: Loaded {} optimizer checkpoint {}".format(name, cp_paths[name]))

            for state in optimizers[name].state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(config.device)

    losses = {}
    for name, fcns in config.loss.items():
        losses[name] = []
        for l in fcns:
            losses[name].append( get_loss(l) )
            assert losses[name][-1], "Loss function {} for {} could not be found, please check your config".format(l, name)

    if 'logger' in config:
        logger.info("INFO: Initialising the experiment logger")
        exp_logger = get_experiment_logger(config.result_path, config.logger)
        if last_epoch > 0:
            exp_logger.fast_forward(last_epoch, len(loader_train))

    logger.info("INFO: Creating training manager and configuring callbacks")
    trainer = get_trainer(models, optimizers, losses, exp_logger, config)

    trainer_engine = Engine(trainer.train)
    evaluator_engine = Engine(trainer.evaluate)

    trainer.attach("train_loader", loader_train)
    trainer.attach("validation_loader", loader_val)
    trainer.attach("evaluation_engine", evaluator_engine)
    trainer.attach("train_engine", trainer_engine)

    for phase in config.metrics.keys():
        if phase == "train": engine = trainer_engine
        if phase == "validation": engine = evaluator_engine

        for name, metric in config.metrics[phase].items():
            metric = get_metric(metric)
            if metric is not None:
                metric.attach(engine, name)
            else:
                logger.warning("WARNING: Metric {} could not be created for {} phase".format(name, phase))

    # Register default callbacks to run the validation stage
    if loader_val is not None:
        if len(loader_train) > 2000:
            # Validate 4 times an epoch
            num_batch = len(loader_train)//4

            def validate_run(engine):
                if engine.state.iteration % num_batch == 0:
                    evaluator_engine.run(loader_val)

            trainer_engine.add_event_handler(Events.ITERATION_COMPLETED, validate_run)
        else:
            trainer_engine.add_event_handler(Events.EPOCH_COMPLETED, lambda engine: evaluator_engine.run(loader_val))

    # Initialise the Epoch from the checkpoint - this is a hack because Ignite is dumb
    if last_epoch > 0:
        def set_epoch(engine, last_epoch):
            engine.state.epoch = last_epoch

        trainer_engine.add_event_handler(Events.STARTED, set_epoch, last_epoch)


    schedulers = {"batch": {}, "epoch": {}}
    if 'scheduler' in config:
        for sched_name, sched in config.scheduler.items():
            if sched_name in optimizers:
                logger.info("INFO: Setting up LR scheduler for {}".format(sched_name))
                sched_fn, sched_scheme = get_lr_scheduler(optimizers[sched_name], sched)
                assert sched_fn, "Learning Rate scheduler for {} could not be found, please check your config".format(sched_name)
                assert sched_scheme in ["batch", "epoch"], "ERROR: Invalid scheduler scheme, must be either epoch or batch"

                schedulers[sched_scheme][sched_name] = sched_fn

        def epoch_scheduler(engine):
            for name, sched in schedulers["epoch"].items():
                sched.step()

        def batch_scheduler(engine):
            for name, sched in schedulers["batch"].items():
                sched.step()

        trainer_engine.add_event_handler(Events.ITERATION_COMPLETED, lambda engine: batch_scheduler(engine))
        trainer_engine.add_event_handler(Events.EPOCH_COMPLETED, lambda engine: epoch_scheduler(engine))

    if exp_logger is not None:
        trainer_engine.add_event_handler(Events.ITERATION_COMPLETED, exp_logger.log_iteration, phase="train", models=models, optims=optimizers)
        trainer_engine.add_event_handler(Events.EPOCH_COMPLETED, exp_logger.log_epoch, phase="train", models=models, optims=optimizers)
        evaluator_engine.add_event_handler(Events.ITERATION_COMPLETED, exp_logger.log_iteration, phase="evaluate", models=models, optims=optimizers)
        evaluator_engine.add_event_handler(Events.EPOCH_COMPLETED, exp_logger.log_epoch, phase="evaluate", models=models, optims=optimizers)

    if "monitor" in config and config.monitor.early_stopping:
        logger.info("INFO: Enabling early stopping, monitoring {}".format(config.monitor.score))
        score_fn = lambda e: config.monitor.scale * e.state.metrics[config.monitor.score]
        es_handler = EarlyStopping(patience=config.monitor.patience, score_function=score_fn, trainer=trainer_engine)
        evaluator_engine.add_event_handler(Events.COMPLETED, es_handler)

    if "monitor" in config and config.monitor.save_score:
        logger.info("INFO: Saving best model based on {}".format(config.monitor.save_score))
        score_fn = lambda e: config.monitor.save_scale * e.state.metrics[config.monitor.save_score]
        ch_handler = ModelCheckpoint(config.result_path, 'best_checkpoint', score_function=score_fn, score_name=config.monitor.save_score, n_saved=1, require_empty=False, save_as_state_dict=True)
        to_save = dict(models, **optimizers)
        evaluator_engine.add_event_handler(Events.EPOCH_COMPLETED, ch_handler, to_save)

    if config.save_freq > 0:
        ch_handler = ModelCheckpoint(config.result_path, 'checkpoint', save_interval=config.save_freq, n_saved=config.nsave, require_empty=False, save_as_state_dict=True)
        to_save = dict(models, **optimizers)
        trainer_engine.add_event_handler(Events.EPOCH_COMPLETED, ch_handler, to_save)

    # Register custom callbacks with the engines
    if check_if_implemented(trainer, "on_iteration_start"):
        trainer_engine.add_event_handler(Events.ITERATION_STARTED, trainer.on_iteration_start, phase="train")
        evaluator_engine.add_event_handler(Events.ITERATION_STARTED, trainer.on_iteration_start, phase="evaluate")
    if check_if_implemented(trainer, "on_iteration_end"):
        trainer_engine.add_event_handler(Events.ITERATION_COMPLETED, trainer.on_iteration_end, phase="train")
        evaluator_engine.add_event_handler(Events.ITERATION_COMPLETED, trainer.on_iteration_end, phase="evaluate")
    if check_if_implemented(trainer, "on_epoch_start"):
        trainer_engine.add_event_handler(Events.EPOCH_STARTED, trainer.on_epoch_start, phase="train")
        evaluator_engine.add_event_handler(Events.EPOCH_STARTED, trainer.on_epoch_start, phase="evaluate")
    if check_if_implemented(trainer, "on_epoch_end"):
        trainer_engine.add_event_handler(Events.EPOCH_COMPLETED, trainer.on_epoch_end, phase="train")
        evaluator_engine.add_event_handler(Events.EPOCH_COMPLETED, trainer.on_epoch_end, phase="evaluate")

    # Save the config for this experiment to the results directory, once we know the params are good
    config.save()

    def signal_handler(sig, frame):
        print('You pressed Ctrl+C!')
        if exp_logger is not None:
            exp_logger.teardown()
            sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    logger.info("INFO: Starting training...")
    trainer_engine.run(loader_train, max_epochs=config.epochs)

    if exp_logger is not None:
        exp_logger.teardown()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', default=None, type=str, required=True, help='config file path (default: None)')
    parser.add_argument('--checkpoint', default=None, type=str, help='Checkpoint tag to reload')    
    parser.add_argument('--checkpoint_dir', default=None, type=str, help='Checkpoint directory to reload')
    parser.add_argument('--suffix', default=None, type=str, help='Add to the name')
    parser.add_argument('--epochs', default=None, type=int, help='Number of epochs')
    parser.add_argument('--resume_from', default=None, type=int, help='Epoch to resume from, allows using checkpoints as initialisation')
    args = parser.parse_args()

    OVERLOADABLE = ['checkpoint', 'epochs', 'checkpoint_dir', 'resume_from']

    overloaded = {}
    for k, v in vars(args).items():
        if (k in OVERLOADABLE) and (v is not None):
            overloaded[k] = v

    config = Experiment.load_from_path(args.config, overloaded, args.suffix)

    assert config, "Config could not be loaded."

    # Else load the saved config from the results dir or throw an error if one doesn't exist
    if len(config.checkpoint) > 0:
        logger.warning("WARNING: --config specifies resuming, overriding config with exising experiment config.")
        # resume_config = Experiment(config.name, desc=config.desc, result_dir=config.result_dir).load()
        # assert resume_config is not None, "No experiment {} exists, cannot resume training".format(config.name)
        # config = resume_config
        assert config, "Config could not be loaded for resume"
    # If we have resume_from in the config but have it < 0 to start a fresh training run then throw and error if the directory already exists
    elif config.overwrite is False:
        assert not config.exists(), "Results directory {} already exists! Please specify a new experiment name or the remove old files.".format(config.result_path)
    else:
        empty_folder(config.result_path)

    main(config)
