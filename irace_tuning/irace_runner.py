#!/home/callumm/.virtualenvs/enc-vpn-uncertainty-class-repl/bin/python
import os


import copy
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from easyfsl.samplers import TaskSampler
from easyfsl.datasets import FeaturesDataset
from torch.utils.data import DataLoader

from easyfsl.methods import FewShotClassifier, MatchingNetworks, BDCSPN

from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

from easyfsl.utils import evaluate


import sys
import logging
import random

from model_setups.prototype_model_runner import train_val_dataloader_getter, get_model, training_epoch


import os
os.environ["TQDM_DISABLE"] = "True"

from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

import torch
#torch.set_num_threads(16)



logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG,
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename='irace_runner.log', filemode='a', )
logger = logging.getLogger(__name__)
#LOSS_FUNCTION = nn.CrossEntropyLoss()
#LOSS_FUNCTION = nn.NLLLoss()
LOSS_FUNCTION = nn.MSELoss()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#DEVICE = "cpu"
tb_logs_dir = Path("./logs/tb_logs")

def main():

    configuration_id = sys.argv[1]
    instance_id = sys.argv[2]
    seed = sys.argv[3]
    instance = sys.argv[4]
    cand_params = sys.argv[5:]

    logger.debug(
        f"Configuration ID: {configuration_id}, Instance ID: {instance_id}, Seed: {seed}, Instance: {instance}")
    # set seeds
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    # dataset specific parameters
    episodes_per_epoch = 20000
    validation_runs = 100
    n_way = 5
    n_support = 5
    n_query = 30

    n_workers = 1
    model_name = "ptmap"
    train_loader, val_loader = train_val_dataloader_getter(episodes_per_epoch, n_query, n_support, n_way,
                                                           validation_runs)

    # Get model
    few_shot_classifier = get_model(model_name)
    # default hyperparameters
    n_epochs = 1
    scheduler_milestones = [20, 50, 100]
    scheduler_gamma = 0.01
    learning_rate = 0.001
    weight_decay = 0.0
    momentum = 0.9


    while cand_params:
        param = cand_params.pop(0)
        value = cand_params.pop(0)
        logger.debug(f"Parameter parsed: {param} = {value}")
        match param:
            case "--lr":
                learning_rate = float(value)
            case "--weight_decay":
                weight_decay = float(value)
            case "--gamma":
                scheduler_gamma = float(value)
            case "--momentum":
                momentum = float(value)

    train_optimizer = SGD(
        few_shot_classifier.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay
    )
    train_scheduler = MultiStepLR(
        train_optimizer,
        milestones=scheduler_milestones,
        gamma=scheduler_gamma,
    )

    tb_writer = SummaryWriter(log_dir=str(tb_logs_dir))

    # start train loop
    best_state = few_shot_classifier.state_dict()
    best_validation_accuracy = 0.0
    final_loss = 100000
    for epoch in range(n_epochs):
        average_loss = training_epoch(few_shot_classifier, train_loader, train_optimizer)
        validation_accuracy = evaluate(
            few_shot_classifier, val_loader, device=DEVICE, use_tqdm=False, tqdm_prefix="Validation"
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_state = copy.deepcopy(few_shot_classifier.state_dict())
            logger.info(f"Best validation accuracy: {best_validation_accuracy}, at epoch {epoch}")
            # state_dict() returns a reference to the still evolving model's state so we deepcopy
            # https://pytorch.org/tutorials/beginner/saving_loading_models

        tb_writer.add_scalar("Train/loss", average_loss, epoch)
        tb_writer.add_scalar("Val/acc", validation_accuracy, epoch)
        final_loss = average_loss

        # Warn the scheduler that we did an epoch
        # so it knows when to decrease the learning rate
        train_scheduler.step()
    logger.info(f"validation loss: {final_loss}")
    print(final_loss)

    #torch.save(few_shot_classifier.state_dict(), f"trained_models/{model_name}.pt")


if __name__ == "__main__":
    main()
