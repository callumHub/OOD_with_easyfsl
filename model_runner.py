import numpy as np
import copy
from pathlib import Path
from statistics import mean

import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from easyfsl.samplers import TaskSampler
from easyfsl.datasets import FeaturesDataset
from torch.utils.data import DataLoader

from easyfsl.methods import FewShotClassifier, MatchingNetworks, BDCSPN
from easyfsl.utils import evaluate_on_one_task
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

from easyfsl.utils import evaluate


import sys
import logging
import random

from model_setups.prototype_model_runner import train_val_dataloader_getter, get_model, training_epoch, cal_test_dataloader_getter


import os
#os.environ["TQDM_DISABLE"] = "True"

from functools import partialmethod

from parameter_class import HyperparamStore

#tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)




logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG,
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename='irace_runner.log', filemode='a')
logger = logging.getLogger(__name__)
#LOSS_FUNCTION = nn.CrossEntropyLoss()
LOSS_FUNCTION = nn.NLLLoss()
#LOSS_FUNCTION = nn.MSELoss()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#DEVICE = "cpu"
def main():
    tb_logs_dir = Path("./logs/march_tb_logs")

    learning_rate = 0.0001
    scheduler_gamma = 0.447
    momentum = 0.921219

    episodes_per_epoch = 20000
    validation_runs = 100
    n_way = 5
    n_support = 5
    n_query = 30

    model_name = "bdcspn"
    few_shot_classifier = get_model(model_name)
    params = HyperparamStore().get_model_params("bdcspn", "hidden")
    params.episodes_per_epoch = 44
    params.training_epochs = 500
    params.n_epochs = 500
    params.scheduler_milestones = [250, 400, 469]
    n_workers = 1
    train_loader, val_loader = train_val_dataloader_getter(params.episodes_per_epoch, params.n_query, params.n_support, params.n_way,
                                                           validation_runs)
    n_epochs = 2
    scheduler_milestones = [2, 5, 10]
    weight_decay = 0.0


    train_optimizer = SGD(
        few_shot_classifier.parameters(), lr=params.learning_rate, momentum=params.momentum, weight_decay=params.weight_decay
    )
    train_scheduler = MultiStepLR(
        train_optimizer,
        milestones=params.scheduler_milestones,
        gamma=params.scheduler_gamma,
    )

    tb_writer = SummaryWriter(log_dir=str(tb_logs_dir))

    # start train loop
    best_state = few_shot_classifier.state_dict()
    best_validation_accuracy = 0.0
    final_loss = 100000
    for epoch in range(params.n_epochs):
        average_loss = training_epoch(few_shot_classifier, train_loader, train_optimizer)
        validation_accuracy, f1, calibration_error = evaluate(
            few_shot_classifier, val_loader, device=DEVICE, use_tqdm=False, tqdm_prefix="Validation"
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_state = copy.deepcopy(few_shot_classifier.state_dict())
            print(f"Best validation accuracy: {best_validation_accuracy}, at epoch {epoch}")
            # state_dict() returns a reference to the still evolving model's state so we deepcopy
            # https://pytorch.org/tutorials/beginner/saving_loading_models

        tb_writer.add_scalar("Train/loss", average_loss, epoch)
        tb_writer.add_scalar("Val/acc", validation_accuracy, epoch)
        tb_writer.add_scalar("Val/f1", torch.mean(torch.tensor(f1)), epoch)
        tb_writer.add_scalar("Val/calibration_error", torch.mean(torch.tensor(calibration_error)), epoch)
        final_loss = average_loss
        # Warn the scheduler that we did an epoch
        # so it knows when to decrease the learning rate
        train_scheduler.step()
    print(final_loss)
    print(best_validation_accuracy)
    torch.save(few_shot_classifier.state_dict(), f"trained_models/{model_name}.pt")

def run_calibrate_and_test(fs_classifier: FewShotClassifier, cal_loader, test_loader, val_loader, bandwidth):
    g = None
    final_p_vals, final_acc, final_calib, final_micro, final_confusion, final_macro = [], [], [], [], [], []
    sort_dex = None
    for support_examples, support_labels, query_examples, query_labels, _ in cal_loader:

        n_class = len(query_labels.unique())
        n_cal = int(query_examples.shape[0]/n_class)
        g = fs_classifier.compute_kernels_from_calibration_set(query_examples, query_labels, support_examples, support_labels, n_class, n_cal, bandwidth)
    for support_examples, support_labels, query_examples, query_labels, _ in test_loader:
        n_class = len(query_labels.unique())
        n_cal = int(query_examples.shape[0]/n_class)
        #fs_classifier.process_support_set(support_examples, support_labels)
        for se, sl, qe, ql, _ in val_loader:
            correct, total, final_micro, final_calib, final_macro, final_confusion= evaluate_on_one_task(fs_classifier, se, sl, qe, ql)
            final_acc = correct/total
        final_p_vals, _, _, _,_ = fs_classifier.ood_test_alg(query_examples, query_labels, n_class, n_cal, g)
    return final_p_vals, final_acc, final_calib, final_micro, final_macro, final_confusion, g



if __name__ == "__main__":
    main()
    few_shot_classifier = get_model("bdcspn")
    few_shot_classifier.load_state_dict(torch.load("trained_models/bdcspn.pt"))
    p_vals, acc, calib, micro, confusion, g_k = run_calibrate_and_test(few_shot_classifier)
    pvals = np.asarray(p_vals)
    print(np.mean(p_vals))
    print(np.std(p_vals))
