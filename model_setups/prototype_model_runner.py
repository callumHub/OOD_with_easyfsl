'''
This script is to test out easy fsl
directly adapted from https://colab.research.google.com/drive/1IvFa97JEy9kUSfa8cQdk2DoV7hXWheit#scrollTo=N160lWB7rg7B&uniqifier=1

Plan is to reorganize everything in a neat way after prototype functions
'''

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

from easyfsl.methods import FewShotClassifier, MatchingNetworks, BDCSPN, LaplacianShot, PTMAP, TIM, RelationNetworks, PrototypicalNetworks

from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

from easyfsl.utils import evaluate
import logging
from model_setups.bdcspn_with_l2 import BDLPN


from .model_definitions import my_mlp, my_bilstm, pt_map_mlp
import torch.nn.functional as F

import os
#os.environ["TQDM_DISABLE"] = "True"

from functools import partialmethod
#tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"




#LOSS_FUNCTION = nn.CrossEntropyLoss()
LOSS_FUNCTION = nn.NLLLoss() # NLL loss for bdcspn.
#LOSS_FUNCTION = nn.MSELoss()
logger = logging.getLogger(__name__)
def main():
    episodes_per_epoch = 2000
    validation_runs = 100
    n_way = 5
    n_support = 5
    n_query = 30

    n_workers = 1

    model_name = "bdcspn"#"mlp_matching_networks"
    train_loader, val_loader = train_val_dataloader_getter(episodes_per_epoch, n_query, n_support, n_way, n_workers,
                                                           validation_runs)

    # Get model
    few_shot_classifier = get_model("bdcspn")

    # Train setup


    n_epochs = 1000
    scheduler_milestones = [300, 450, 750, 900]
    scheduler_gamma = 0.01
    learning_rate = 0.0004
    tb_logs_dir = Path("./logs/tb_logs")

    train_optimizer = SGD(
        few_shot_classifier.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-8
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
    for epoch in range(n_epochs):
        print(f"Epoch {epoch}")
        average_loss = training_epoch(few_shot_classifier, train_loader, train_optimizer)
        validation_accuracy = evaluate(
            few_shot_classifier, val_loader, device=DEVICE, tqdm_prefix="Validation"
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_state = copy.deepcopy(few_shot_classifier.state_dict())
            # state_dict() returns a reference to the still evolving model's state so we deepcopy
            # https://pytorch.org/tutorials/beginner/saving_loading_models
            print("Ding ding ding! We found a new best model!")

        tb_writer.add_scalar("Train/loss", average_loss, epoch)
        tb_writer.add_scalar("Val/acc", validation_accuracy, epoch)

        # Warn the scheduler that we did an epoch
        # so it knows when to decrease the learning rate
        train_scheduler.step()

    torch.save(few_shot_classifier.state_dict(), f"trained_models/{model_name}.pt")


def train_val_dataloader_getter(episodes_per_epoch, n_query, n_support, n_way, validation_runs):
    # Get datasets
    train_loader = get_dataloader(n_query=n_query, n_support=n_support, n_way=n_way, num_runs=episodes_per_epoch, split="train")
    val_loader = get_dataloader(n_query=n_query, n_support=n_support, n_way=n_way, num_runs=validation_runs, split="cal")
    return train_loader, val_loader

def cal_test_dataloader_getter(episodes_per_epoch, n_query, n_support, n_way, validation_runs):
    cal_loader = get_dataloader(n_query=n_query, n_support=n_support, n_way=n_way, num_runs=1, split="cal")
    test_loader = get_dataloader(n_query=n_query, n_support=n_support, n_way=n_way, num_runs=1, split="test")
    return cal_loader, test_loader

def get_dataloader(n_query, n_support, n_way, num_runs, split, k_fold=False, run=None, frac=None):
    if os.path.exists("../data"):
        data_path = "./data"
    else:
        data_path = "../../../data"
    if k_fold:
        data_path = f"../enc-vpn-uncertainty-class-repl/processed_data/stable_cal_fraction/min_max_normalized/run{run}/frac_{frac}/{split}.jsonl"
        dataset, min_class = load_features_dataset(data_path)
    else:
        dataset, min_class = load_features_dataset(data_path+f"/vpn_dataset/{split}.jsonl")
    sampler = TaskSampler(
        dataset,  n_way=n_way, n_shot=n_support, n_query=n_query, n_tasks=num_runs
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0,
        pin_memory=True,
        collate_fn=sampler.episodic_collate_fn,
    )
    return loader


def get_model(model_name) -> FewShotClassifier:
    x_dim = 128
    # according to easyfsl/modules/predesigned modules.py, q encoder just one single lstm cell:
    match model_name:
        case "matching_networks":
            query_encoder = nn.LSTMCell(x_dim*2, x_dim)
            support_encoder = my_bilstm(128, 128, 64, device=DEVICE)
            fs_classifier = MatchingNetworks(feature_dimension=128, support_encoder=support_encoder, query_encoder=query_encoder)
            return fs_classifier
        case "relation_networks":
            encoder = my_mlp(256, 64, 5)
            return RelationNetworks(feature_dimension=128, relation_module=encoder)
        case "bdcspn":
            encoder = my_mlp(x_dim, 64, 64)
            fs_classifier = BDCSPN(backbone=encoder)
            return fs_classifier
        case "bdlpn":
            encoder = my_mlp(x_dim, 64, 64)
            fs_classifier = BDLPN(backbone=encoder)
            return fs_classifier
        case "laplacian_shot":
            encoder = my_mlp(x_dim, 64, 64)
            fs_classifier = LaplacianShot(backbone=encoder)
            return fs_classifier
        case "ptmap":
            encoder = pt_map_mlp(x_dim, 128, 64).extend(nn.Sequential(nn.ReLU()))
            fs_classifier = PTMAP(backbone=encoder, lambda_regularization=16)
            return fs_classifier
        case "tim":
            encoder = my_mlp(x_dim, 64, 64)
            fs_classifier = TIM(backbone=encoder)
            return fs_classifier
        case "pnet":
            encoder = my_mlp(x_dim, 64, 64)
            fs_classifier =  PrototypicalNetworks(backbone=encoder)
            return fs_classifier

def load_model(model_name):
    model = get_model(model_name)
    model.load_state_dict(torch.load(f"trained_models/{model_name}.pt"))
    return model



def load_features_dataset(path):
    combined_data = pd.read_json(path, lines=True)
    combined_data["class_name"] = combined_data.labels
    combined_data["embedding"] = combined_data.data.apply(lambda x: np.array([y for y in np.asarray(x, dtype=np.float32)]))
    min_class_num = min(combined_data["class_name"].value_counts().tolist())
    del combined_data["labels"]
    del combined_data["data"]
    return FeaturesDataset.from_dataframe(combined_data), min_class_num


def training_epoch(
    model: FewShotClassifier, data_loader: DataLoader, optimizer: Optimizer
):
    all_loss = []
    model.to(DEVICE, non_blocking=True)
    model.train()
    with tqdm(
        enumerate(data_loader), total=len(data_loader), desc="Training"
    ) as tqdm_train:
        for episode_index, (
            support_examples,
            support_labels,
            query_examples,
            query_labels,
            _,
        ) in tqdm_train:
            optimizer.zero_grad()
            support_examples = support_examples
            query_examples = query_examples
            model.process_support_set(
                support_examples.to(DEVICE, non_blocking=True), support_labels.to(DEVICE, non_blocking=True)
            )
            classification_scores = model(query_examples.to(DEVICE, non_blocking=True))
            classification_scores = F.log_softmax(classification_scores, dim=1)
            #classification_scores = classification_scores.view(int(len(query_examples)),5,5).max(2).values.float()
            # #Uncomment for relation_networks
            #classification_scores = classification_scores#.T # Uncomment for PTMAP
            #query_labels = query_labels.float() # Uncomment for PTMAP
            #query_labels = query_labels.expand(classification_scores.T.shape).T

            loss = LOSS_FUNCTION(classification_scores.to(DEVICE, non_blocking=True), query_labels.to(DEVICE, non_blocking=True))
            #assert torch.isfinite(classification_scores).all(), "NaN/Inf detected in logits"

            #assert torch.isfinite(loss).all(), "NaN/Inf detected in loss"
            #logger.info(f"Loss: {loss}")
            assert loss >= 0, f"negative loss detected: {loss}"
            loss.backward()
            optimizer.step()

            all_loss.append(loss.item())

            tqdm_train.set_postfix(loss=mean(all_loss))

    return mean(all_loss)

if __name__ == "__main__":
    main()




