"""
General utilities
"""
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision
from matplotlib import pyplot as plt
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from easyfsl.methods import FewShotClassifier
import torch.nn.functional as F
from torchmetrics.classification import MulticlassCalibrationError, MulticlassF1Score, MulticlassConfusionMatrix

def plot_images(images: Tensor, title: str, images_per_row: int):
    """
    Plot images in a grid.
    Args:
        images: 4D mini-batch Tensor of shape (B x C x H x W)
        title: title of the figure to plot
        images_per_row: number of images in each row of the grid
    """
    plt.figure()
    plt.title(title)
    plt.imshow(
        torchvision.utils.make_grid(images, nrow=images_per_row).permute(1, 2, 0)
    )


def sliding_average(value_list: List[float], window: int) -> float:
    """
    Computes the average of the latest instances in a list
    Args:
        value_list: input list of floats (can't be empty)
        window: number of instances to take into account. If value is 0 or greater than
            the length of value_list, all instances will be taken into account.

    Returns:
        average of the last window instances in value_list

    Raises:
        ValueError: if the input list is empty
    """
    if len(value_list) == 0:
        raise ValueError("Cannot perform sliding average on an empty list.")
    return np.asarray(value_list[-window:]).mean()


def predict_embeddings(
    dataloader: DataLoader,
    model: nn.Module,
    device: Optional[str] = None,
) -> pd.DataFrame:
    """
    Predict embeddings for a dataloader.
    Args:
        dataloader: dataloader to predict embeddings for. Must deliver tuples (images, class_names)
        model: model to use for prediction
        device: device to cast the images to. If none, no casting is performed. Must be the same as
            the device the model is on.
    Returns:
        dataframe with columns embedding and class_name
    """
    all_embeddings = []
    all_class_names = []
    with torch.no_grad():
        for images, class_names in tqdm(
            dataloader, unit="batch", desc="Predicting embeddings"
        ):
            if device is not None:
                images = images.to(device)
            all_embeddings.append(model(images).detach().cpu())
            if isinstance(class_names, torch.Tensor):
                all_class_names += class_names.tolist()
            else:
                all_class_names += class_names

    concatenated_embeddings = torch.cat(all_embeddings)

    return pd.DataFrame(
        {"embedding": list(concatenated_embeddings), "class_name": all_class_names}
    )


def evaluate_on_one_task(
    model: FewShotClassifier,
    support_images: Tensor,
    support_labels: Tensor,
    query_images: Tensor,
    query_labels: Tensor,
) -> Tuple[int, int, float, float, torch.Tensor, torch.Tensor]:
    """
    Returns the number of correct predictions of query labels, and the total number of
    predictions.
    """
    model.process_support_set(support_images, support_labels)
    
    #query_labels = F.one_hot(query_labels, num_classes=5).float().view(-1, 5)
    predictions = model(query_images).detach().data#.view(int(len(query_labels)),5,5).max(2)[0] # Uncomment for relation networks
    number_of_correct_predictions = (
        (torch.max(predictions, 1)[1] == query_labels).sum().item()
    )
    #print(query_labels.unique())
    micro_f1 = MulticlassF1Score(num_classes=len(query_labels.unique()), average="micro")
    calibration_error = MulticlassCalibrationError(num_classes=len(query_labels.unique()), norm="l1")
    mic_f1 = micro_f1(predictions.detach().cpu(), query_labels.detach().cpu())
    cali_err = calibration_error(predictions.detach().cpu(), query_labels.detach().cpu())

    macro_f1 = MulticlassF1Score(num_classes=len(query_labels.unique()), average="macro")
    mac_f1 = macro_f1(predictions.detach().cpu(), query_labels.detach().cpu())

    confusion_matrix = MulticlassConfusionMatrix(num_classes=len(query_labels.unique()))
    confusion = confusion_matrix(predictions.detach().cpu(), query_labels.detach().cpu())

    return number_of_correct_predictions, len(query_labels), mic_f1, cali_err, mac_f1, confusion


def evaluate(
    model: FewShotClassifier,
    data_loader: DataLoader,
    device: str = "cuda",
    use_tqdm: bool = True,
    tqdm_prefix: Optional[str] = None,
) -> float:
    """
    Evaluate the model on few-shot classification tasks
    Args:
        model: a few-shot classifier
        data_loader: loads data in the shape of few-shot classification tasks*
        device: where to cast data tensors.
            Must be the same as the device hosting the model's parameters.
        use_tqdm: whether to display the evaluation's progress bar
        tqdm_prefix: prefix of the tqdm bar
    Returns:
        average classification accuracy
    """
    # We'll count everything and compute the ratio at the end
    total_predictions = 0
    correct_predictions = 0
    micros, calibs, macs, confusions = [], [], [], []
    # eval mode affects the behaviour of some layers (such as batch normalization or dropout)
    # no_grad() tells torch not to keep in memory the whole computational graph
    model.eval()
    with torch.no_grad():
        # We use a tqdm context to show a progress bar in the logs
        with tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            disable=not use_tqdm,
            desc=tqdm_prefix,
        ) as tqdm_eval:
            for _, (
                support_images,
                support_labels,
                query_images,
                query_labels,
                _,
            ) in tqdm_eval:
                correct, total, macro, calib, mac, confusion = evaluate_on_one_task(
                    model,
                    support_images.to(device),
                    support_labels.to(device),
                    query_images.to(device),
                    query_labels.to(device),
                )

                total_predictions += total
                correct_predictions += correct
                micros.append(macro)
                calibs.append(calib)
                macs.append(mac)
                confusions.append(confusion)
                # Log accuracy in real time
                tqdm_eval.set_postfix(accuracy=correct_predictions / total_predictions)

    return correct_predictions / total_predictions, micros, calibs, macs, confusions


def compute_average_features_from_images(
    dataloader: DataLoader,
    model: nn.Module,
    device: Optional[str] = None,
):
    """
    Compute the average features vector from all images in a DataLoader.
    Assumes the images are always first element of the batch.
    Returns:
        Tensor: shape (1, feature_dimension)
    """
    all_embeddings = torch.stack(
        predict_embeddings(dataloader, model, device)["embedding"].to_list()
    )
    average_features = all_embeddings.mean(dim=0)
    if device is not None:
        average_features = average_features.to(device)
    return average_features
