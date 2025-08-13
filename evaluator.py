from model_setups.prototype_model_runner import get_dataloader, load_model, training_epoch, get_model
from parameter_class import HyperparamStore, ModelParameters
from easyfsl.utils import evaluate
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import MultiStepLR
import copy
import numpy as np
import os
def main():
    model_name = "ptmap"
    save_path = f"./k_fold_eval_outs/{model_name}/"
    os.makedirs(save_path, exist_ok=True)
    num_runs = 5
    num_fractions = 4
    store = HyperparamStore()
    params = store.get_model_params(model_name, "hidden")
    all_accs, all_macros, all_calibs = k_fold_eval(num_runs, num_fractions, params)
    np.save(save_path+"all_accs.npy", np.asarray(all_accs))
    np.save(save_path+"all_macros.npy", np.asarray(all_macros))
    np.save(save_path+"all_calibs.npy", np.asarray(all_calibs))


def k_fold_eval(num_runs, num_fractions, params: ModelParameters):
    all_accuracies, all_macros, all_calibs = [], [], []
    for i in range(num_runs):
        print("run", i)
        run_accuracies, run_macros, run_calibs = [], [] ,[]
        params.n_query = 30 # start at 30, need to reduce as fraction decrease for sufficient data in each set
        for j in range(num_fractions):
            percent_train = int(80 - 10 * j)  # fraction from inner loop iterator
            if percent_train == 70: params.n_query = 25
            if percent_train == 60: params.n_query = 15
            if percent_train == 50: params.n_query = 6
            if j == 8: percent_train = 5
            print(f"Testing with {percent_train}% of training data, run {i}, n_query {params.n_query}")
            train_loader = get_dataloader(params.n_query, params.n_support, params.n_way, params.episodes_per_epoch,
                                          split="train", k_fold=True, run=i, frac=percent_train)
            val_loader = get_dataloader(params.n_query, params.n_support, params.n_way, params.validation_runs,
                                        split="cal", k_fold=True, run=i, frac=percent_train)
            test_loader = get_dataloader(params.n_query, params.n_support, params.n_way, 1,
                                          split="test", k_fold=True, run=i, frac=percent_train)
            few_shot_classifier = get_model(params.model_name)
            train_model(params, few_shot_classifier, train_loader, val_loader)
            accuracy, macros, calibs = evaluate(few_shot_classifier, test_loader, device=params.device)
            print(f"Test accuracy {accuracy}, at run {i}, {percent_train}% of training data")
            run_accuracies.append(accuracy)
            run_macros.append(macros)
            run_calibs.append(calibs)
        all_accuracies.append(run_accuracies)
        all_calibs.append(run_calibs)
        all_macros.append(run_macros)


    return all_accuracies, all_macros, all_calibs



def train_model(params: ModelParameters, few_shot_classifier, train_loader, val_loader):
    train_optimizer = SGD(
        few_shot_classifier.parameters(), lr=params.learning_rate, momentum=params.momentum, weight_decay=params.weight_decay
    )
    train_scheduler = MultiStepLR(
        train_optimizer,
        milestones=params.scheduler_milestones,
        gamma=params.scheduler_gamma,
    )
    best_validation_accuracy = 0.0
    best_state = copy.deepcopy(few_shot_classifier.state_dict())
    for epoch in range(params.training_epochs):
        loss = training_epoch(few_shot_classifier, train_loader, train_optimizer)
        train_scheduler.step()
        validation_accuracy, _, _, macs, confusion = evaluate(
            few_shot_classifier, val_loader, device=params.device, tqdm_prefix="Validation"
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_state = copy.deepcopy(few_shot_classifier.state_dict())
    return few_shot_classifier


def evaluate_model(params: ModelParameters):
    few_shot_classifier = load_model(params.model_name).to(params.device)
    test_loader = get_dataloader(params.n_query, params.n_support, params.n_way, num_runs=10, split="test")
    accuracy = evaluate(few_shot_classifier, test_loader, device=params.device)
    return accuracy



if __name__ == '__main__':
    main()