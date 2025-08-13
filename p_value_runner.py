import os
from easyfsl.samplers import TaskSampler
from torch.utils.data import DataLoader
import numpy as np
import torch



from model_runner import run_calibrate_and_test
from model_setups.prototype_model_runner import load_features_dataset, get_model
from parameter_class import HyperparamStore, ModelParameters
# TODO: train_model should not be in evaluator
from evaluator import train_model
COMBINED_PATH="../enc-vpn-uncertainty-class-repl/processed_data/coarse_grain_ood/all_classes"
HELD_OUT_PATH="../enc-vpn-uncertainty-class-repl/processed_data/coarse_grain_ood/no_spotify"
ood_class = "SPOTIFY"
iid_class = "YOUTUBE"


def main():
    model_name = "bdlpn_just_get_performances" ## PREV USED CAL TO MAHAL
    run_query_data_experiment(model_name)
    model_name = "bdcspn_decrease_cal_0p05bw_sup_to_make_mahal_spotify_10run"
    run_query_data_experiment(model_name)

def run_query_data_experiment(model_name):
    model_to_load = model_name.split("_")[0]
    params = HyperparamStore().get_model_params(model_to_load, "hidden")
    fracs = [20, 30, 40, 50, 60, 70, 80]
    fracs = [70]
    bws = [0.05]
    for k in range(1):
        for i in range(10):
            for j in range(len(fracs)):
                print("RUNNING WITH Cal Frac: ", fracs[j])
                #combined_path = f"../enc-vpn-uncertainty-class-repl/processed_data/decreasing_cal_stable_train_ut/run{i}/frac_{fracs[j]}/all_classes"
                #one_out_path = f"../enc-vpn-uncertainty-class-repl/processed_data/decreasing_cal_stable_train_ut/run{i}/frac_{fracs[j]}/no_spotify"
                combined_path = f"../enc-vpn-uncertainty-class-repl/processed_data/stable_cal_fraction_ut/min_max_normalized/decreasing_cal_decrease_train/run{i}/frac_60"
                one_out_path = combined_path
                run_pval_experiment(model_name, save_modifier=f"run_{i+k*10}/frac_{fracs[j]}",
                                    params=params, model_to_load=model_to_load,
                                    combined_path=combined_path, held_out_path=one_out_path, bw=bws[0],counter=j)



def run_pval_experiment(model_name, save_modifier, params, model_to_load, combined_path, held_out_path, bw, n_episodes=None, counter=None):
    # load model
    params.n_classes = 14
    few_shot_classifier = get_model(model_to_load)
    if n_episodes:
        params.n_episodes = n_episodes
        params.episodes_per_epoch = n_episodes
        params.n_epochs = 10
        params.training_epochs = 10
        params.model_name = model_to_load+f"_{n_episodes}"

    held_out_data_train, combined_data_train = load_data(combined_path=combined_path, held_out_path=held_out_path,
                                                         params=params, split="train", counter=counter)


    held_out_data_val, combined_data_val = load_data(combined_path=combined_path, held_out_path=held_out_path,
                                                     params=params, split="test", val=True, counter=counter)
    held_out_data_cal, combined_data_cal = load_data(combined_path=combined_path, held_out_path=held_out_path,
                                                     params=params, split="cal", counter=counter)
    held_out_data_test, combined_data_test = load_data(combined_path=combined_path, held_out_path=held_out_path,
                                                       params=params, split="test", counter=counter)
    specific_ood_class_test = find_ood_class_index(held_out_data_test, combined_data_test)
    specific_ood_class_cal = find_ood_class_index(held_out_data_cal, combined_data_cal)
    # map_cal_to_test = cal_test_kernel_map(combined_data_test, combined_data_cal)

    # train on held out set
    params.n_classes = 14
    params.n_way = 14
    few_shot_classifier = train_model(params, few_shot_classifier, held_out_data_train, held_out_data_val)
    # calibrate on held out set
    few_shot_classifier.to("cpu")
    few_shot_classifier.backbone.eval()
    with torch.no_grad():
        p_vals, acc, calib, micro, macro, confusion, g_k = run_calibrate_and_test(few_shot_classifier,
                                                                           cal_loader=held_out_data_cal,
                                                                           test_loader=held_out_data_test,
                                                                           val_loader=held_out_data_val,
                                                                           bandwidth=bw)
    g_k_class_names = held_out_data_cal.dataset.class_names
    params.model_name = model_name
    # check p-vals/test on combined set

    with torch.no_grad():
        ood_scores = class_pval_getter(few_shot_classifier, combined_data_test, g_k, g_k_class_names, ood_class)
        iid_scores = class_pval_getter(few_shot_classifier, combined_data_test, g_k, g_k_class_names, iid_class)

    save(np.array(ood_scores), "before_train", params.model_name+"/"+save_modifier)
    save(np.array(iid_scores), "before_train_iid", params.model_name+"/"+save_modifier)
    save(np.array(acc), "before_acc", params.model_name + "/" + save_modifier)
    save(np.asarray(calib), "before_calib", params.model_name + "/" + save_modifier)
    save(np.asarray(micro), "before_micro", params.model_name + "/" + save_modifier)
    save(np.asarray(macro), "before_macro", params.model_name + "/" + save_modifier)
    save(np.array(confusion), "before_confusion", params.model_name + "/" + save_modifier)
    pass
    # OLD LOCATION OF INCREASE CLASS COUNT
    params.n_classes = 14
    params.n_way = 14

    few_shot_classifier = get_model(model_to_load)
    few_shot_classifier = train_model(params, few_shot_classifier, combined_data_train, combined_data_val)
    few_shot_classifier.to("cpu")
    few_shot_classifier.backbone.eval()
    with torch.no_grad():
        p_vals, acc, calib, micro, macro, confusion, g_k = run_calibrate_and_test(few_shot_classifier,
                                                                           cal_loader=combined_data_cal,
                                                                           test_loader=combined_data_test,
                                                                           val_loader=combined_data_val,
                                                                           bandwidth=bw)
    g_k_class_names = combined_data_cal.dataset.class_names
    with torch.no_grad():
        ood_scores = class_pval_getter(few_shot_classifier, combined_data_test, g_k, g_k_class_names, ood_class)
        iid_scores = class_pval_getter(few_shot_classifier, combined_data_test, g_k, g_k_class_names, iid_class)
    save(np.array(ood_scores), "after_train", params.model_name+"/"+save_modifier)
    save(np.array(iid_scores), "after_train_iid", params.model_name + "/" + save_modifier)
    save(np.array(acc), "after_acc", params.model_name+"/"+save_modifier)
    save(np.asarray(calib), "after_calib", params.model_name+"/"+save_modifier)
    save(np.asarray(micro), "after_micro", params.model_name+"/"+save_modifier)
    save(np.asarray(macro), "after_macro", params.model_name+"/"+save_modifier)
    save(np.array(confusion), "after_confusion", params.model_name + "/" + save_modifier)




def class_pval_getter(fs_classifier, dataloader, g_k, g_k_class_names, specific_class):
    pvals = []
    fs_classifier.eval()
    specific_class_index = dataloader.dataset.class_names.index(specific_class)
    #g_k_class_index = g_k_class_names[specific_class_index]
    class_data_indices = (torch.tensor(dataloader.dataset.labels) == specific_class_index).nonzero(as_tuple=True)[0]
    specific_samples = dataloader.dataset.embeddings.clone().detach()[class_data_indices]
    #specific_labels = dataloader.dataset.labels[class_data_indices]
    pvals = fs_classifier.ood_score_sample(specific_samples, g_k)
    return pvals


def load_data(combined_path, held_out_path, params: ModelParameters, split, val=False, counter=None):
    if split != "train": params.episodes_per_epoch = 1
    if split == "test": params.episodes_per_epoch = 25
    if split == "train": params.episodes_per_epoch = 20000
    if val: params.episodes_per_epoch = 25
    def _load(path, n_classes, s):
        params.n_support = 5
        data, min_class = load_features_dataset(path+f"/{s}.jsonl")
        n_classes = len(data.class_names)
        min_class = min_class -1

        if split == "train":
            min_class = min_class - params.n_classes
            print("Train Query Examples: ", min_class-params.n_support)
        if split == "cal":
            min_class = min_class - params.n_classes + 6
            print("Calibration Query Examples: ", min_class-params.n_support)
        if split == "test":
            min_class = min_class - params.n_classes + 6
            print("Test Query Examples: ", min_class-params.n_support)
        sampler = TaskSampler(data, n_way=n_classes, n_shot=n_classes,
                              n_query=min_class-params.n_support, n_tasks=params.episodes_per_epoch)
        loader = DataLoader(data, batch_sampler=sampler,
                            num_workers=0, pin_memory=True, collate_fn=sampler.episodic_collate_fn)
        return loader

    return _load(held_out_path, params.n_classes, split), _load(combined_path, params.n_classes, split) # JUST FOR PER CLASS EVAL
    return _load(held_out_path, params.n_classes-1, split), _load(combined_path, params.n_classes, split)




def cal_test_kernel_map(dl1, dl2):
    print("per class counts OOD", [len(list(dl1.sampler.items_per_label.values())[i]) for i in range(len(dl1.sampler.items_per_label))])
    print("per class counts IID", [len(list(dl2.sampler.items_per_label.values())[i]) for i in range(len(dl2.sampler.items_per_label))])
    ood_len_list = [len(list(dl1.sampler.items_per_label.values())[i]) for i in range(len(dl1.sampler.items_per_label))]
    iid_len_list = [len(list(dl2.sampler.items_per_label.values())[i]) for i in range(len(dl2.sampler.items_per_label))]
    cal_test_index_map = {}

    return cal_test_index_map




def find_ood_class_index(dl1, dl2):
    ood_counts = [len(list(dl1.batch_sampler.items_per_label.values())[i]) for i in range(len(dl1.batch_sampler.items_per_label))]
    iid_counts = [len(list(dl2.batch_sampler.items_per_label.values())[i]) for i in range(len(dl2.batch_sampler.items_per_label))]
    ood_class_index = len(iid_counts)
    for i in range(len(ood_counts)):
        if ood_counts[i] != iid_counts[i]:
            ood_class_index = i
            break
    return ood_class_index

def save(data: np.array, data_name, model_name):
    os.makedirs(f"pval_tests/{model_name}/", exist_ok=True)
    np.save(arr=data, file=f"pval_tests/{model_name}/{data_name}.npy")

if __name__ == '__main__':
    main()