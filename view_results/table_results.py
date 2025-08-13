import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
def main():
    model_name = "bdlpn_just_get_performances"
    num_runs = 10
    data_dict = read_in_values(model_name, num_runs)
    masking_effect = [get_masking_effect(data_dict["ood_before"][i]) for i in range(len(data_dict["ood_before"]))]
    masking_effect_std = np.std(masking_effect)
    masking_effect = np.mean(masking_effect)
    swamping_effect_ood = [get_swamping_effect(data_dict["ood_after"][i]) for i in range(len(data_dict["ood_after"]))]
    swamping_effect_ood_std = np.std(swamping_effect_ood)
    swamping_effect_ood = np.mean(swamping_effect_ood)
    swamping_effect_iid = [get_swamping_effect(data_dict["iid_before"][i]) for i in range(len(data_dict["iid_before"]))]
    swamping_effect_iid_std = np.std(swamping_effect_iid)
    swamping_effect_iid = np.mean(swamping_effect_iid)
    swamping_effect_iid_plus_class = [get_swamping_effect(data_dict["iid_after"][i]) for i in range(len(data_dict["iid_after"]))]
    swamping_effect_iid_plus_class_std = np.std(swamping_effect_iid_plus_class)
    swamping_effect_iid_plus_class = np.mean(swamping_effect_iid_plus_class)
    avg_caliber_before, std_caliber_before = get_performance_data(data_dict["calib_before"][:])
    avg_f1_before, std_f1_before = get_performance_data(data_dict["f1_before"][:])
    avg_acc_before, std_acc_before = get_performance_data(data_dict["acc_before"][:])
    avg_caliber_after, std_caliber_after = get_performance_data(data_dict["calib_after"][:])
    avg_f1_after, std_f1_after = get_performance_data(data_dict["f1_after"][:])
    avg_acc_after, std_acc_after = get_performance_data(data_dict["acc_after"][:])
    ood_roc_auc_before = [get_auroc(data_dict["ood_before"][i], data_dict["ood_after"][i]) for i in range(len(data_dict["ood_before"]))]
    ood_roc_auc_before_std = np.std(ood_roc_auc_before)
    ood_roc_auc = np.mean(ood_roc_auc_before)
    #ood_roc_auc_after = get_auroc(data_dict["ood_before"][:, :], data_dict["ood_after"][:, :])
    out_string = ("{:.2f}+-{:.2f}\t{:.2f}+-{:.2f}\t{:.2f}+-{:.2f}\t{:.2f}+-{:.2f}\t{:.2f}+-{:.2f}\t{:.2f}+-{:.2f}\t{:.2f}+-{:.2f}\t{:.2f}+-{:.2f}\t{:.2f}+-{:.2f}\t{:.2f}+-{:.2f}\t{:.2f}"
                  "+-{:.2f}")
    out_string = out_string.format(masking_effect, masking_effect_std, swamping_effect_ood, swamping_effect_ood_std,
                                   swamping_effect_iid, swamping_effect_iid_std,
                                   swamping_effect_iid_plus_class, swamping_effect_iid_plus_class_std,
                                   avg_caliber_before, std_caliber_before, avg_f1_before, std_f1_before, avg_acc_before,
                                   std_acc_before,
                                   avg_caliber_after, std_caliber_after, avg_f1_after, std_f1_after, avg_acc_after,
                                   std_acc_after,
                                   ood_roc_auc, ood_roc_auc_before_std)
    print(out_string)
    show_confusion_matrix(data_dict["confusion_before"].squeeze(), np.average(data_dict["macro_before"]))





def read_in_values(path, num_runs):
    path = f"../pval_tests/{path}/"
    all_before_accs, all_after_accs, all_before_calibs, all_after_calibs, all_before_f1,\
    all_after_f1, all_before_ood, all_after_ood, all_before_iid, all_after_iid,\
    all_before_macro, all_before_confusion = [],[],[],[],[],[],[],[],[],[], [], []
    for i in range(num_runs):
        loading_path = path+f"run_{i}/frac_70/"
        before_acc = np.load(loading_path+"before_acc.npy")
        before_calib = np.load(loading_path+"before_calib.npy")
        before_f1 = np.load(loading_path+"before_micro.npy")
        before_ood = np.load(loading_path+"before_train.npy")
        before_iid = np.load(loading_path+"before_train_iid.npy")
        after_acc = np.load(loading_path + "after_acc.npy")
        after_calib = np.load(loading_path + "after_calib.npy")
        after_f1 = np.load(loading_path + "after_micro.npy")
        after_ood = np.load(loading_path + "after_train.npy")
        after_iid = np.load(loading_path + "after_train_iid.npy")

        before_macro = np.load(loading_path + "before_macro.npy")
        before_confusion = np.load(loading_path + "before_confusion.npy")

        all_before_accs.append(before_acc)
        all_after_accs.append(after_acc)
        all_before_calibs.append(before_calib)
        all_after_calibs.append(after_calib)
        all_before_f1.append(before_f1)
        all_after_f1.append(after_f1)
        all_before_iid.append(before_iid)
        all_after_iid.append(after_iid)
        all_before_ood.append(before_ood)
        all_after_ood.append(after_ood)

        all_before_macro.append(before_macro)
        all_before_confusion.append(before_confusion)


    return {
        "acc_before": np.array(all_before_accs),
        "acc_after": np.array(all_after_accs),
        "calib_before": np.array(all_before_calibs),
        "calib_after": np.array(all_after_calibs),
        "f1_before": np.array(all_before_f1),
        "f1_after": np.array(all_after_f1),
        "iid_before": np.array(all_before_iid),
        "iid_after": np.array(all_after_iid),
        "ood_before": np.array(all_before_ood),
        "ood_after": np.array(all_after_ood),
        "macro_before": np.array(all_before_macro),
        "confusion_before": np.array(all_before_confusion),
    }

def get_auroc(true, true2, iid=False):
    pred = np.array([int(x>0.95) for x in true.flatten()])
    pred2 = np.array([int(x>0.95) for x in true2.flatten()])
    if iid:
        positive_label=0
        negative_label=1
    else:
        positive_label=1
        negative_label=0
    true = [positive_label]*len(pred)
    others = [negative_label]*len(pred2)
    true = np.append(true, others)
    pred = np.append(pred, pred2)
    fpr, tpr, thresholds = metrics.roc_curve(true, pred, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    #metrics.RocCurveDisplay.from_predictions(true, pred)
    #plt.show()
    return roc_auc


def get_masking_effect(ood_scores):
    """
    :param ood_scores: numpy arr of ood scores
    :return: # of outliers falsely detected as inliers
    """
    total_examples = ood_scores.shape[-1]*ood_scores.shape[0]
    total_iid = len(np.extract(ood_scores < 0.95, ood_scores))
    return total_iid/total_examples


def get_swamping_effect(iid_scores):
    """
    :param iid_scores: numpy arr of ood scores
    :return: # of inliers falsely detected as outliers
    """
    # Total examples are # of examples per run * number of runs.
    total_examples = iid_scores.shape[-1]*iid_scores.shape[0]
    total_ood_per = len(np.extract(iid_scores >= 0.95, iid_scores))
    return total_ood_per/total_examples

def get_performance_data(performance_data):
    # shape: n_runs, n_z_dims
    average_metric = performance_data.mean(axis=0)
    std_metric = performance_data.std(axis=0)
    return average_metric, std_metric

def show_confusion_matrix(confusion_matrix, macro_f1, data_frac=None):
    #confusion_matrix = confusion_matrix.mean(axis=0)
    class_map = {0: "C2", 1: "CHAT", 2 : "FT", 3: "STREAM", 4: "VOIP"}
    if len(confusion_matrix[0]) > 8:
        class_map = {
            "FACEBOOK": 0, "TWITTER": 1, "REDDIT": 2, "INSTAGRAM": 3, "PINTREST": 4,
            "YOUTUBE": 5, "NETFLIX": 6, "HULU": 7, "SPOTIFY": 8, "PANDORA": 9,
            "DRIVE": 11, "DROPBOX": 12, "GMAIL": 13, "MESSENGER": 14
        } # NOTE HANGOUT AND MAPS HAD NO EXAMPLES
    elif len(confusion_matrix[0]) == 7:
        class_map = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7}
    elif len(confusion_matrix[0]) == 6:
        class_map = {0: "C2", 1: "CHAT", 2: "FT", 3: "STREAM", 4: "VOIP", 5: "SPOTIFY"}
    index = int((0.8) * 10)
    all_data = []
    for i in range(len(confusion_matrix)):
        data = confusion_matrix[i]/confusion_matrix[i, 0, :].sum()
        if i == 0: all_data = data
        else: all_data += data
    all_data = all_data/len(confusion_matrix)
    plot_df = pd.DataFrame(data=all_data,
                           columns=class_map.values(), index=class_map.values())
    ax = sns.heatmap(plot_df, annot=True, fmt=".1f", yticklabels=class_map.values(), xticklabels=True, cbar=False)
    #cbar = ax.collections[0].colorbar
    #cbar.set_label("Class Balance", rotation=270, labelpad=15)
    #cbar = ax.colle
    legend_elements = [
        Line2D([0], [0], color='none', label=f"{i}: {label}")
        for i, label in class_map.items()
    ]

    legend_elements.append(
        Line2D([0], [0], color='none', label=f"\n\n\n\nMacro F1 Score: {macro_f1:.2f}")
    )
    plt.legend(
        handles=legend_elements,
        title="Class to Index Mapping",
        bbox_to_anchor=(1.05, 1), loc='upper left'
    )
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()