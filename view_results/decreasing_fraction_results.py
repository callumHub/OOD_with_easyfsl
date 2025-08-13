import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from table_results import get_performance_data, get_auroc ,get_swamping_effect, get_masking_effect

# Query frac & example count:
# [(20,119), (30,103), (40,88), (50, 73), (60, 57), (70, 42), (80,25)]
def main():
    out_folder = "bdlpn_1query_0p05bw_proto_to_make_mahal_spotify_10run_test_jun_25_increasing_query" # MAY 5: "bdlpn_decrease_cal_0p05bw_sup_to_make_mahal" tabled results make sense
    fractions = [20, 30, 40, 50, 60, 70, 80]
    #fractions=[20,30]
    #fractions = [0.05, 0.025, 0.01, 0.005, 0.001, 0.00075, 0.0005, 0.00034, 0.00025, 0.0001]
    result_dict = read_in_decreasing_fractions(10, fractions, out_folder)
    for frac in fractions:
        print("Fraction: ", frac)
        print(get_out_string(result_dict[frac]))



def read_in_decreasing_fractions(num_runs, fractions, folder_name):
    path_template = f"../pval_tests/{folder_name}/"

    out_dict = {}
    for frac in fractions:
        out_dict[frac] =\
            {
                "acc_before": [],
                "calib_before": [],
                "f1_before": [],
                "ood_before": [],
                "iid_before": [],
                "acc_after": [],
                "calib_after": [],
                "f1_after": [],
                "ood_after": [],
                "iid_after": [],
            }
    for i in range(num_runs):

        mid_path = path_template + f"run_{i}/"
        for frac in fractions:
            loading_path = mid_path + f"frac_{frac}/"
            before_acc = np.load(loading_path + "before_acc.npy")
            before_calib = np.load(loading_path + "before_calib.npy")
            before_f1 = np.load(loading_path + "before_micro.npy")
            before_ood = np.load(loading_path + "before_train.npy")
            before_iid = np.load(loading_path + "before_train_iid.npy")
            after_acc = np.load(loading_path + "after_acc.npy")
            after_calib = np.load(loading_path + "after_calib.npy")
            after_f1 = np.load(loading_path + "after_micro.npy")
            after_ood = np.load(loading_path + "after_train.npy")
            after_iid = np.load(loading_path + "after_train_iid.npy")

            mid_dict =\
            {
                "acc_before": before_acc,
                "calib_before": before_calib,
                "f1_before": before_f1,
                "ood_before": before_ood,
                "iid_before": before_iid,
                "acc_after": after_acc,
                "calib_after": after_calib,
                "f1_after": after_f1,
                "ood_after": after_ood,
                "iid_after": after_iid,
            }



            for key, value in out_dict.get(frac, {}).items():
                out_dict[frac][key].append(mid_dict[key])
    for k,v in out_dict.items():
        for key, value in out_dict.get(k, {}).items():
            out_dict[k][key] = np.array(out_dict[k][key])

    return out_dict


def get_out_string(data_dict):
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
    return out_string





if __name__ == '__main__':
    main()