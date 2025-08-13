import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import torch
bws = [0.05, 0.025, 0.01, 0.005, 0.001, 0.00075, 0.0005, 0.00034, 0.00025, 0.0001]
bws = [20, 30, 40, 50, 60, 70, 80]
#bws = [20, 30, 40, 50]#,30, 40,50,60,70,80]
def main():
    # Sup to make mahal seems to be slightly better.
    load_dir = "bdlpn_1query_0p05bw_proto_to_make_mahal_spotify_10run_test_jun_25_increasing_query"
    ood_before_medians, ood_after_medians, iid_before_medians, iid_after_medians = [], [], [], []
    for bw in bws:
        runs = 10
        before, after = read_in_values(load_dir, runs, bw)
        iid_before, iid_after = read_iid_values(load_dir, runs, bw)
        df_before, df_after = prepare_df(before, after)
        iid_df_before, iid_df_after = prepare_df(iid_before, iid_after)
        runs = 10
        mid_df1 = average_all_ood(df_before, runs)
        mid_df2 = average_all_ood(df_after, runs)
        iid_mid_df1 = average_all_ood(iid_df_before, runs)
        iid_mid_df2 = average_all_ood(iid_df_after, runs)
        #df_before = df_before[df_before["Run"] == 3]
        #df_after = df_after[df_after["Run"] == 3]
        df_before["avg_OOD"] = mid_df1
        df_after["avg_OOD"] = mid_df2
        iid_df_before["avg_OOD"] = iid_mid_df1
        iid_df_after["avg_OOD"] = iid_mid_df2


        data_to_table = "avg_OOD" # NOTE ALL TABLES ARE AVG OOD MEDIANS, OOD SCORE MEDIAN ARE SLIGHTLY DIFFERENT
        ood_before_medians.append((df_before["OOD_score"].median(), df_before["OOD_score"].std()))
        ood_after_medians.append((df_after["OOD_score"].median(), df_after["OOD_score"].std()))
        iid_before_medians.append((iid_df_before["OOD_score"].median(), iid_df_before["OOD_score"].std()))
        iid_after_medians.append((iid_df_after["OOD_score"].median(), iid_df_after["OOD_score"].std()))


        df = pd.concat([df_before, df_after])
        iid_df = pd.concat([iid_df_before, iid_df_after])
        plt.figure(figsize=(14, 7))  # Increase figure size
        sns.violinplot(
            data=df,
            x="Condition",
            y="OOD_score",
            hue="Condition",
            split=False,  # Change to False for better visibility
            dodge=True,  # Separate distributions
            inner="box",  # Adds a boxplot inside each violin
            linewidth=1.2,  # Make outlines sharper
            palette="Set2",
            bw_method=.14,
            width=0.75,
            density_norm="area",
            fill=False,
            cut=-0.1  # Avoid long tails from outliers
        )
        plt.title(f"\'SPOTIFY\'OOD Data w/ BDLPN), Calibration Split = {int(bw/10)-1}")
        plt.legend(title="Condition", loc="lower left")
        plt.grid(axis="y", linestyle="--", alpha=0.7)  # Add light gridlines for readability
        plt.xticks(fontsize=12)  # Make axis labels larger
        plt.yticks(fontsize=12)
        plt.show()


    print("OOD BEFORE")
    print(["{:.2f}+-{:.2f}".format(x[0], x[1]) for x in ood_before_medians])
    print("OOD AFTER")
    print(["{:.2f}+-{:.2f}".format(x[0], x[1]) for x in ood_after_medians])
    print("IID BEFORE")
    print(["{:.2f}+-{:.2f}".format(x[0], x[1]) for x in iid_before_medians])
    print("IID AFTER")
    print(["{:.2f}+-{:.2f}".format(x[0], x[1]) for x in iid_after_medians])


def read_in_values(load_dir, run_count, bws=None):
    run = 0
    ood_before, ood_after = [], []

    while run < run_count:
        if bws is not None:
            path_template = f"../pval_tests/{load_dir}/run_{run}/frac_{bws}/"
        else:
            path_template = f"../pval_tests/{load_dir}/run_{run}/"
        before = path_template+"before_train.npy"
        after = path_template+"after_train.npy"
        ood_before.append(np.load(before))
        ood_after.append(np.load(after))
        run += 1
    return np.array(ood_before).squeeze(), np.array(ood_after).squeeze()


def read_iid_values(load_dir, run_count, bws=None):
    run = 0
    iid_before, iid_after = [], []
    while run < run_count:
        if bws is not None:
            path_template = f"../pval_tests/{load_dir}/run_{run}/frac_{bws}/"
        else:
            path_template = f"../pval_tests/{load_dir}/run_{run}/"
        before = path_template+"before_train_iid.npy"
        after = path_template+"after_train_iid.npy"
        iid_before.append(np.load(before))
        iid_after.append(np.load(after))
        run += 1
    return np.array(iid_before).squeeze(), np.array(iid_after).squeeze()

def prepare_df(before, after):
    # Flatten data for plotting
    def prepare_dataframe(scores, label):
        if len(scores.shape) == 1:
            scores = scores.reshape(1, -1)

        runs, samples = scores.shape
        data = []
        for run in range(runs):
            for sample in range(samples):
                data.append([run, scores[run, sample], label])
        return pd.DataFrame(data, columns=["Run", "OOD_score", "Condition"])

    # Prepare data
    df_before = prepare_dataframe(before, "Before Retrain")
    df_after = prepare_dataframe(after, "After Retrain")

    # Combine into a single DataFrame
    return df_before, df_after

def average_all_ood(df, num_runs):

    average_ood_scores = []
    size=int(len(df["OOD_score"])/num_runs)
    print(len(df["OOD_score"])/num_runs)
    for i in range(int(len(df["OOD_score"])/num_runs)):
        new_ood_score = 0
        for j in range(num_runs):
            new_ood_score += df[df["Run"]==0]["OOD_score"].iloc[i]
        average_ood_scores.append(new_ood_score/num_runs)
    average_ood_scores = np.array(torch.tensor(average_ood_scores).expand(num_runs, size).reshape(size*num_runs))
    return average_ood_scores


if __name__ == "__main__":
    main()