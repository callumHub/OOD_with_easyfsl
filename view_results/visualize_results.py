import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
frac_values = [0.8, 0.7, 0.6, 0.5]
def main():
    path = "./k_fold_eval_outs/bdcspn/"
    f1_data = np.load(path+"all_macros.npy")
    calib_data = np.load(path+"all_calibs.npy")
    acc_data = np.load(path+"all_accs.npy")
    micro_f1_against_data_frac(f1_data, "5-fold Macro F1")
    micro_f1_against_data_frac(acc_data, "5-fold Accuracy")
    ece_against_data_frac(calib_data, "5-fold Ece")


def micro_f1_against_data_frac(micro_f1_scores, title=None):
    general_plot(micro_f1_scores, title)

def ece_against_data_frac(ece_scores, title=None):
    general_plot(ece_scores, title)

def show_confusion_matrix(confusion_matrix, data_frac=None):
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

    if data_frac is None:
        for i in range(len(frac_values)):
            #ax = sns.heatmap(confusion_matrix, )
            row_sums = confusion_matrix.sum(axis=1, keepdims=True)
            normalized_cm = confusion_matrix / row_sums

            # Convert to DataFrame for seaborn
            plot_df = pd.DataFrame(data=normalized_cm, columns=class_map.values(), index=class_map.values())

            # Plot heatmap
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(plot_df, annot=True, fmt=".0%", cmap="magma", yticklabels=class_map.values(),
                        xticklabels=class_map.values(), ax=ax)
            class_totals = confusion_matrix.sum(axis=1)
            total_samples = class_totals.sum()
            class_proportions = {class_map[i]: (class_totals[i] / total_samples) * 100 for i in
                                 range(len(class_totals))}

            # Create a custom legend for class proportions
            legend_texts = [f"Class {label}: {prop:.1f}%" for label, prop in class_proportions.items()]
            legend = "\n".join(legend_texts)

            # Add text as an annotation outside the plot
            plt.gcf().text(1.05, 0.5, legend, fontsize=12, verticalalignment='center')

            plt.title("Confusion Matrix with Class Proportions")
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.show()
    else:
        index = int((0.8-data_frac)*10)
        plot_df = pd.DataFrame(data=confusion_matrix[index]/confusion_matrix[index,0,:].sum(), columns=class_map.values(), index=class_map.values())
        ax = sns.heatmap(plot_df, annot=True, fmt=".0%", yticklabels=class_map.values(), xticklabels=True)
        cbar = ax.collections[0].colorbar
        cbar.set_label("Class Balance", rotation=270, labelpad=15)
        plt.show()

def general_plot(data, title):
    axis = 0
    data = np.asarray(data)
    # TODO: replace y=data with y=mean when k fold running
    means = np.mean(data, axis=axis).flatten()
    std_errors = np.std(data, axis=axis, ddof=1) / np.sqrt(data.shape[axis])

    plt.figure(figsize=(8, 6))
    sns.lineplot(x=frac_values, y=means, marker='o', label="Mean of test data")
    plt.errorbar(frac_values, means, yerr=std_errors.flatten(), fmt='o', capsize=5, label="Std Error")

    plt.xlabel("Fraction of Data Used for Training")
    plt.ylabel(title.split(" ")[0])
    plt.title(title)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()