import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import torch
from p_value_runner import load_data
import time
from evaluator import train_model
from model_runner import run_calibrate_and_test
from model_setups.prototype_model_runner import load_features_dataset, get_model
from parameter_class import HyperparamStore, ModelParameters
from model_setups.prototype_model_runner import load_features_dataset, get_model
from utils.mahal_utils import Mahalanobis
# TODO: THIS IS VERY JANKY AND DOES NOT SHOW TRUE KDE
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
combined_path = "../../enc-vpn-uncertainty-class-repl/processed_data/decreasing_cal_stable_train/run0/frac_80/all_classes"
held_out_path = "../../enc-vpn-uncertainty-class-repl/processed_data/decreasing_cal_stable_train/run0/frac_80/no_ft"
#full_path = "../../enc-vpn-uncertainty-class-repl/processed_data/stable_cal_fraction/min_max_normalized/run0/frac_80"
model_to_load = "bdlpn"
params = HyperparamStore().get_model_params(model_to_load, "hidden")
# Example: simulate your data
n_class = 5  # Number of classes
n_examples = 100 # Large number of examples per class
held_out_data_test, combined_data_test = load_data(combined_path=combined_path, held_out_path=held_out_path,
                                                       params=params, split="test")
# initialize protonet
few_shot_classifier = get_model("bdcspn")
few_shot_classifier.load_state_dict(torch.load("../trained_models/bdcspn.pt"))
few_shot_classifier.backbone.eval()
with torch.no_grad():
    p_vals, acc, calib, micro, confusion, g_k = run_calibrate_and_test(few_shot_classifier,
                                                                       cal_loader=held_out_data_test,
                                                                       test_loader=held_out_data_test,
                                                                       bandwidth=0.1)




few_shot_classifier.backbone.eval()
#pnet.train()
z_s = None
x_q = None
dataloader = held_out_data_test
specific_class_index = dataloader.dataset.class_names.index("CHAT")
class_data_indices = (torch.tensor(dataloader.dataset.labels) == specific_class_index).nonzero(as_tuple=True)[0]
specific_samples = dataloader.dataset.embeddings.clone().detach()[class_data_indices]
xs = specific_samples
xq = specific_samples
n_support = len(xs[0])
n_support = n_cal = int(xs.shape[0]/5)
x = torch.cat([xs.view(n_class * n_support, 128),
               xq.view(n_class * n_cal, 128)], 0)
with torch.no_grad():
    z = few_shot_classifier.backbone.forward(x)
z_support = z[:n_class * n_support].view(n_class, n_support, 64)
z_cal = z[n_class * n_support:].view(n_class, n_cal, 64)
z_support = z_support.mean(0).expand(n_class, n_cal, 64)
rmd = Mahalanobis(z_support, 5, n_cal)  # Now compute mahalanobis # USE support to set mahal vars
# use calibrate to compute rel_mahal (if using sup, ood scores at test time will be higher
#m_k_rel = torch.min(rmd.relative_mahalanobis_distance(z_cal), dim=1).values.view(n_class, n_cal)
m_k_rel = rmd.relative_mahalanobis_distance(z_cal).view(n_class, n_cal, -1).min(dim=0).values.view(n_class, n_cal)
max_val = m_k_rel.max()
min_val = m_k_rel.min()
test_point = min_val-(max_val-min_val)*0.5
#print(np.mean([bw_selection.improved_sheather_jones(np.asarray(m_k_rel[i]).reshape(-1,1)) for i in range(n_class)]))
def get_kde(rel_mahalanobis, target_inds, n_way):
    class_kernel_densities = [0 for _ in range(n_way)]
    for idx in range(len(class_kernel_densities)):
        # TODO: Sometimes gives singular covariance matrix error, must handle
        class_kernel_densities[idx] = gaussian_kde(rel_mahalanobis[idx], bw_method=0.1)
    return class_kernel_densities
g_k = get_kde(m_k_rel, [], n_class)


# Define a common grid for KDE evaluation:

x_grid = np.linspace(test_point,max_val+(max_val-min_val)*0.541, 250000)

# Create subplots: one per class in a grid layout
n_cols = 2
n_rows = int(np.ceil(n_class / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 3), sharex=True, sharey=True)
axes = axes.flatten()
data = {f"Class {i}": np.random.normal(loc=0, scale=1.0, size=n_examples) for i in range(n_class)}
ind = 0
for ax, (label, samples) in zip(axes, data.items()):
    # Create the KDE object
    kde = g_k[ind]
    ind += 1
    # Evaluate the density on the grid
    density = kde(x_grid)

    # Plot the KDE curve
    plt.xlim(min_val+(max_val-min_val)*0.1, max_val+(max_val-min_val)*0.541)
    ax.plot(x_grid, density, label=label)
    ax.fill_between(x_grid, density, alpha=0.5)
    ax.set_title(label)
    ax.legend()

# Remove any unused subplots
for i in range(len(data), len(axes)):
    fig.delaxes(axes[i])

# INTEGRATION TIME TESTING
time1 = time.time()
#print(f"quad(g_k[3].pdf, a=min_val, b=max_val, limit=50000), prints: {quad(g_k[3].pdf, a=min_val-0.05, b=max_val+0.05, limit=500000, epsabs=1e-15, epsrel=1e-15)}")
#print("quad time: ", time.time() - time1)
#print(f"quad(g_k[3].pdf, a=-1, b=1, limit=50000), prints: {quad(g_k[3].pdf, a=-1, b=1, limit=500)}")
#print(f"quad(g_k[3].pdf, a=-np.inf, b=np.inf, limit=50000), prints: {quad(g_k[3].pdf, a=-np.inf, b=np.inf, limit=500)}")
time2= time.time()
print([np.trapz(g_k[i].pdf(x_grid), x_grid, dx=0.00001) for i in range(len(g_k))])
print("TRAPZ TIME", time.time()-time2)
time1 = time.time()
print(torch.trapz(y=torch.tensor(g_k[3].pdf(x_grid)), x=torch.tensor(x_grid)).item())
print("TORCH TIME", time.time()-time1)

plt.tight_layout()
plt.show()