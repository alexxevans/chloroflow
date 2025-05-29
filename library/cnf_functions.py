import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import chi2

# Updated CNF data‐prep and plotting to include LAI, ANT as inferred parameters and additional conditioning inputs

def load_and_prepare_data(forward_csv):
    """
    Reads CSV containing:
      - 8 target parameters: ['N','Cab','Car','Cbrown','Cw','Cm','lai','ant']
      - reflectance bands: ['Refl_*.nm']
      - 5 conditioning inputs: ['tts','tto','psi','hspot','lidfa']
    Normalises all columns to zero mean and unit variance.
    Returns:
      data_to_learn: np.ndarray [N x 8]
      conditioning_data: np.ndarray [N x (bands+5)]
      normalisation_params: dict mapping col->(mean,std)
    """
    df = pd.read_csv(forward_csv)

    # Define column groups
    param_cols = ['N', 'Cab', 'Car', 'Cbrown', 'Cw', 'Cm', 'lai']
    refl_cols = [c for c in df.columns if c.startswith('Refl_')]
    data_cols = ['tts', 'tto', 'psi', 'hspot', 'lidfa']

    # Check presence
    req = set(param_cols + refl_cols + data_cols)
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # Clean
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=param_cols + refl_cols + data_cols, inplace=True)

    # Normalise
    norm = {}
    for c in param_cols + refl_cols + data_cols:
        mu, sigma = df[c].mean(), df[c].std()
        norm[c] = (mu, sigma)
        df[c] = (df[c] - mu) / sigma

    # Extract arrays
    data_to_learn = df[param_cols].values
    conditioning_data = df[refl_cols + data_cols].values
    return data_to_learn, conditioning_data, norm


def split_data(data_to_learn, conditioning_data, test_size=0.3, random_state=42):
    """
    Splits into train, validation, test.
    Returns train_data, valid_data, test_data, train_cond, valid_cond, test_cond
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        data_to_learn, conditioning_data,
        test_size=test_size, random_state=random_state
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5, random_state=random_state
    )
    return X_train, X_valid, X_test, y_train, y_valid, y_test


def convert_to_tensors(device, train_data, valid_data, test_data,
                       train_cond, valid_cond, test_cond):
    """Converts arrays to torch tensors."""
    train_data = torch.tensor(train_data, dtype=torch.float32).to(device)
    valid_data = torch.tensor(valid_data, dtype=torch.float32).to(device)
    test_data  = torch.tensor(test_data,  dtype=torch.float32).to(device)
    train_cond = torch.tensor(train_cond, dtype=torch.float32).to(device)
    valid_cond = torch.tensor(valid_cond, dtype=torch.float32).to(device)
    test_cond  = torch.tensor(test_cond,  dtype=torch.float32).to(device)
    return train_data, valid_data, test_data, train_cond, valid_cond, test_cond


def plot_initial_distribution(data_to_learn, conditioning_data,
                              wavelengths=None, n_samples=4, seed=42):
    """
    Show representative samples:
      (a) scatter of true parameters (8 dims),
      (b) reflectance line,
      plus annotation of the 5 data inputs.
    """
    np.random.seed(seed)
    ids = np.random.choice(data_to_learn.shape[0], n_samples, replace=False)
    param_labels = ['N','Cab','Car','Cbrown','Cw','Cm','lai']
    total_feats = conditioning_data.shape[1]
    data_count  = 5
    refl_count  = total_feats - data_count
    if wavelengths is None:
        wavelengths = np.arange(refl_count)
    fig, axes = plt.subplots(2, n_samples, figsize=(4*n_samples, 6))
    for col, idx in enumerate(ids):
        ax_p = axes[0, col]; ax_r = axes[1, col]
        # param scatter
        ax_p.scatter(param_labels, data_to_learn[idx], c='black')
        ax_p.set_xticks(range(len(param_labels)))
        ax_p.set_xticklabels(param_labels, rotation=45)
        ax_p.set_title(f"Sample \\#{idx} Params")
        # reflectance
        refl = conditioning_data[idx, :refl_count]
        ax_r.plot(wavelengths, refl, c='black')
        ax_r.set_title(f"Sample \\#{idx} Reflectance")
        ax_r.set_xlabel('Wavelength idx')
        # data annotation
        data_vals = conditioning_data[idx, refl_count:]
        txt = (
            f"tts={data_vals[0]:.2f}\n"
            f"tto={data_vals[1]:.2f}\n"
            f"psi={data_vals[2]:.2f}\n"
            f"hspot={data_vals[3]:.2f}\n"
            f"lidfa={data_vals[4]:.2f}"
        )
        ax_r.text(0.01, -0.3, txt, transform=ax_r.transAxes,
                  fontsize=8, va='top')
    plt.tight_layout(pad=2)
    plt.savefig('./plots/CNF_initial_samples.png', dpi=300)
    plt.show()


def plot_trained_distribution(test_data, test_cond, model,
                              wavelengths=None, posterior_samples=100,
                              seed=0, n_display=4):
    """
    For n_display random points, plot:
      - errorbar (mean±std) vs true over 8 params,
      - reflectance line + annotation of 5 data inputs.
    """
    import torch, numpy as np, matplotlib.pyplot as plt
    torch.manual_seed(seed); np.random.seed(seed)
    N, D = test_data.shape
    total_feats = test_cond.shape[1]
    data_count  = 5
    refl_count  = total_feats - data_count
    ids = np.random.choice(N, min(n_display, N), replace=False)
    # sample posterior
    samples = np.stack([
        model.sample(N, context=test_cond)[0].detach().cpu().numpy()
        for _ in range(posterior_samples)
    ], axis=0)
    means = samples.mean(axis=0); stds = samples.std(axis=0)
    param_labels = ['N','Cab','Car','Cbrown','Cw','Cm','lai']
    if wavelengths is None:
        wavelengths = np.arange(refl_count)
    fig, axes = plt.subplots(2, len(ids), figsize=(4*len(ids), 6))
    for j, idx in enumerate(ids):
        ax_p = axes[0, j]; ax_r = axes[1, j]
        # params
        ax_p.errorbar(np.arange(D), means[idx], yerr=stds[idx], fmt='o', color='red', ecolor='salmon')
        ax_p.scatter(np.arange(D), test_data[idx].detach().cpu().numpy(), marker='x', c='black')
        ax_p.set_xticks(np.arange(D)); ax_p.set_xticklabels(param_labels, rotation=45)
        ax_p.set_title(f"Posterior Params \\#{idx}")
        # reflectance
        refl = test_cond[idx, :refl_count].detach().cpu().numpy()
        ax_r.plot(wavelengths, refl, c='black')
        ax_r.set_xlabel('Wavelength idx')
        ax_r.set_title(f"Reflectance \\#{idx}")
        # data annotation
        data_vals = test_cond[idx, refl_count:].detach().cpu().numpy()
        txt = (
            f"tts={data_vals[0]:.2f}\n"
            f"tto={data_vals[1]:.2f}\n"
            f"psi={data_vals[2]:.2f}\n"
            f"hspot={data_vals[3]:.2f}\n"
            f"lidfa={data_vals[4]:.2f}"
        )
        ax_r.text(0.01, -0.3, txt, transform=ax_r.transAxes,
                  fontsize=8, va='top')
    plt.tight_layout(pad=2)
    plt.savefig('./plots/CNF_uncertainty.png', dpi=300)
    plt.show()


def evaluate_model(model, test_data, test_cond, k=1.96, posterior_samples=100):
    """
    Computes avg LL, NLL, perplexity, RMSE, R^2, and coverage for 8 params.
    """
    model.eval()
    with torch.no_grad():
        lp = model.log_prob(test_data, test_cond)
        avg_ll = lp.mean().item(); nll = -avg_ll
        perp = float(torch.exp(-lp.mean()))
        # sample posterior
        samples = np.stack([
            model.sample(test_data.shape[0], context=test_cond)[0].detach().cpu().numpy()
            for _ in range(posterior_samples)
        ], axis=0)
        mu = samples.mean(axis=0); sigma = samples.std(axis=0)
        true = test_data.detach().cpu().numpy()
        # RMSE and R2
        rmse_per = np.sqrt(np.mean((mu-true)**2, axis=0))
        rmse_all = np.sqrt(np.mean((mu-true)**2))
        mean_true = true.mean(axis=0)
        ss_res = ((mu-true)**2).sum(axis=0)
        ss_tot = ((true-mean_true)**2).sum(axis=0)
        r2_per = 1 - ss_res/ss_tot
        r2_all = 1 - ss_res.sum()/((true-true.mean())**2).sum()
        # coverage
        lower = mu - k*sigma; upper = mu + k*sigma
        cov = ((true>=lower)&(true<=upper)).mean(axis=0)
    # print
    labels = ['N','Cab','Car','Cbrown','Cw','Cm','lai']
    print(f"Avg LL: {avg_ll:.4f}, NLL: {nll:.4f}, Perp: {perp:.4f}")
    print("RMSE per param:")
    for l,v in zip(labels,rmse_per): print(f"  {l}: {v:.4f}")
    print(f"Overall RMSE: {rmse_all:.4f}")
    print("R^2 per param:")
    for l,v in zip(labels,r2_per): print(f"  {l}: {v:.4f}")
    print(f"Overall R^2: {r2_all:.4f}")
    print(f"Coverage ±{k}σ:")
    for l,v in zip(labels,cov): print(f"  {l}: {v*100:.1f}%")


def plot_training_and_validation_loss(train_loss_hist, val_loss_hist, lrs):
    # Prepare the figure and axes
    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))

    ax1.set_box_aspect(0.675)  # Set the aspect ratio (height/width)

    # Plot the training and validation loss on the primary y-axis
    ax1.plot(range(1, len(train_loss_hist) + 1), train_loss_hist, label='Training Loss', color='deepskyblue')
    ax1.plot(range(1, len(val_loss_hist) + 1), val_loss_hist, label='Validation Loss', color='orange')
    ax1.set_xscale('log')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('KLD')
    ax1.legend(loc='lower left', frameon=False)
    ax1.spines['top'].set_visible(False)

    # Create a twin y-axis to plot the learning rate
    ax2 = ax1.twinx()
    ax2.set_box_aspect(0.675)
    ax2.plot(range(1, len(lrs) + 1), lrs, label='Learning Rate', color='black', linestyle='--', alpha=0.6)
    ax2.set_yscale('log')
    ax2.set_ylabel('Learning Rate')
    ax2.spines['top'].set_visible(False)
    ax2.legend(loc='upper right', frameon=False)

    plt.tight_layout(w_pad=8.0)
    plt.savefig('./plots/CNF_loss.png', dpi=300)
    plt.show()