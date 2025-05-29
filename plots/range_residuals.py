import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 1) Load posterior summary (means, stds, and true values)
summary_file = '../samples/posterior_summary.csv'
df = pd.read_csv(summary_file)

# 2) Define parameters and output directory
param_cols = ['N', 'Cab', 'Car', 'Cbrown', 'Cw', 'Cm']
residuals_dir = 'residuals/scatter'
binned_dir    = 'residuals/binned'
os.makedirs(residuals_dir, exist_ok=True)
os.makedirs(binned_dir, exist_ok=True)

# 3) Plot raw residual scatter and separate binned uncertainty for each parameter
n_bins = 10  # number of bins for |residual|
for param in param_cols:
    true_col = f'{param}_true'
    mean_col = f'{param}_mean'
    std_col = f'{param}_std'

    # compute residuals and absolute residuals
    residuals = df[mean_col] - df[true_col]
    abs_resid = np.abs(residuals.values)
    std_vals = df[std_col].values
    true_vals = df[true_col].values

    # --- Plot 1: raw residual scatter ---
    plt.figure(figsize=(6,6))
    plt.scatter(true_vals, residuals, alpha=0.6)
    plt.axhline(0, color='gray', linewidth=0.8)
    plt.title(f'Residuals for {param}')
    plt.xlabel(f'True {param}')
    plt.ylabel('Residual (mean − true)')
    plt.tight_layout()
    plt.savefig(f'{residuals_dir}/residuals_scatter_{param}.png', dpi=300)
    plt.close()

        # --- Prepare bins over TRUE parameter values ---
    # instead of binning by residual, bin by the true parameter range
    param_min, param_max = true_vals.min(), true_vals.max()
    bins = np.linspace(param_min, param_max, n_bins+1)
    bin_centers = 0.5*(bins[:-1] + bins[1:])
    mean_unc = []
    for i in range(n_bins):
        mask = (true_vals >= bins[i]) & (true_vals < bins[i+1])
        if mask.any():
            mean_unc.append(std_vals[mask].mean())
        else:
            mean_unc.append(np.nan)

    # --- Plot 2: binned uncertainty vs TRUE parameter ---
    plt.figure(figsize=(6,6))
    plt.plot(bin_centers, mean_unc, marker='o', linestyle='-', color='red')
    plt.title(f'Mean Posterior STD vs True {param} (binned)')
    plt.xlabel(f'True {param}')
    plt.ylabel('Mean Posterior STD')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{binned_dir}/binned_uncertainty_true_{param}.png', dpi=300)
    plt.close()

print(f"Saved scatter plots in '{residuals_dir}' and binned plots in '{binned_dir}'")
