import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 1) Load posterior summary (means, stds, and true values)
summary_file = '../samples/posterior_summary.csv'
df = pd.read_csv(summary_file)

# 2) Define parameters and output directory
param_cols = ['N', 'Cab', 'Car', 'Cbrown', 'Cw', 'Cm']
output_dir = 'uncertainty_vs_residual'
os.makedirs(output_dir, exist_ok=True)

# 3) For each parameter, plot |residual| vs uncertainty
for param in param_cols:
    true_col = f'{param}_true'
    mean_col = f'{param}_mean'
    std_col  = f'{param}_std'

    # Compute residuals and their absolute value
    residuals = df[mean_col] - df[true_col]
    abs_resid = np.abs(residuals.values)
    uncertainty = df[std_col].values

    # Scatter plot of abs(residual) vs posterior std
    plt.figure(figsize=(6, 6))
    plt.scatter(abs_resid, uncertainty, alpha=0.6, label='Samples')

    # Fit and plot a linear trend
    coeffs = np.polyfit(abs_resid, uncertainty, 1)
    trend = np.poly1d(coeffs)
    x_vals = np.linspace(abs_resid.min(), abs_resid.max(), 100)
    plt.plot(x_vals, trend(x_vals), color='red', linestyle='--',
             label=f'Trend (slope={coeffs[0]:.2f})')

    plt.title(f'Uncertainty vs |Residual| for {param}')
    plt.xlabel('|Residual|')
    plt.ylabel('Posterior Std')
    plt.legend()
    plt.tight_layout()

    # Save figure
    plt.savefig(f'{output_dir}/unc_vs_absresid_{param}.png', dpi=300)
    plt.close()

print(f"Saved uncertainty vs |residual| plots to '{output_dir}'")