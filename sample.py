import torch
import numpy as np
import pandas as pd
import os
import pickle

from library.cnf_functions import load_and_prepare_data, split_data
import normflows as nf

# 1) Build CNF model architecture (must match training)
def create_cnf_model(latent_size, conditioning_size):
    K = 8
    hidden_units = 128
    hidden_layers = 4
    flows = []
    for i in range(K):
        flows += [nf.flows.CoupledRationalQuadraticSpline(
            latent_size, hidden_layers, hidden_units,
            num_context_channels=conditioning_size)]
        flows += [nf.flows.LULinearPermute(latent_size)]
    q0 = nf.distributions.DiagGaussian(latent_size, trainable=False)
    model = nf.ConditionalNormalizingFlow(q0, flows, None)
    return model

# 2) Load data and normalization params
data_csv = 'data/forward_data.csv'
data_to_learn, conditioning_data, norm_params = load_and_prepare_data(data_csv)

# 3) Split into train/val/test (we'll summarise on test)
train_d, valid_d, test_d, train_c, valid_c, test_c = split_data(data_to_learn, conditioning_data)

# 4) Load trained CNF model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
latent_size = test_d.shape[1]
conditioning_size = test_c.shape[1]
cnf = create_cnf_model(latent_size, conditioning_size).to(device)
cnf.load_state_dict(torch.load('models/CNF.pth', map_location=device))
cnf.eval()

# 5) Function to sample and compute mean/std
def summarise_posterior(context, n_samples=1000, batch_size=256):
    # prepare context tensor
    ctx = torch.tensor(context, dtype=torch.float32).to(device).unsqueeze(0)
    ctx_all = ctx.repeat(n_samples, 1)
    # collect samples in batches
    collected = []
    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            batch_ctx = ctx_all[start:start+batch_size]
            s, _ = cnf.sample(batch_ctx.size(0), context=batch_ctx)
            collected.append(s.cpu().numpy())
    samples = np.vstack(collected)  # shape [n_samples, latent_size]
    # compute mean and std
    mean = samples.mean(axis=0)
    std = samples.std(axis=0)
    return mean, std

# 6) Generate summaries for each test point
enabled_points = len(test_d)
posterior_samples = 100
param_cols = ['N', 'Cab', 'Car', 'Cbrown', 'Cw', 'Cm']

# Prepare output DataFrame
data = []
for idx, (context, true_vals) in enumerate(zip(test_c, test_d)):
    print(f"summarising posterior for test point {idx}...")
    mean, std = summarise_posterior(context, n_samples=posterior_samples)
    # unnormalise posterior mean and std
    mean_un = []
    std_un = []
    for col, m_val, s_val in zip(param_cols, mean, std):
        m0, s0 = norm_params[col]
        mean_un.append(m_val * s0 + m0)
        std_un.append(s_val * s0)
    # unnormalise true values
    true_un = []
    for col, true_norm in zip(param_cols, true_vals):
        m0, s0 = norm_params[col]
        true_un.append(true_norm * s0 + m0)
    # assemble row dict
    row = {'index': idx}
    for col, mu, sd, true_val in zip(param_cols, mean_un, std_un, true_un):
        row[f'{col}_mean'] = mu
        row[f'{col}_std'] = sd
        row[f'{col}_true'] = true_val
    data.append(row)

# Create DataFrame and save
df_summary = pd.DataFrame(data)
output_dir = 'samples'
os.makedirs(output_dir, exist_ok=True)
summary_file = os.path.join(output_dir, 'posterior_summary.csv')
df_summary.to_csv(summary_file, index=False)
print(f"Saved posterior summary (mean & std) for {enabled_points} points to {summary_file}")