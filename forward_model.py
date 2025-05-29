import prosail
import numpy as np
import os
from scipy.stats import qmc

# 1) Define parameter (theta) and data (X) ranges
param_ranges = {
    'N':      (0.8,   2.5),   # leaf structure
    'Cab':    (0.0,  90.0),   # chlorophyll a+b
    'Car':    (0.0,  20.0),   # carotenoids
    'Cbrown': (0.0,   1.0),   # brown pigments
    'Cw':     (0.001, 0.02),  # equivalent water thickness
    'Cm':     (0.001, 0.02),  # dry matter content
    'lai':    (0.0,  10.0),   # leaf area index (inferred)
}
data_ranges = {
    'tts':   (0.0, 90.0),     # solar zenith angle (°)
    'tto':   (0.0, 90.0),     # view zenith angle (°)
    'psi':   (0.0, 90.0),     # relative azimuth (°)
    'hspot': (0.0, 0.15),     # hotspot parameter (m⁻¹)
    'lidfa': (0.5, 2.0),      # leaf angle distribution factor
}

# Combine into one sampling dictionary
names = list(param_ranges.keys()) + list(data_ranges.keys())
mins  = np.array([param_ranges[n][0] for n in param_ranges] +
                 [data_ranges[n][0] for n in data_ranges])
maxs  = np.array([param_ranges[n][1] for n in param_ranges] +
                 [data_ranges[n][1] for n in data_ranges])

# 2) Sobol sampling over all dims
n_samples  = 1048576
dim        = len(names)
sampler    = qmc.Sobol(d=dim, scramble=True)
unit_samps = sampler.random(n=n_samples)
scaled_samps = qmc.scale(unit_samps, mins, maxs)
print(f"Running {n_samples} Sobol forward runs over dims {names}...")

# 3) Fixed soil reflectance
soil0 = np.zeros(2101)  # flat soil reflectance
ant = 0

# 4) Precompute target wavelength indices (120bands)
full_wl = np.arange(400, 2501)
target  = np.linspace(400, 2500, 120)
idx     = [np.abs(full_wl - wl).argmin() for wl in target]

# 5) Noise: simple Gaussian per band
noise_std = 0  # 1% reflectance noise

# 6) Run PROSAIL and add noise
out_rows = []
for theta in scaled_samps:
    # unpack theta (8) and data (5)
    N, Cab, Car, Cbrown, Cw, Cm, lai, \
    tts, tto, psi, hspot, lidfa = theta

    # forward model call (PROSPECT-D + SAIL)
    R2101 = prosail.run_prosail(
        N, Cab, Car, Cbrown, Cw, Cm,
        lai, lidfa, hspot,
        tts, tto, psi,
        ant,
        rsoil0=soil0
    )
    # subset reflectance bands
    Rsub = R2101[idx]

    # add Gaussian noise and clip
    noise = np.random.normal(0.0, noise_std, size=Rsub.shape)
    Rnoisy = np.clip(Rsub + noise, 0.0, 1.0)

    # record parameters + noisy reflectance
    out_rows.append(list(theta) + Rnoisy.tolist())

# 7) Save to CSV
oheader = names + [f'Refl_{int(wl)}nm' for wl in target]
out_arr = np.array(out_rows, dtype=float)
os.makedirs('data', exist_ok=True)
np.savetxt(
    'data/forward_data.csv',
    out_arr,
    delimiter=',',
    header=','.join(oheader),
    comments=''
)
print('Saved → data/forward_data.csv with all params sampled')