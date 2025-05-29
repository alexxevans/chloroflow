import prosail
from scipy.optimize import minimize
import numpy as np
import os

class inversion_of_prosail:
    def __init__(self, params_initial, method, bounds):
        # Store starting guess, optimizer method, and parameter bounds
        self.params_initial = params_initial
        self.method = method
        self.bounds = bounds

    def merit_function(self, params, meas):
        """
        Given a candidate parameter vector `params` and measured spectrum `meas`,
        run PROSAIL to get a synthetic reflectance and compute a goodness‐of‐fit.
        """
        # (Optional) load a soil reflectance file if you want to include real soil
        soil = np.loadtxt('materials/soil_prosail.dat')

        # Run the PROSAIL forward model: returns simulated reflectance at each band
        prosail_output = prosail.run_prosail(
            params[0], params[1], params[2], params[3],
            params[4], params[5],
            5,     # leaf structure parameter (N)
            -0.35, # solar/view geometry
            0,     # hotspot
            50,    # solar zenith angle
            0,     # relative azimuth
            45,    # view zenith angle
            0,     # other angle
            rsoil0=np.zeros(2101)  # here using flat soil spectrum
            # to instead use real soil:
            # rsoil=soil[:,1], psoil=0.5
        )

        # Merit = sum of squared difference normalized by measurement
        return np.sum(((meas - prosail_output) ** 2) / meas)

    def inversion(self, data):
        """
        Run a bounded minimisation (e.g. TNC) of the merit_function,
        starting from self.params_initial and returning the best‐fit params.
        """
        result = minimize(
            self.merit_function,
            self.params_initial,
            args=data,            # pass `data` as the `meas` argument
            method=self.method,
            bounds=self.bounds
        )
        return result.x  # the optimized parameter vector

# ---- Define parameter ranges and prior stats ----
std = 0.1
parameters = {
    'N':      {'limits': (0.8, 2.5), 'stats': (1.5, std * 1.5)},
    'Cab':    {'limits': (0, 90),   'stats': (40,  std * 40)},
    'Car':    {'limits': (0, 20),   'stats': (8,   std * 8)},
    'Cbrown': {'limits': (0, 1),    'stats': (0.1, std * 0.1)},
    'Cw':     {'limits': (0.001, 0.02), 'stats': (0.01, std * 0.01)},
    'Cm':     {'limits': (0.001, 0.02), 'stats': (0.008, std * 0.008)}
}

# Extract labels, bounds and prior‐distribution stats
labels = np.array(list(parameters.keys()))
limits = np.array([v['limits'] for v in parameters.values()])
stats  = np.array([v['stats']  for v in parameters.values()])

# Define the chlorophyll levels (used as our “independent variable”)
params = ['25','30','35','40','45','50','55','60','65','70','75','80']
chl_levels = np.array([int(x) for x in params])
num_simulations = 100

# Which viewing angles to process
deg_angles = ['0deg']  # could add '-10deg','5deg', etc.
sza = 50               # solar zenith angle

# Prepare to collect summary results: first column is the true chl_levels
summary_results = [ chl_levels.reshape(-1,1) ]
header_columns  = ['Chl']

# ---- Loop over angles and pixel sizes ----
for deg in deg_angles:
    for pxl in [1, 3, 5]:
        # Build file paths to the raw reflectances and their uncertainties
        refl_path = f'outputs/wheat/sza{sza}/{deg}/refls/11_refls_chl_{pxl}pxl_{deg}.txt'
        unc_path  = f'outputs/wheat/sza{sza}/{deg}/refls/11_refls_chl_{pxl}pxl_{deg}_unc.txt'

        # Load: each file has shape [bands × n_levels] (columns correspond to each chl level)
        reflectance_data = np.loadtxt(refl_path, delimiter=',')
        uncertainty_data = np.loadtxt(unc_path,  delimiter=',')

        # Prepare array to hold all simulated inversion outputs
        all_chl_estimates = np.zeros((num_simulations, len(params)))

        # ---- Monte‐Carlo loop: perturb data & invert repeatedly ----
        for sim in range(num_simulations):
            chl_output = []

            # For each chlorophyll level (each column in the loaded files)…
            for i, level in enumerate(params):
                # Extract the measured spectrum and its uncertainty for this level
                refl = reflectance_data[:, i]    # true reflectance vs band
                unc  = uncertainty_data[:, i]    # per‐band uncertainty

                # Perturb reflectance within its ±uncertainty plus a small systematic noise
                refl_perturbed = (
                    refl
                    + np.random.uniform(-unc, unc)            # random within ±unc
                    + refl * np.random.uniform(-0.035, 0.035) # ±3.5% systematic
                )

                # Draw a random initial guess for PROSAIL inversion
                initial_parameters = [
                    # for Cab: uniform over its bounds
                    np.random.uniform(limits[j][0], limits[j][1])
                    if labels[j] == 'Cab'
                    # for all others: sample from a Gaussian prior
                    else np.random.normal(stats[j][0], stats[j][1])
                    for j in range(len(labels))
                ]

                # Perform inversion to recover parameters; extract Cab (index 1)
                inv = inversion_of_prosail(initial_parameters, 'TNC', limits)
                best_params = inv.inversion(refl_perturbed)
                chl_output.append(best_params[1])

            # Store this simulation’s recovered Cab values (one per true level)
            all_chl_estimates[sim, :] = chl_output

        # ---- Save full inversion results to disk ----
        out_file = f"outputs/wheat/sza{sza}/{deg}/inversion/inversion_results_{pxl}pxl_{deg}_rsoil0.txt"
        header   = ",".join(params)
        np.savetxt(out_file, all_chl_estimates,
                   delimiter=",", header=header, comments='')

        # ---- Compute mean & standard‐error of the recovered Cab across sims ----
        chl_means = np.mean(all_chl_estimates, axis=0)
        chl_stds  = (np.std(all_chl_estimates, axis=0)
                     / np.sqrt(all_chl_estimates.shape[0]))

        # Append these summary stats as new columns
        summary_results.append(chl_means.reshape(-1,1))
        summary_results.append(chl_stds.reshape(-1,1))
        header_columns.extend([f'{pxl}pxl_{deg}_mean', f'{pxl}pxl_{deg}_std'])

# ---- Finally, write out a combined summary file ----
summary_output = np.hstack(summary_results)
header         = ','.join(header_columns)
out_summary = f'outputs/wheat/sza{sza}/{deg}/inversion/chl_summary_sza{sza}_all_pxl_{deg}_rsoil0.txt'
np.savetxt(out_summary, summary_output,
           delimiter=',', header=header, comments='')