"""
Exponential Cost Fitting for Halo Energy Distributions

This script fits an exponential function to the energy distribution of halo particles in a cosmological simulation.
It minimizes an error-based cost function to determine optimal parameters. The end goal of this approach is to run
comparisons between halos classified by this energy fitting, and halos classified by the multi-snapshot algorithm,
  to see if we can achieve similar levels of accuracy with a computationally cheaper process.

Features:
- Reads binary simulation data (`.bin` files) containing energy and radius values.
- Fits an exponential function using `scipy.optimize.minimize`.
- Saves multiple plots showcasing the fitting results and error.

Author: John Hugon
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import minimize
import os

# Set default colormap
plt.set_cmap("inferno")

# Define relative output directory
OUTPUT_DIR = "output/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data from binary files
rorb = np.fromfile('rorb_binned.bin', dtype=np.float32)
rinf = np.fromfile('rinf_binned.bin', dtype=np.float32)
eorb = np.fromfile('eorb_binned.bin', dtype=np.float32)
einf = np.fromfile('einf_binned.bin', dtype=np.float32)

# Concatenate data
r = np.concatenate((rorb, rinf))
e = np.concatenate((eorb, einf))

# Downsample data for visualization
rc = r[::200]
ec = e[::200]

w = 0.15  # Bandwidth for energy and radius

def exp_cost(abc):
    """Vectorized cost function for evaluating exponential fit error."""
    a, b, c = abc
    model = a * np.exp(-b * (rc + w)) + c
    return np.count_nonzero((ec + w > model) & (e - w < model))

def final_exp_cost(abc):
    """Final error evaluation and visualization."""
    a, b, c = abc
    model = a * np.exp(-b * r) + c
    errors = (e > model).sum()
    
    print(f'Final Error Percentage: {100 * errors / len(r):.2f}%')
    
    m = np.linspace(0, r.max(), 10000)
    model_m = a * np.exp(-b * m) + c
    model_m_upper = a * np.exp(-b * (m + w)) + c - w
    model_m_lower = a * np.exp(-b * (m - w)) + c + w
    
    # Generate and save plots
    plt.hist2d(r, e, bins=(100, 100))
    plt.plot(m, model_m, c='r')
    plt.plot(m, model_m_upper, c='r', linestyle='dashed')
    plt.plot(m, model_m_lower, c='r', linestyle='dashed')
    plt.savefig(os.path.join(OUTPUT_DIR, "Exponential_Cost_Plot.png"))
    plt.close()
    
    return errors

def my_minimize(bnds, x0, method):
    """Optimize the exponential fit using scipy.optimize.minimize."""
    result = minimize(exp_cost, x0, bounds=bnds, method=method)['x']
    a, b, c = result
    print(f'Optimized Parameters: a={a:.4f}, b={b:.4f}, c={c:.4f}')
    
    errors = final_exp_cost((a, b, c))
    percent_error = 100 * errors / len(r)
    print(f'% Error = {percent_error:.2f}%')

def main():
    """Main function to run the optimization and error evaluation."""
    bnds = [(3.2, 4.4), (2.7, 3.7), (-0.8, -0.6)]  # Bounds for a, b, c
    x0 = [3.8, 3.4, -0.7]  # Initial guess
    method = "Powell"
    my_minimize(bnds, x0, method)
    
if __name__ == "__main__":
    main()
