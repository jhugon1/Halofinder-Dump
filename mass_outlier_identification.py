"""
Halo Catalog Outlier Identification

This script generates visualizations to highlight discrepancies between two halo-finding algorithms:
- Fast HaloFinder (single-snapshot method)
- Rafa's HaloFinder (multi-snapshot method)

It compares halo properties, including mass distributions and particle densities, to identify key differences.
The goal is to assess classification accuracy and explore biases between methods.

Features:
- Loads halo catalogs and filters valid halos
- Computes and plots mass ratios between matched halos
- Generates 2D histograms and Gaussian-fitted histograms for mass errors
- Plots density distributions as a function of radius
- Highlights significant outliers in the halo catalog comparison

Author: John Hugon
"""

from os.path import join
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from halofinder.config import HALOS_PATH

plt.inferno()

fast_halo_data_path_rafa_halos = "/path/to/updated_halofinder_rafa_halos/halo_catalogue.h5"
fast_halo_data_path_rockstar_seeds = "/path/to/updated_halofinder_data_rockstar_seeds/halo_catalogue.h5"

# SAMPLE_PATH = fast_halo_data_path_rafa_halos
SAMPLE_PATH = fast_halo_data_path_rockstar_seeds

with h5.File(HALOS_PATH, 'r') as hdf:
    Rafa_morb = hdf['Morb'][()]
    Rafa_hid = hdf['OHID'][()]
    mask = Rafa_morb > 0
    Rafa_morb = Rafa_morb[mask]
    Rafa_hid = Rafa_hid[mask]


with h5.File(SAMPLE_PATH, 'r') as sample_cat:
    fast_morb = sample_cat['Morb'][()]
    fast_hid = sample_cat['OHID'][()]
    target_hid = 2586181
    index = list(fast_hid).index(target_hid)
    print(f"Part Count {target_hid}: {fast_morb[index]/7.754657e+10}")
    print(len(Rafa_hid[np.isin(Rafa_hid, fast_hid)]))

def compute_radius(x, y, z):
    return np.linalg.norm([x, y, z], axis=0)

def find_mass_ratio(
        fast_morb: np.ndarray,
        Rafa_morb: np.ndarray,
        fast_hid: np.ndarray,  # Add fast_hid as a parameter
        Rafa_hid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Calculates the log of the ratio of fast and Rafa morb values and returns the corresponding
    ln_ratio, Rafa_morb_match, and fast_hid values for matched halos.
    '''

    _, rafa_ind, fast_ind = np.intersect1d(Rafa_hid, fast_hid, return_indices=True)

    # Match and sort fast morb and fast HID arrays
    _, rafa_ind, fast_ind = np.intersect1d(Rafa_hid, fast_hid, return_indices=True)
    ln_ratio = np.log(fast_morb[fast_ind] / Rafa_morb[rafa_ind])

    print(f"Total Matched Halos: {len(ln_ratio)}")
    print(f"High Error Halo Count: {np.count_nonzero(np.abs(ln_ratio) > 0.5)}")

    return ln_ratio, Rafa_morb[rafa_ind], Rafa_hid[rafa_ind], fast_hid[fast_ind]

def mass_ratio_2dhist(
    ln_ratio: np.ndarray,
    Rafa_morb_match: np.ndarray,
    error_cutoff: float,
    sigma_line: np.ndarray
) -> None:
    """
    Plots a 2d histogram of the mass ratio log(Mfast/MRafa) across a massbin range

    Params:
        ln_ratio (array): of log(Mfast/MRafa)
        Rafa_morb_match (array): array of Rafa_morb for haloes contained in fast_morb
        error_cutoff (float): sets error limit for plot scaling
        simga_line (array): array of tuples corresponding to values of mean + 2 sigma and
                                    corresponding massbin values
    """
    mask = (np.log10(Rafa_morb_match) > bin_range[0]) & (np.log10(Rafa_morb_match) < bin_range[1])
    ratio_bin = ln_ratio[mask]

    if len(ratio_bin) == 0:
        return None  # Skip empty bins

    median = np.median(ratio_bin)
    sigma = 1.4826 * np.median(np.abs(ratio_bin - median))
    filtered_bin = ratio_bin[(ratio_bin > median - 2 * sigma) & (ratio_bin < median + 2 * sigma)]

    mean, std_dev = norm.fit(filtered_bin)

    # Plot
    x = np.linspace(-error_cutoff, error_cutoff, 1000)
    y = norm.pdf(x, mean, std_dev)

    plt.hist(ratio_bin, bins=35, density=True, alpha=0.6)
    plt.plot(x, y, label=f"Mean: {mean:.3f}, Std: {std_dev:.3f}", color="red")

    plt.xlabel('ln(Morb_Fast / Morb_Rafa)')
    plt.ylabel("Density")
    plt.title(f"Mass Ratio Distribution in {bin_range[0]} < log(M_Rafa) < {bin_range[1]}")
    plt.legend()

    if save_plots:
        plt.savefig(f"mass_ratio_{bin_range}.png")
    plt.close()

    return mean - 2 * std_dev, mean + 2 * std_dev, np.mean(bin_range)

def mass_ratio_hist(
    ln_ratio: np.ndarray,
    Rafa_morb_match: np.ndarray,
    bin: tuple,
    error_cutoff: float,
    save_plots: bool
) -> tuple[float, float, float]:
    '''
    Plots a normalized histogram of the mass ratio log(Mfast/MRafa) at a massbin
    and fits a Gaussian within +- 2 Sigma of the histogram

    params:
        ln_ratio (array): of log(Mfast/MRafa)
        Rafa_morb_match (array): array of Rafa_morb for haloes contained in fast_morb
        bin (tuple): Mass bin to plot the histogram for
        error_cutoff (float): sets error limit for plot scaling
        save_plots (bool): boolean = True if desire to save plots as files

    returns:
        tuple of mean + 2sigma, mean-2sigma, average massbin value
    '''
    bin_mask = np.where(((np.log10(Rafa_morb_match) > bin[0]) & (np.log10(Rafa_morb_match) < bin[1])))
    ratio_bin = ln_ratio[bin_mask]

    m = np.median(ratio_bin)
    s = 1.4826*np.median(np.abs(ratio_bin - m))
    # "Cheap" Standard deviation based on median absolute deviation

    std_mask = np.where((ratio_bin > m-2*s) & (ratio_bin < m+2*s))
    std_bin = ratio_bin[std_mask] # Bin containing values within 2 sigma
    mean,std =norm.fit(std_bin) # Fit the gaussian to the downsampled bin

    # outlier_count = (np.abs(ratio_bin-mean)/std > 3).sum()
    # outlier_fraction = outlier_count/len(ratio_bin)  Fraction of haloes outside 3 sigma
    # print(f"Outlier Fraction {outlier_fraction}")
    error_mask = np.where(np.abs(ln_ratio[bin_mask]) < error_cutoff)

    x = np.linspace(-error_cutoff, error_cutoff, 1000) 
    y = norm.pdf(x, mean, std) # Fit gaussian across x data
    bin_size = 35
    plt.hist(ratio_bin[error_mask], bins = bin_size, density = True)
              # label = f"Fraction > 3*Std: {'%.1f'%(100*outlier_fraction)}%")
    plt.plot(x, y, label = f"Mean: {'%.3f'%mean} \n Std: {'%.3f'%std}")
    # plt.scatter(0, 0, s = 1, label = f"Halo Count: {len(ratio_bin)}") # Using this scatter only to display halo count on the plot legend
    
    plt.xlabel('ln(Morb_Fast/Morb_Rafa)')
    plt.ylabel("Halo Count Normed")
    text = r'$log_{10}$'
    plt.title(f"Mass Difference Histogram --- Bin: {bin[0]} < {text}(Morb_Rafa) < {bin[1]}")
    plt.legend(loc = "upper right")
    plt.yticks([]) # Delete the y_ticks as histogram is normalized
    if save_plots:
        plot_name = "ratio_histogram.png"
        plt.savefig(f'{bin}_{plot_name}')
    plt.close()

    # Create 2 Sigma line:
    return (-2*std+mean, 2*std+mean, (bin[0]+bin[1])/2)

def plot_density_vs_radius(fast_xyz: tuple[np.ndarray, np.ndarray, np.ndarray], 
                           Rafa_xyz: tuple[np.ndarray, np.ndarray, np.ndarray], 
                           labels=("Fast", "Rafa")):
    """
    Plots the density of orbiting particles as a function of radius for two datasets.

    Params:
        fast_xyz (tuple): (x, y, z) coordinates for the fast particles
        Rafa_xyz (tuple): (x, y, z) coordinates for the Rafa particles
        labels (tuple): Labels for the two datasets for legend (default: ("Fast", "Rafa"))
    """
    fast_r, Rafa_r = compute_radius(*fast_xyz), compute_radius(*Rafa_xyz)
    bins = np.linspace(0, max(fast_r.max(), Rafa_r.max()), 100)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    fast_hist, _ = np.histogram(fast_r, bins=bins)
    Rafa_hist, _ = np.histogram(Rafa_r, bins=bins)
    volumes = (4 / 3) * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3)

    plt.plot(bin_centers, fast_hist / volumes, label=f"{labels[0]} Density", marker='o')
    plt.plot(bin_centers, Rafa_hist / volumes, label=f"{labels[1]} Density", marker='s', linestyle="--")

    plt.xlabel('Radial Distance r')
    plt.ylabel('Density of Orbiting Particles')
    plt.title('Density vs. Radius')
    plt.legend()
    plt.grid(True)
    plt.savefig("density_vs_radius.png")
    plt.close()

# Sample usage

def make_full_histogram(Rafa_morb_match, ln_ratio, error_mask, mass_mask):
    plt.hist2d(np.log10(Rafa_morb_match[error_mask & mass_mask]), 
               ln_ratio[error_mask & mass_mask], bins=(100, 100), cmap="inferno")
    
    plt.xlabel(r'$log_{10}(\mathrm{Morb\ Rafa})$')
    plt.ylabel(r'$ln(\mathrm{Morb_{Fast}} / \mathrm{Morb_{Rafa}})$')
    plt.title("Mass Ratio Distribution")

    plt.colorbar(label="Counts")
    plt.savefig(output_path)
    plt.close()
    
if __name__ == "__main__":
    # Compute log mass ratio
    ln_ratio, Rafa_morb_match, Rafa_hid_match, fast_hid_match = find_mass_ratio(
        fast_morb, Rafa_morb, fast_hid, Rafa_hid
    )

    # Define filtering masks
    error_mask = np.abs(ln_ratio) < 1000
    mass_mask = np.log10(Rafa_morb_match) > 14.5  # Adjust threshold if necessary

    # Scatter plot for mass discrepancies
    x_data = np.log10(Rafa_morb_match[error_mask & mass_mask])
    y_data = ln_ratio[error_mask & mass_mask]

    plt.scatter(x_data, y_data, s=10)

    # Highlight extreme outliers
    outlier_mask = (y_data > 0.32) | (y_data < -0.38)
    filtered_hids = Rafa_hid_match[error_mask & mass_mask][outlier_mask]

    # Assign unique markers for highlighted halos
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff9966', '#cc99ff', '#66ffcc']
    letters = 'ABCDEFGHIJKLMNOPQRS'
    halo_to_letter_map = {int(hid): letters[i % len(letters)] for i, hid in enumerate(filtered_hids)}

    for i, hid in enumerate(filtered_hids):
        plt.scatter(x_data[outlier_mask][i], y_data[outlier_mask][i], 
                    s=500, color=colors[i % len(colors)], marker='*', edgecolor='white')
        plt.text(x_data[outlier_mask][i], y_data[outlier_mask][i], 
                 halo_to_letter_map[int(hid)], fontsize=8, ha='center', va='center', color='black')

    # Generate legend for labeled outliers
    legend_elements = [plt.Line2D([0], [0], marker='*', color='w', 
                                  label=f'{halo_to_letter_map[int(hid)]}: HID {int(hid)}',
                                  markerfacecolor=colors[i % len(colors)], markersize=10)
                       for i, hid in enumerate(filtered_hids)]
    
    plt.legend(handles=legend_elements, loc='upper right', title="Halo IDs", fancybox=True)
    plt.xlabel(r'$log_{10}(\mathrm{Morb\ Rafa})$')
    plt.ylabel(r'$ln(\mathrm{Morb_{Fast}} / \mathrm{Morb_{Rafa}})$')
    plt.title("Mass Difference Percent Error - ln(Mfast/MRafa)")
    plt.tight_layout(pad=2)

    # Save scatter plot
    scatter_output = "/path/to/halofinder/halofinder/morb_plots/rockstar_scatter_stars.png"
    plt.savefig(scatter_output)
    plt.close()

    # Generate full histogram
    histogram_output = "/path/to/halofinder/halofinder/morb_plots/new_rock_seeds_hist.png"
    make_full_histogram(Rafa_morb_match, ln_ratio, error_mask, mass_mask, histogram_output)
    plt.close()
