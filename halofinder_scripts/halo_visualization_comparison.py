"""
Halo Finder Comparison: Fast vs. Multi-Snapshot Methods

This script generates PDF visualizations comparing two halo-finding algorithms:
- Fast HaloFinder: A single-snapshot method optimized for computational efficiency.
- Rafa's HaloFinder: A multi-snapshot approach providing more accurate halo evolution tracking.

Purpose:
- Compare particle distributions and dynamics across both halo-finding methods.
- Visualize difference in halo mass, velocity dispersion, and radial profiles.
- Identify potential outliers where the methods significantly disagree.

Features:
- Reads HDF5 simulation data containing halo and particle information.
- Implements multiple masking functions to filter relevant halos and particles.
- Generates 2D histograms and overlay plots for detailed visual analysis.
- Produces side-by-side comparisons in PDF format for easy review.

Author: John Hugon
"""

import numpy as np
import h5py as h5
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as backend_pdf
from halofinder.config import PART_PATH, ORBITS_PATH, TRUTH_PATH, MEMBSIZE, HALOS_PATH
from halofinder.subbox import load_subbox_data
from halofinder.cosmology import G_GRAV
from time import time

# Drawing
def draw_particles(ax, x, y, xlabel, ylabel, title,
                   lims=None, vmin=1, vmax=200, bins=[100, 100],
                   map="inferno", background=None, xticks=None, yticks=None):
    
    """ Plots particles on the axes """
    mask = np.full(x.size, True)
    if lims is not None:
        x = np.hstack((x, lims[0]))
        y = np.hstack((y, lims[1]))
        mask = (x >= lims[0][0]) & (x <= lims[0][1]) &\
                (y >= lims[1][0]) & (y <= lims[1][1])

    if background is not None:
        ax.set_facecolor(background)

    _, _, _, im = ax.hist2d(x[mask], y[mask],
            bins=bins, cmap=map,
            norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax))
    
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return im


def draw_halo_circles(
        ax,
        center_xs: np.ndarray, center_ys: np.ndarray,
        radii: np.ndarray,
        color: str, label: str
):
    """ Plot halo circles """
    angles = np.linspace(0, 2*np.pi, 200)
    for i in range(len(radii)):
        x = center_xs[i] + radii[i]*np.cos(angles)
        y = center_ys[i] + radii[i]*np.sin(angles)
        set_label = None
        if i == 0: set_label = label
        ax.plot(x, y, c=color, label=set_label, linewidth=2, alpha=0.6)


# Masking data
def mask_fast_halo_particles(
        halo_id: int,
        part_fast_tags
):
    """ Generate particle mask for a given Fast Halo"""
    return part_fast_tags == halo_id
    

def mask_rockstar_halo_particles(
        halo_id: int,
        all_part_ids: np.ndarray
):
    """ Generate particle mask for a given Rockstar Halo"""

    with h5.File(ORBITS_PATH, 'r', driver='family', memb_size=MEMBSIZE) as orbits:
            with h5.File(TRUTH_PATH, 'r') as truth_table:
                mask = orbits["HID"][...] == halo_id
                halo_part_ids = orbits["PID"][mask]
                halo_orb_tags = truth_table["CLASS"][mask]
                halo_orb_part_ids = halo_part_ids[halo_orb_tags]
    
    if np.count_nonzero(mask) == 0:
        print("ERROR: no such HID in Rockstar")
    
    mask = np.isin(all_part_ids, halo_orb_part_ids)
    return mask


def mask_all_halo_particles(
        halo_x: float, halo_y: float, halo_z: float,
        halo_r200: float, 
        all_part_pos: np.ndarray,
        limit=2.0
):
    """ Generate mask of all particles near a given Halo"""
    rel_pos = all_part_pos - [halo_x, halo_y, halo_z]
    mask = np.all(np.abs(rel_pos) < limit*halo_r200, axis=1)
    return mask


def mask_nearby_halos(
        center_x: float, center_y: float, center_z: float,
        halo_xs: np.ndarray, halo_ys: np.ndarray, halo_zs: np.ndarray,
        size: float
):
    """ Mask halos close to a center """
    mask = (np.abs(halo_xs - center_x) <= size) & \
    (np.abs(halo_ys - center_y) <= size) & \
    (np.abs(halo_zs - center_z) <= size)

    return mask


def mask_bin_halos(
        mass_lims: list,
        diff_lims: list,
        fast_mass: np.ndarray, acc_mass: np.ndarray
):
    """ Mask halos in a given M - dM bin"""
    mask = np.full(fast_mass.size, False, dtype=bool)

    match_mask = acc_mass > 0.0
    matched_fast_mass = fast_mass[match_mask]
    matched_acc_mass = acc_mass[match_mask]
    m_deltas = np.log(matched_fast_mass / matched_acc_mass)

    mask[match_mask] = (matched_acc_mass > 10**mass_lims[0]) & \
            (matched_acc_mass < 10**mass_lims[1]) & \
            (m_deltas > diff_lims[0]) & \
            (m_deltas < diff_lims[1])
    
    return mask


# Loading data

def get_mini_box_id(
    x: np.ndarray,
    boxsize: float,
    minisize: float,
) -> int:
    """Returns the mini box ID to which the coordinates `x` fall into

    Parameters
    ----------
    x : np.ndarray
        Position in cartesian coordinates
    boxsize : float
        Size of simulation box
    minisize : float
        Size of mini box

    Returns
    -------
    int
        ID of the mini box
    """
    # Number of mini boxes per side
    boxes_per_side = np.int_(np.ceil(boxsize / minisize))
    # Shift in each dimension for numbering mini boxes
    shift = np.array(
        [1, boxes_per_side, boxes_per_side * boxes_per_side], dtype=np.uint32)
    # In the rare case an object is located exactly at the edge of the box,
    # move it 'inwards' by a tiny amount so that the box id is correct.
    x[np.where(x==boxsize)] -= 1e-8
    x[np.where(x==0)] += 1e-8
    if x.ndim > 1:
        return np.int_(np.sum(shift * np.floor(x / minisize), axis=1))
    else:
        return np.int_(np.sum(shift * np.floor(x / minisize)))

def load_fast_part_hids(
        part_hids_path: str
):
    """ Load fast HIDs of all particles"""
    with h5.File(part_hids_path, "r") as data:
        part_hids = data["HID"][...]
    return part_hids


def load_fast_halo_data(
        fast_halos_path: str
):
    """ Load data of all Fast Halos """
    with h5.File(fast_halos_path, "r") as data:
        halo_id = data["OHID"][...]
        halo_m200 = data["M200"][...]
        halo_morb = data["Morb"][...]
        halo_r200 = data["R200"][...]
        halo_x = data["x"][...]
        halo_y = data["y"][...]
        halo_z = data["z"][...]
        halo_vx = data["vx"][...]
        halo_vy = data["vy"][...]
        halo_vz = data["vz"][...]
    
    return (halo_id,
            halo_x, halo_y, halo_z,
            halo_vx, halo_vy, halo_vz,
            halo_morb, halo_m200,
            halo_r200)


def load_acc_halo_data():
    """ Load data about Accretion Halos """
    with h5.File(HALOS_PATH, "r") as data:
        halo_ids = data["OHID"][...]
        halo_xs = data["x"][...]
        halo_ys = data["y"][...]
        halo_zs = data["z"][...]
        halo_vxs = data["vx"][...]
        halo_vys = data["vy"][...]
        halo_vzs = data["vz"][...]
        halo_r200s = data["R200m"][...]
        halo_m200s = data["M200m"][...]
        halo_morbs = data["Morb"][...]
    return (halo_ids,
            halo_r200s, halo_m200s, halo_morbs,
            halo_xs, halo_ys, halo_zs,
            halo_vxs, halo_vys, halo_vzs)

# Calculations
def calculate_e0_v0(
        halo_m200: np.ndarray,
        halo_r200: np.ndarray,
):
    """ Calculate E0 and V0 for a given halo (for rescaling) """
    e0 = G_GRAV * halo_m200 / halo_r200
    v0 = np.sqrt(e0)
    return (e0, v0)


def calculate_radii(
        part_pos: np.ndarray,
        halo_x: float, halo_y: float, halo_z: float
):
    """ Calculate Rs of particles from a given halo center"""
    halo_pos = np.array([halo_x, halo_y, halo_z])
    return np.sqrt(np.sum((part_pos-halo_pos)**2, axis=1))


def calculate_total_vels(
        part_vels: np.ndarray,
        halo_vx: float, halo_vy: float, halo_vz: float
):
    """ Calculate Rs of particles from a given halo center"""
    halo_vel = np.array([halo_vx, halo_vy, halo_vz])
    return np.sqrt(np.sum((part_vels-halo_vel)**2, axis=1))


def calculate_radial_vels(
        part_pos: np.ndarray,
        part_vels: np.ndarray,
        halo_x: float, halo_y: float, halo_z: float,
        halo_vx: float, halo_vy: float, halo_vz: float        
):
    """ Find radial velocities with respect to a given halo """
    halo_pos = np.array([halo_x, halo_y, halo_z])
    halo_vel = np.array([halo_vx, halo_vy, halo_vz])
    part_radii = calculate_radii(part_pos, halo_x, halo_y, halo_z)
    dot_products = np.sum(np.multiply(part_vels-halo_vel, part_pos-halo_pos), axis=1)
    part_vrs = dot_products / part_radii
    return part_vrs
    
def calculate_halo_radius(
        halo_morb: np.ndarray
):
    "Calculate a typical halo size"
    return 0.8186 * ((halo_morb / 5e13) ** 0.244)

def plot_near_fast_halos(
    save_path: str,
    center_hid: int,
    all_part_ids: np.ndarray, all_part_pos: np.ndarray, all_part_vel: np.ndarray,
    fast_halo_data_object: tuple, fast_halo_part_hids: np.ndarray,
    acc_halo_data_object: tuple
):
    """ Plot halos near a given halo """
    # Extract data from the fast halo catalog
    halo_ids, \
    halo_xs, halo_ys, halo_zs, \
    halo_vxs, halo_vys, halo_vzs, \
    halo_morbs, halo_m200s, halo_r200s = fast_halo_data_object

    # Extract data from the acc halo catalog (used only for morb values in this case)
    _, _, _, acc_halo_morbs, _, _, _, _, _, _ = acc_halo_data_object

    # Identify the center halo in the fast catalog
    center_halo_mask = halo_ids == center_hid
    center_x = halo_xs[center_halo_mask][0]
    center_y = halo_ys[center_halo_mask][0]
    center_z = halo_zs[center_halo_mask][0]
    center_vx = halo_vxs[center_halo_mask][0]
    center_vy = halo_vys[center_halo_mask][0]
    center_vz = halo_vzs[center_halo_mask][0]
    center_r200 = halo_r200s[center_halo_mask][0]
    center_m200 = halo_m200s[center_halo_mask][0]
    _, center_v0 = calculate_e0_v0(center_m200, center_r200)

    # Mask halos near the center halo
    near_halo_mask = mask_nearby_halos(
        center_x, center_y, center_z,
        halo_xs, halo_ys, halo_zs,
        2 * center_r200
    )

    # Extract halos to plot
    hids_to_plot = halo_ids[near_halo_mask]
    radii_to_plot = calculate_halo_radius(halo_m200s[near_halo_mask]) / center_r200

    # Initialize PDF for plots
    pdf = backend_pdf.PdfPages(save_path)

    # First plot all particles
    total_mask = mask_all_halo_particles(center_x, center_y, center_z, center_r200, all_part_pos)
    fig, axs = plt.subplots(1, 2)
    
    # Draw particles
    draw_particles(
        axs[0],
        (all_part_pos[total_mask][:, 1] - center_y) / center_r200,
        (all_part_pos[total_mask][:, 2] - center_z) / center_r200,
        "y / R200", "z / R200", "All particles",
        lims=[[-2, 2], [-2, 2]],
        background="black",
        vmax=65,
        bins=[100, 100]
    )
    axs[0].set_aspect(1)

    # Draw central halo
    draw_halo_circles(
        axs[0],
        (halo_ys[near_halo_mask][:1] - center_y) / center_r200,
        (halo_zs[near_halo_mask][:1] - center_z) / center_r200,
        radii_to_plot[:1], "white", "Central halo"
    )

    # Draw nearby fast halos
    draw_halo_circles(
        axs[0],
        (halo_ys[near_halo_mask][1:] - center_y) / center_r200,
        (halo_zs[near_halo_mask][1:] - center_z) / center_r200,
        radii_to_plot[1:], "blue", "Fast halos"
    )

    # Calculate radii and radial velocities for the particles
    part_radii = calculate_radii(all_part_pos[total_mask], center_x, center_y, center_z)
    part_rad_vels = calculate_radial_vels(
        all_part_pos[total_mask], all_part_vel[total_mask],
        center_x, center_y, center_z,
        center_vx, center_vy, center_vz
    )
    
    draw_particles(
        axs[1],
        part_radii / center_r200, part_rad_vels / center_v0,
        "R / R200", "Vr / V0", "All Vr-R Profile",
        lims=[[0, 3], [-4, 4]],
        map="viridis",
        background="black",
        vmin=1, vmax=8,
        bins=[100, 100]
    )
    axs[1].plot([1.0, 1.0], [-4, 4], color="white", alpha=0.6, label="R200")

    plt.suptitle(f"HID={int(center_hid)})")
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close()
    print(f"Total plotted")

    # Plot halos individually
    for hid in hids_to_plot:
        halo_mask = halo_ids == hid
        halo_x = halo_xs[halo_mask][0]
        halo_y = halo_ys[halo_mask][0]
        halo_z = halo_zs[halo_mask][0]
        halo_vx = halo_vxs[halo_mask][0]
        halo_vy = halo_vys[halo_mask][0]
        halo_vz = halo_vzs[halo_mask][0]
        halo_r200 = halo_r200s[halo_mask][0]
        halo_m200 = halo_m200s[halo_mask][0]
        _, halo_v0 = calculate_e0_v0(halo_m200, halo_r200)

        # Mask particles for this specific fast halo
        mask = mask_fast_halo_particles(hid, fast_halo_part_hids)
        part_pos_masked = all_part_pos[mask]
        part_vel_masked = all_part_vel[mask]

        part_radii = calculate_radii(part_pos_masked, halo_x, halo_y, halo_z)
        part_rad_vels = calculate_radial_vels(
            part_pos_masked, part_vel_masked,
            halo_x, halo_y, halo_z,
            halo_vx, halo_vy, halo_vz
        )

        fig, axs = plt.subplots(1, 2)
        draw_particles(
            axs[0],
            (part_pos_masked[:, 1] - center_y) / center_r200,
            (part_pos_masked[:, 2] - center_z) / center_r200,
            "y / R200", "z / R200", "Fast particles",
            lims=[[-2, 2], [-2, 2]],
            background="black",
            vmax=65,
            bins=[100, 100]
        )
        axs[0].set_aspect(1)

        draw_halo_circles(
            axs[0],
            (halo_ys[near_halo_mask][:1] - center_y) / center_r200,
            (halo_zs[near_halo_mask][:1] - center_z) / center_r200,
            radii_to_plot[:1], "white", "Central halo"
        )

        draw_halo_circles(
            axs[0],
            (halo_ys[near_halo_mask][1:] - center_y) / center_r200,
            (halo_zs[near_halo_mask][1:] - center_z) / center_r200,
            radii_to_plot[1:], "blue", "Fast halos"
        )

        draw_particles(
            axs[1],
            part_radii / halo_r200, part_rad_vels / halo_v0,
            "R / R200", "Vr / V0", "Orb Vr-R Profile",
            lims=[[0, 3], [-4, 4]],
            map="viridis",
            background="black",
            vmin=1, vmax=8,
            bins=[100, 100]
        )
        axs[1].plot([1.0, 1.0], [-4, 4], color="white", alpha=0.6, label="R200")

        plt.suptitle(f"HID={int(hid)}")
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()
        print(f"{hid} plotted")

    pdf.close()
    print("PDF saved")


def plot_near_combined_halos(
    save_path: str,
    center_hid: int,
    all_part_ids: np.ndarray, all_part_pos: np.ndarray, all_part_vel: np.ndarray,
    fast_halo_data_object: tuple, fast_halo_part_hids: np.ndarray,
    acc_halo_data_object: tuple
):
    """ Plot halos near a given halo with All, Fast, and Acc particles in one row """
    
    # Extract relevant data from halo data objects
    halo_ids, \
    halo_xs, halo_ys, halo_zs, \
    halo_vxs, halo_vys, halo_vzs, \
    halo_morbs, halo_m200s, halo_r200s = fast_halo_data_object

    acc_halo_ids, \
    acc_halo_r200s, acc_halo_m200s, acc_halo_morbs, \
    acc_halo_xs, acc_halo_ys, acc_halo_zs, \
    acc_halo_vxs, acc_halo_vys, acc_halo_vzs = acc_halo_data_object

    acc_center_halo_mask = acc_halo_ids == center_hid

    center_x = acc_halo_xs[acc_center_halo_mask][0]
    center_y = acc_halo_ys[acc_center_halo_mask][0]
    center_z = acc_halo_zs[acc_center_halo_mask][0]
    center_vx = acc_halo_vxs[acc_center_halo_mask][0]
    center_vy = acc_halo_vys[acc_center_halo_mask][0]
    center_vz = acc_halo_vzs[acc_center_halo_mask][0]

    acc_center_r200 = acc_halo_r200s[acc_center_halo_mask][0]
    acc_center_m200 = acc_halo_m200s[acc_center_halo_mask][0]
    _, acc_center_v0 = calculate_e0_v0(acc_center_m200, acc_center_r200)

    # NOW find necessary vars for acc halos
    _, idx_acc, idx_fast = np.intersect1d(acc_halo_ids, halo_ids, return_indices=True)
    
    # Align the data by reordering based on the matching indices from fast and acc
    fast_halo_ids_matched = halo_ids[idx_fast]
    acc_halo_ids_matched = acc_halo_ids[idx_acc]

    acc_halo_m200s_matched = acc_halo_m200s[idx_acc]
    acc_halo_r200s_matched = acc_halo_r200s[idx_acc]

    print(len(acc_halo_m200s_matched[np.nonzero(acc_halo_m200s_matched)]))

    halo_m200s_matched = halo_m200s[idx_fast]
    halo_r200s_matched = halo_r200s[idx_fast]


    fast_center_hid = fast_halo_ids_matched[np.isin(acc_halo_ids_matched, center_hid)]
    # This can only occur if this halo is in the fast catalog run on rafas halos, but not in the fast catalog run on rockstar seeds
    if fast_center_hid.size == 0:
        print(f"{center_hid} Skipped")
        return
    
    fast_center_halo_mask = fast_halo_ids_matched == fast_center_hid
    fast_center_r200 = halo_r200s_matched[fast_center_halo_mask][0]
    fast_center_m200 = halo_m200s_matched[fast_center_halo_mask][0]
    _, fast_center_v0 = calculate_e0_v0(fast_center_m200, fast_center_r200)


    near_fast_halo_mask = mask_nearby_halos(center_x, center_y, center_z,
                                  halo_xs, halo_ys, halo_zs, 2*fast_center_r200)
    near_acc_halo_mask = mask_nearby_halos(center_x, center_y, center_z,
                                  acc_halo_xs, acc_halo_ys, acc_halo_zs, 2*acc_center_r200)

    near_acc_halo_morb = acc_halo_morbs[near_acc_halo_mask]
    sorted_indices_acc = np.argsort(-near_acc_halo_morb)

    fast_radii_to_plot = calculate_halo_radius(halo_m200s[near_fast_halo_mask])/fast_center_r200
    acc_radii_to_plot = calculate_halo_radius(acc_halo_m200s[near_acc_halo_mask][sorted_indices_acc])/acc_center_r200

    pdf = backend_pdf.PdfPages(save_path)


    fig, axs = plt.subplots(2, 3, figsize=(15, 10))  # Create 2 rows, 3 columns
    
    # Plot All particles
    total_mask = mask_all_halo_particles(center_x, center_y, center_z, fast_center_r200, all_part_pos)
    draw_particles(axs[0, 0],  # Top-left subplot for All particles
                    (all_part_pos[total_mask][:, 1]-center_y)/fast_center_r200,
                    (all_part_pos[total_mask][:, 2]-center_z)/fast_center_r200,
                    "y / R200", "z / R200", "All particles",
                    lims=[[-2, 2], [-2, 2]],
                    background="black",
                    vmax=65,
                    bins=[100, 100])
    axs[0, 0].set_aspect(1)

    # Plot Fast particles
    fast_mask = mask_fast_halo_particles(center_hid, fast_halo_part_hids)
    part_pos_masked = all_part_pos[fast_mask]
    draw_particles(axs[0, 1],  # Top-middle subplot for Fast particles
                    (part_pos_masked[:, 1]-center_y)/fast_center_r200,
                    (part_pos_masked[:, 2]-center_z)/fast_center_r200,
                    "y / R200", "z / R200", "Fast particles",
                    lims=[[-2, 2], [-2, 2]],
                    background="black",
                    vmax=65,
                    bins=[100, 100])
    axs[0, 1].set_aspect(1)

    # Plot Acc particles
    acc_mask = mask_rockstar_halo_particles(center_hid, all_part_ids)
    acc_part_pos_masked = all_part_pos[acc_mask]
    draw_particles(axs[0, 2],  # Top-right subplot for Acc particles
                    (acc_part_pos_masked[:, 1]-center_y)/acc_center_r200,
                    (acc_part_pos_masked[:, 2]-center_z)/acc_center_r200,
                    "y / R200", "z / R200", "Acc particles",
                    lims=[[-2, 2], [-2, 2]],
                    background="black",
                    vmax=65,
                    bins=[100, 100])
    axs[0, 2].set_aspect(1)

    # Plot Vr-R profiles for All, Fast, and Acc particles
    part_radii = calculate_radii(all_part_pos[total_mask], center_x, center_y, center_z)
    part_rad_vels = calculate_radial_vels(all_part_pos[total_mask],
                                            all_part_vel[total_mask],
                                            center_x, center_y, center_z,
                                            center_vx, center_vy, center_vz)
    draw_particles(axs[1, 0],  # Bottom-left subplot for All Vr-R profile
                    part_radii/fast_center_r200, part_rad_vels/fast_center_v0,
                    "R / R200", "Vr / V0", "All Vr-R Profile",
                    lims=[[0, 3], [-4, 4]],
                    map="viridis",
                    background="black",
                    vmin=1, vmax=8,
                    bins=[100, 100])
    axs[1, 0].plot([1.0, 1.0], [-4, 4], color="white", alpha=0.6, label="R200")

    draw_halo_circles(axs[0][1],
                    (halo_ys[near_fast_halo_mask][:1]-center_y)/fast_center_r200,
                    (halo_zs[near_fast_halo_mask][:1]-center_z)/fast_center_r200,
                    (fast_radii_to_plot)[:1],
                    "white", "Central halo")
    
    draw_halo_circles(axs[0][1],
                      (halo_ys[near_fast_halo_mask][1:]-center_y)/fast_center_r200,
                      (halo_zs[near_fast_halo_mask][1:]-center_z)/fast_center_r200,
                      (fast_radii_to_plot)[1:],
                      "blue", "Fast halos")
    
    draw_halo_circles(axs[0][2],
                    (acc_halo_ys[near_acc_halo_mask][sorted_indices_acc][:1]-center_y)/acc_center_r200,
                    (acc_halo_zs[near_acc_halo_mask][sorted_indices_acc][:1]-center_z)/acc_center_r200,
                    (acc_radii_to_plot)[:1],
                    "white", "Central halo")
    
    draw_halo_circles(axs[0][2],
                      (acc_halo_ys[near_acc_halo_mask][sorted_indices_acc][1:]-center_y)/acc_center_r200,
                      (acc_halo_zs[near_acc_halo_mask][sorted_indices_acc][1:]-center_z)/acc_center_r200,
                      (acc_radii_to_plot)[1:],
                      "blue", "Acc halos")

    # Fast Vr-R profile
    fast_part_radii = calculate_radii(part_pos_masked, center_x, center_y, center_z)
    fast_part_rad_vels = calculate_radial_vels(part_pos_masked, all_part_vel[fast_mask],
                                                center_x, center_y, center_z,
                                                center_vx, center_vy, center_vz)
    draw_particles(axs[1, 1],  # Bottom-middle subplot for Fast Vr-R profile
                    fast_part_radii/fast_center_r200, fast_part_rad_vels/fast_center_v0,
                    "R / R200", "Vr / V0", "Fast Vr-R Profile",
                    lims=[[0, 3], [-4, 4]],
                    map="viridis",
                    background="black",
                    vmin=1, vmax=8,
                    bins=[100, 100])
    axs[1, 1].plot([1.0, 1.0], [-4, 4], color="white", alpha=0.6, label="R200")

    # Acc Vr-R profile
    acc_part_radii = calculate_radii(acc_part_pos_masked, center_x, center_y, center_z)
    acc_part_rad_vels = calculate_radial_vels(acc_part_pos_masked, all_part_vel[acc_mask],
                                                center_x, center_y, center_z,
                                                center_vx, center_vy, center_vz)
    draw_particles(axs[1, 2],  # Bottom-right subplot for Acc Vr-R profile
                    acc_part_radii/acc_center_r200, acc_part_rad_vels/fast_center_v0,
                    "R / R200", "Vr / V0", "Acc Vr-R Profile",
                    lims=[[0, 3], [-4, 4]],
                    map="viridis",
                    background="black",
                    vmin=1, vmax=8,
                    bins=[100, 100])
    axs[1, 2].plot([1.0, 1.0], [-4, 4], color="white", alpha=0.6, label="R200")

    # Add a title to the figure
    plt.suptitle(f"HID={int(center_hid)}")
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close()
    print(f"{center_hid} plotted")
    pdf.close()

def compare_halo_list(hid_list):
    for hid in hid_list:
        hid = int(hid)
        plot_near_combined_halos(f"/path_to/halofinder/halofinder/outlier_new_plots/rafa_halos_list/halo_{hid}_comp.pdf",
            hid,
            all_part_ids, all_part_pos, all_part_vel,
            rafa_fast_halo_data_object, fast_halo_part_hids, acc_halo_data_object)
        plot_near_combined_halos(f"/path_to/halofinder/halofinder/outlier_new_plots/rockstar_seeds_list/halo_{hid}_comp.pdf",
            hid,
            all_part_ids, all_part_pos, all_part_vel,
            rock_fast_halo_data_object, fast_halo_part_hids, acc_halo_data_object)
        
def plot_halo_by_halo(hid_list):
    for hid in hid_list:
        hid = int(hid)
        plot_near_fast_halos(f"/path_to/halofinder/halofinder/outlier_new_plots/rockstar_halo_by_halo/halo_{hid}.pdf",
                                hid,
                                all_part_ids, all_part_pos, all_part_vel,
                                rock_fast_halo_data_object, fast_halo_part_hids, acc_halo_data_object)


if __name__ == "__main__":
    # pass

    seed_data, part_data = load_subbox_data(subbox_filename=None)
    seed_ids, seed_vmaxs, seed_pos, seed_vel, _, _ = seed_data
    all_part_ids, all_part_pos, all_part_vel = part_data

    fast_halo_data_path_rafa_halos = "/path_to/updated_halofinder_rafa_halos/halo_catalogue.h5"
    fast_halo_data_path_rockstar_seeds = "/path_to/updated_halofinder_data_rockstar_seeds/halo_catalogue.h5"

    fast_part_hid_path = "/path_to/halofinder_data/halofinder_on_rafa_halos/hid_particles.h5"
    acc_halo_data_path = "/path_to/simulations/a_inf.hdf5"`
    
    rafa_fast_halo_data_object = load_fast_halo_data(fast_halo_data_path_rafa_halos)
    rock_fast_halo_data_object = load_fast_halo_data(fast_halo_data_path_rockstar_seeds)

    fast_halo_part_hids = load_fast_part_hids(fast_part_hid_path)
    acc_halo_data_object = load_acc_halo_data()

    example_hid_list = [4048708, 271769, 1799676, 2586181, 2890117, 3945931]
    compare_halo_list(example_hid_list)
    plot_halo_by_halo(example_hid_list)
