"""
Orbital Classification and Halo Energy Analysis

This script analyzes halo energy distributions and classifies orbits based on particle motion
in a cosmological simulation. It computes R200 and M200 values for seeds and classifies particles
and halos based on their velocities and positions.

Features:
- Reads binary simulation data and HDF5 catalogs.
- Computes R200, M200, and velocity-based classifications.
- Iteratively assigns orbital and infall classifications to particles.
- Sorts and processes seed halos based on mass and vmax.
- Saves results in a structured format for further analysis.

Author: John Hugon
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import os
from os.path import join, abspath

# Define constants
PARTMASS = 6.754657e+10  # Mass of each particle in M_sun / h
COSMO = {
    "flat": True,
    "H0": 70,
    "Om0": 0.3,
    "Ob0": 0.0469,
    "sigma8": 0.8355,
    "ns": 1,
}
RSOFT = 0.015  # Softening length in Mpc/h
BOXSIZE = 1_000  # Simulation box size in Mpc / h
RHOCRIT = 2.77536627e+11  # Critical density in h^2 M_sun / Mpc^3
RHOM = RHOCRIT * COSMO["Om0"]  # Matter density in h^2 M_sun / Mpc^3
MEMBSIZE = int(10 * 1000**3)  # Memory allocation for large files
G = 4.3e-9  # Gravitational constant
MPCRANGE = 5  # Range for particle selection
DS = 1  # Downsampling factor

# Define paths (anonymized)
SRC = abspath('/path/to/simulations/')
SDD = abspath('/path/to/halo_model/')

# Ensure output directory exists
OUTPUT_DIR = "output/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def max_vmax_idx(infile):
    """Returns the index of the halo with the maximum Vmax value."""
    return np.argmax(infile['vmax'][...])

def xyz(infile, idx):
    """Extracts the (x, y, z) position of a given index from the input file."""
    return infile['x'][idx], infile['y'][idx], infile['z'][idx]

def vxvyvz(infile, idx):
    """Extracts the velocity components (vx, vy, vz) of a given index."""
    return infile['vx'][idx], infile['vy'][idx], infile['vz'][idx]

def compute_r200_m200(seed_xyz, particle_xyz):
    """Computes the R200 and M200 values for a given seed position based on surrounding particles."""
    part_r = np.linalg.norm(particle_xyz - seed_xyz, axis=1)  # Compute distances
    part_r.sort()  # Sort distances in ascending order
    vol = (4/3) * np.pi * part_r**3  # Compute spherical volume
    idx = np.searchsorted((1 + np.arange(len(part_r))) * PARTMASS / vol, 200 * RHOM, side='right')
    return part_r[idx], idx * PARTMASS if idx < len(part_r) else (None, None)

def assign_orb_inf(seed_idx, tags, pos_vel, ref_pos_vel, boxed_seed_idxs, files):
    """Assigns orbital and infall classifications based on velocity and position criteria."""
    seed_file, particle_file = files
    seed_tags, part_tags = tags
    part_xyz, seed_xyz, part_vxvyvz, seed_vxvyvz = pos_vel
    ref_pos, ref_vel = ref_pos_vel
    
    free_seed_mask = seed_tags == -1  # Mask for free seeds
    free_part_mask = part_tags == -1  # Mask for free particles
    
    ref_r200, ref_m200 = compute_r200_m200(ref_pos, part_xyz[free_part_mask])
    ref_v200 = G * ref_m200 / ref_r200
    
    # Compute relative distances and velocities
    part_r = 1000 * np.linalg.norm(part_xyz[free_part_mask] - ref_pos, axis=1) / ref_r200
    seed_r = 1000 * np.linalg.norm(seed_xyz[free_seed_mask] - ref_pos, axis=1) / ref_r200
    
    log_part_vsquare = np.log(np.linalg.norm(part_vxvyvz - ref_vel, axis=1)**2 / ref_v200**2)
    log_seed_vsquare = np.log(np.linalg.norm(seed_vxvyvz - ref_vel, axis=1)**2 / ref_v200**2)
    
    part_vr = np.einsum('ij,ij->i', part_xyz - ref_pos, part_vxvyvz - ref_vel)  # Radial velocity
    seed_vr = np.einsum('ij,ij->i', seed_xyz - ref_pos, seed_vxvyvz - ref_vel)
    
    split_cal = np.array([-1.6, 0.3])  # Classification threshold
    
    # Update particle and seed classifications
    part_tags[(part_tags == -1) & ((part_vr > 0) | (part_r > 1.3) | (log_part_vsquare < split_cal[0] * part_r + split_cal[1]))] = 1
    seed_tags[(seed_tags == -1) & ((seed_vr > 0) | (seed_r > 1.3) | (log_seed_vsquare < split_cal[0] * seed_r + split_cal[1]))] = 1
    
    return seed_tags, part_tags

def main():
    """Main function to execute orbital classification."""
    with h5.File(join(SRC, 'rockstar/hlist_1.00000.h5'), 'r') as seed_file, h5.File(join(SRC, 'particle_catalogue.h5'), 'r') as particle_file:
        data = {name: np.fromfile(f'{name}.bin', dtype=np.float32) for name in ['boxed_seed_idxs', 'particle_coords', 'seed_coords', 'part_vxvyvz', 'seed_vxvyvz']}
        iterate_assignments((seed_file, particle_file), data['boxed_seed_idxs'], data['particle_coords'], data['seed_coords'], data['part_vxvyvz'], data['seed_vxvyvz'])

if __name__ == "__main__":
    main()
