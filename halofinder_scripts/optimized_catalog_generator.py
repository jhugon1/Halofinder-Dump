"""
Optimized Dynamic Halo Finder for Cosmological Simulations

This script implements a computationally efficient halo finding algorithm for dark matter simulations.
It identifies and catalogs dark matter halos using orbital/infall energy cuts to distinguish bound particles
from those on escaping trajectories. The algorithm processes seeds by decreasing vmax values and applies
dynamical criteria to assign particles to halos, providing a faster alternative to multi-snapshot methods.

Authors: John Hugon, Vadim Bernshteyn
"""

import time
import os
from typing import Tuple, Optional, Dict

import h5py as h5
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from halofinder.cosmology import PARTMASS, G_GRAV
from halofinder.halo import find_r200_m200, relative_coords
from halofinder.utils import timer

# Define output directory
OUTPUT_DIR = "output/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Calibrated cut parameters (consider making these adaptive)
CUT_K = -1.694
CUT_B_VR_NEG = 1.441
CUT_B_VR_POS = 2.215

# Performance constants
MAX_SEARCH_RADIUS = 8.0  # Mpc/h - cap search radius for efficiency
MIN_PARTICLES_FOR_TREE = 1000  # Use KDTree only for large datasets


class OptimizedHaloFinder:
    """Optimized halo finder with vectorized operations and smart caching."""
    
    def __init__(self, seed_data: Tuple, part_data: Tuple):
        """Initialize with simulation data and build spatial indices."""
        self.seed_ids, self.seed_vmaxs, self.seed_pos, self.seed_vel = seed_data
        self.part_ids, self.part_pos, self.part_vel = part_data
        
        # Initialize assignment arrays
        self.particle_hids = np.full(len(self.part_ids), -1, dtype=np.int32)
        self.seed_hids = np.full(len(self.seed_ids), -1, dtype=np.int32)
        
        # Build spatial trees for efficient neighbor searches
        self._build_spatial_indices()
        
        # Pre-sort seeds by vmax (descending)
        self.vmax_order = np.argsort(self.seed_vmaxs)[::-1]
        
    def _build_spatial_indices(self):
        """Build KDTree indices for efficient spatial searches."""
        if len(self.part_pos) > MIN_PARTICLES_FOR_TREE:
            self.part_tree = cKDTree(self.part_pos)
            self.seed_tree = cKDTree(self.seed_pos)
            self.use_trees = True
        else:
            self.use_trees = False
            
    def _get_neighbors_vectorized(self, center_pos: np.ndarray, radius: float) -> Tuple[np.ndarray, np.ndarray]:
        """Get particle and seed neighbors efficiently."""
        if self.use_trees:
            # Use KDTree for large datasets
            part_indices = self.part_tree.query_ball_point(center_pos, radius)
            seed_indices = self.seed_tree.query_ball_point(center_pos, radius)
            return np.array(part_indices), np.array(seed_indices)
        else:
            # Use vectorized distance calculation for smaller datasets
            part_dists = np.linalg.norm(self.part_pos - center_pos, axis=1)
            seed_dists = np.linalg.norm(self.seed_pos - center_pos, axis=1)
            return np.where(part_dists < radius)[0], np.where(seed_dists < radius)[0]

    def apply_orbital_cut_vectorized(
        self, 
        center_pos: np.ndarray, 
        center_vel: np.ndarray, 
        r200: float, 
        m200: float,
        part_indices: np.ndarray, 
        seed_indices: np.ndarray, 
        halo_id: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized orbital classification with reduced memory allocation."""
        
        # Extract relevant data (avoid unnecessary copies)
        part_pos = self.part_pos[part_indices]
        part_vel = self.part_vel[part_indices]
        part_hids = self.particle_hids[part_indices]
        
        seed_pos = self.seed_pos[seed_indices]  
        seed_vel = self.seed_vel[seed_indices]
        seed_hids = self.seed_hids[seed_indices]
        
        # Vectorized coordinate transformations
        part_rel_pos = part_pos - center_pos
        part_rel_vel = part_vel - center_vel
        seed_rel_pos = seed_pos - center_pos
        seed_rel_vel = seed_vel - center_vel
        
        # Compute energy and radius ratios (vectorized)
        part_r_sq = np.sum(part_rel_pos**2, axis=1)
        part_v_sq = np.sum(part_rel_vel**2, axis=1)
        seed_r_sq = np.sum(seed_rel_pos**2, axis=1)
        seed_v_sq = np.sum(seed_rel_vel**2, axis=1)
        
        # Normalized quantities
        gm_r200 = G_GRAV * m200 / r200
        part_ekin_ratios = part_v_sq / gm_r200
        part_r_ratios = np.sqrt(part_r_sq) / r200
        
        seed_ekin_ratios = seed_v_sq / gm_r200
        seed_ekin_ratios = np.maximum(seed_ekin_ratios, 1e-6)  # Avoid log(0)
        seed_r_ratios = np.sqrt(seed_r_sq) / r200
        
        # Logarithmic energies
        part_ln_ekins = np.log(part_ekin_ratios)
        seed_ln_ekins = np.log(seed_ekin_ratios)
        
        # Radial motion classification (vectorized dot products)
        part_radial_vel = np.sum(part_rel_vel * part_rel_pos, axis=1)
        seed_radial_vel = np.sum(seed_rel_vel * seed_rel_pos, axis=1)
        
        # Combined orbital classification masks
        part_orbital_mask = self._compute_orbital_mask(
            part_hids, part_r_ratios, part_ln_ekins, part_radial_vel
        )
        seed_orbital_mask = self._compute_orbital_mask(
            seed_hids, seed_r_ratios, seed_ln_ekins, seed_radial_vel
        )
        
        # Handle central seed separately
        central_mask = np.sqrt(seed_r_sq) < 1e-6
        seed_orbital_mask |= central_mask
        
        return part_indices[part_orbital_mask], seed_indices[seed_orbital_mask]
    
    def _compute_orbital_mask(
        self, hids: np.ndarray, r_ratios: np.ndarray, 
        ln_ekins: np.ndarray, radial_vel: np.ndarray
    ) -> np.ndarray:
        """Compute orbital classification mask efficiently."""
        # Only consider unassigned objects within 2*R200
        base_mask = (hids == -1) & (r_ratios < 2.0)
        
        # Vectorized cut calculations
        infall_mask = radial_vel < 0
        outflow_mask = ~infall_mask
        
        # Energy cuts (vectorized)
        infall_cut = (CUT_K * r_ratios + CUT_B_VR_NEG) > ln_ekins
        outflow_cut = (CUT_K * r_ratios + CUT_B_VR_POS) > ln_ekins
        
        # Combined orbital condition
        orbital_condition = (infall_mask & infall_cut) | (outflow_mask & outflow_cut)
        
        return base_mask & orbital_condition

    @timer
    def generate_catalog(
        self,
        output_path: str,
        min_particles: int = 20,
        max_halos: Optional[int] = None
    ) -> pd.DataFrame:
        """Main halo finding pipeline with optimized processing."""
        
        print(f"Processing {len(self.seed_ids)} seeds with optimized algorithm...")
        
        # Initialize results storage
        halo_data = []
        halo_members = {}
        
        # Processing statistics
        start_time = time.time()
        processed_count = 0
        skipped_assigned = 0
        rejected_small = 0
        
        # Adaptive search radius
        search_radius = 4.0
        
        # Main processing loop
        n_process = max_halos if max_halos else len(self.seed_ids)
        
        for i in range(min(n_process, len(self.seed_ids))):
            seed_idx = self.vmax_order[i]
            
            # Skip already assigned seeds
            if self.seed_hids[seed_idx] != -1:
                skipped_assigned += 1
                continue
                
            seed_id = self.seed_ids[seed_idx]
            seed_pos = self.seed_pos[seed_idx]
            seed_vel = self.seed_vel[seed_idx]
            
            # Adaptive search radius (capped for efficiency)
            current_radius = min(search_radius, MAX_SEARCH_RADIUS)
            
            # Get neighbors efficiently
            part_indices, seed_indices = self._get_neighbors_vectorized(seed_pos, current_radius)
            
            # Filter for unassigned particles only
            free_part_mask = self.particle_hids[part_indices] == -1
            free_part_indices = part_indices[free_part_mask]
            
            if len(free_part_indices) < min_particles:
                continue
                
            # Calculate virial properties
            r200, m200 = find_r200_m200(seed_pos, self.part_pos[free_part_indices])
            
            if m200 < 1.0:  # Skip insufficient mass
                continue
                
            # Update search radius for next iteration
            search_radius = r200
            
            # Apply orbital classification
            bound_part_indices, bound_seed_indices = self.apply_orbital_cut_vectorized(
                seed_pos, seed_vel, r200, m200, part_indices, seed_indices, seed_id
            )
            
            # Check minimum particle threshold
            if len(bound_part_indices) < min_particles:
                rejected_small += 1
                continue
                
            # Update assignments (in-place for efficiency)
            self.particle_hids[bound_part_indices] = seed_id
            self.seed_hids[bound_seed_indices] = seed_id
            
            # Store halo data
            orbital_mass = PARTMASS * len(bound_part_indices)
            halo_data.append([
                seed_id, *seed_pos, *seed_vel,
                r200, m200, orbital_mass, len(bound_part_indices)
            ])
            
            # Store member indices (relative to local search)
            local_member_mask = np.isin(part_indices, bound_part_indices)
            halo_members[str(seed_id)] = np.where(local_member_mask)[0]
            
            processed_count += 1
            
            # Progress reporting
            if processed_count % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Processed {processed_count} halos in {elapsed:.1f}s")
        
        # Create final catalog
        columns = ["ID", "x", "y", "z", "vx", "vy", "vz", "R200", "M200", "Morb", "DM_particles"]
        catalog = pd.DataFrame(halo_data, columns=columns)
        
        # Print summary
        total_time = time.time() - start_time
        print(f"\nHalo finding complete!")
        print(f"Found {len(catalog)} halos in {total_time:.2f} seconds")
        print(f"Skipped {skipped_assigned} pre-assigned, rejected {rejected_small} small halos")
        print(f"Processing rate: {processed_count/total_time:.1f} halos/second")
        
        # Save results efficiently
        self._save_results_batch(output_path, catalog, halo_members)
        
        return catalog
    
    def _save_results_batch(self, output_path: str, catalog: pd.DataFrame, members: Dict):
        """Efficient batch saving to HDF5."""
        os.makedirs(output_path, exist_ok=True)
        
        # Save catalog
        catalog_path = os.path.join(output_path, "halo_catalogue.h5")
        with h5.File(catalog_path, "w") as f:
            for col in catalog.columns:
                f.create_dataset(col, data=catalog[col].values, compression="gzip")
        
        # Save assignments
        assignment_path = os.path.join(output_path, "halo_assignments.h5")
        with h5.File(assignment_path, "w") as f:
            f.create_dataset("particle_hids", data=self.particle_hids, compression="gzip")
            f.create_dataset("seed_hids", data=self.seed_hids, compression="gzip")
        
        # Save members
        members_path = os.path.join(output_path, "halo_members.h5")
        with h5.File(members_path, "w") as f:
            for halo_id, member_indices in members.items():
                f.create_dataset(halo_id, data=member_indices, compression="gzip")
        
        print(f"Results saved to {output_path}")


def load_subbox_data_optimized(filename: str, downsample: int = 1) -> Tuple[Tuple, Tuple]:
    """Optimized data loading with memory-efficient downsampling."""
    # This would replace the original load_subbox_data function
    # Implementation depends on your data format
    pass


def main():
    """Main execution function with optimized workflow."""
    
    # Load data (implement based on your data format)
    seed_data, part_data = load_subbox_data_optimized("subbox_data.h5", downsample=100)
    
    # Initialize optimized halo finder
    finder = OptimizedHaloFinder(seed_data, part_data)
    
    # Generate catalog
    catalog = finder.generate_catalog(
        output_path=OUTPUT_DIR,
        min_particles=20,
        max_halos=1000  # Process top 1000 most massive seeds
    )
    
    print(f"Generated catalog with {len(catalog)} halos")
    

if __name__ == "__main__":
    main()
