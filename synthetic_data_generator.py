"""
Synthetic Data Generator for MDS Detection

Based on Section 5 and 10 of "Creating Static Data for MDS Analysis"
Generates realistic synthetic HPC data for benign and MDS attack workloads.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from mds_static_model import MDSStaticModel, MDSVariant, BenignWorkload


@dataclass
class WorkloadProfile:
    """Defines HPC counter distributions for a workload type."""
    name: str
    label: str
    is_attack: bool
    attack_variant: Optional[str]
    # Poisson lambda parameters for each counter
    llc_load_misses_lambda: float
    l1d_load_misses_lambda: float
    branch_misses_lambda: float
    branch_instr_lambda: float
    instructions_lambda: float
    cache_references_lambda: float
    page_faults_lambda: float
    context_switches_lambda: float
    cpu_cycles_lambda: float
    # Noise multiplier for realistic variation
    noise_scale: float = 0.1


class SyntheticDataGenerator:
    """
    Generates synthetic HPC datasets for MDS detection research.

    Implements workload profiles from PDF Sections 5.1 and 5.2,
    using Poisson distributions for realistic counter simulation.
    """

    # Class distribution weights from the research document
    DEFAULT_WEIGHTS = {
        'benign_cpu': 0.227,
        'benign_mem': 0.136,
        'benign_io': 0.091,
        'benign_mixed': 0.091,
        'msbds': 0.136,
        'mfbds': 0.136,
        'mlpds': 0.091,
        'mdsum': 0.091,
    }

    def __init__(self, sampling_interval_ms: int = 100, random_seed: int = 42):
        self.sampling_interval_ms = sampling_interval_ms
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)
        self.profiles = self._build_profiles()

    def _build_profiles(self) -> Dict[str, WorkloadProfile]:
        """Build workload profiles based on PDF Section 5 and Section 10."""
        return {
            # --- Benign workloads (Section 5.1) ---
            'benign_cpu': WorkloadProfile(
                name='CPU-intensive',
                label='benign_cpu',
                is_attack=False,
                attack_variant=None,
                llc_load_misses_lambda=5000,
                l1d_load_misses_lambda=15000,
                branch_misses_lambda=1000,
                branch_instr_lambda=500000,
                instructions_lambda=5000000,
                cache_references_lambda=100000,
                page_faults_lambda=20,
                context_switches_lambda=50,
                cpu_cycles_lambda=8000000,
            ),
            'benign_mem': WorkloadProfile(
                name='Memory-intensive',
                label='benign_mem',
                is_attack=False,
                attack_variant=None,
                llc_load_misses_lambda=12000,
                l1d_load_misses_lambda=40000,
                branch_misses_lambda=800,
                branch_instr_lambda=300000,
                instructions_lambda=3000000,
                cache_references_lambda=200000,
                page_faults_lambda=80,
                context_switches_lambda=30,
                cpu_cycles_lambda=6000000,
            ),
            'benign_io': WorkloadProfile(
                name='IO-intensive',
                label='benign_io',
                is_attack=False,
                attack_variant=None,
                llc_load_misses_lambda=3000,
                l1d_load_misses_lambda=8000,
                branch_misses_lambda=600,
                branch_instr_lambda=200000,
                instructions_lambda=2000000,
                cache_references_lambda=50000,
                page_faults_lambda=150,
                context_switches_lambda=200,
                cpu_cycles_lambda=4000000,
            ),
            'benign_mixed': WorkloadProfile(
                name='Mixed workload',
                label='benign_mixed',
                is_attack=False,
                attack_variant=None,
                llc_load_misses_lambda=7000,
                l1d_load_misses_lambda=20000,
                branch_misses_lambda=900,
                branch_instr_lambda=400000,
                instructions_lambda=4000000,
                cache_references_lambda=120000,
                page_faults_lambda=60,
                context_switches_lambda=80,
                cpu_cycles_lambda=7000000,
            ),
            # --- MDS attack workloads (Section 5.2) ---
            'msbds': WorkloadProfile(
                name='Store Buffer Data Sampling',
                label='msbds',
                is_attack=True,
                attack_variant='msbds',
                llc_load_misses_lambda=50000,
                l1d_load_misses_lambda=80000,
                branch_misses_lambda=3000,
                branch_instr_lambda=600000,
                instructions_lambda=5000000,
                cache_references_lambda=100000,
                page_faults_lambda=500,
                context_switches_lambda=40,
                cpu_cycles_lambda=9000000,
                noise_scale=0.15,
            ),
            'mfbds': WorkloadProfile(
                name='Fill Buffer Data Sampling',
                label='mfbds',
                is_attack=True,
                attack_variant='mfbds',
                llc_load_misses_lambda=55000,
                l1d_load_misses_lambda=85000,
                branch_misses_lambda=2800,
                branch_instr_lambda=550000,
                instructions_lambda=4800000,
                cache_references_lambda=110000,
                page_faults_lambda=600,
                context_switches_lambda=35,
                cpu_cycles_lambda=9500000,
                noise_scale=0.15,
            ),
            'mlpds': WorkloadProfile(
                name='Load Port Data Sampling',
                label='mlpds',
                is_attack=True,
                attack_variant='mlpds',
                llc_load_misses_lambda=45000,
                l1d_load_misses_lambda=70000,
                branch_misses_lambda=3500,
                branch_instr_lambda=650000,
                instructions_lambda=5200000,
                cache_references_lambda=95000,
                page_faults_lambda=450,
                context_switches_lambda=45,
                cpu_cycles_lambda=8500000,
                noise_scale=0.12,
            ),
            'mdsum': WorkloadProfile(
                name='Uncacheable Memory Sampling',
                label='mdsum',
                is_attack=True,
                attack_variant='mdsum',
                llc_load_misses_lambda=48000,
                l1d_load_misses_lambda=75000,
                branch_misses_lambda=2500,
                branch_instr_lambda=500000,
                instructions_lambda=4500000,
                cache_references_lambda=105000,
                page_faults_lambda=550,
                context_switches_lambda=50,
                cpu_cycles_lambda=8800000,
                noise_scale=0.13,
            ),
        }

    def _generate_samples_for_profile(
        self,
        profile: WorkloadProfile,
        n_samples: int,
        samples_per_run: int,
        start_sample_id: int,
        start_run_id: int,
        start_timestamp: float,
    ) -> List[Dict]:
        """Generate synthetic HPC samples for one workload profile."""
        records = []
        interval_sec = self.sampling_interval_ms / 1000.0
        sample_id = start_sample_id
        run_id = start_run_id
        timestamp = start_timestamp
        samples_in_run = 0

        for _ in range(n_samples):
            if samples_in_run >= samples_per_run:
                run_id += 1
                samples_in_run = 0

            llc = int(self.rng.poisson(profile.llc_load_misses_lambda))
            l1d = int(self.rng.poisson(profile.l1d_load_misses_lambda))
            bm = int(self.rng.poisson(profile.branch_misses_lambda))
            bi = int(self.rng.poisson(profile.branch_instr_lambda))
            instr = int(self.rng.poisson(profile.instructions_lambda))
            cref = int(self.rng.poisson(profile.cache_references_lambda))
            pf = int(self.rng.poisson(profile.page_faults_lambda))
            cs = int(self.rng.poisson(profile.context_switches_lambda))
            cyc = int(self.rng.poisson(profile.cpu_cycles_lambda))

            # Add correlated noise
            noise = 1.0 + self.rng.normal(0, profile.noise_scale)
            llc = max(0, int(llc * noise))
            l1d = max(0, int(l1d * noise))

            cache_miss_ratio = llc / max(cref, 1)
            ipc = instr / max(cyc, 1)
            branch_miss_rate = bm / max(bi, 1)

            records.append({
                'timestamp': timestamp,
                'sample_id': sample_id,
                'run_id': run_id,
                'llc_load_misses': llc,
                'l1d_load_misses': l1d,
                'branch_misses': bm,
                'branch_instr': bi,
                'instructions': instr,
                'cache_references': cref,
                'page_faults': pf,
                'context_switches': cs,
                'cpu_cycles': cyc,
                'cache_miss_ratio': cache_miss_ratio,
                'ipc': ipc,
                'branch_miss_rate': branch_miss_rate,
                'label': profile.label,
                'attack_variant': profile.attack_variant if profile.is_attack else '',
            })

            sample_id += 1
            samples_in_run += 1
            timestamp += interval_sec

        return records, sample_id, run_id + 1, timestamp

    def generate_default_dataset(
        self,
        n_samples: int = 10000,
        samples_per_run: int = 100,
        weights: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        Generate a complete synthetic MDS dataset.

        Args:
            n_samples: Total number of samples to generate
            samples_per_run: Samples per experimental run
            weights: Class distribution weights (defaults to paper recommendations)

        Returns:
            DataFrame with synthetic HPC data
        """
        if weights is None:
            weights = self.DEFAULT_WEIGHTS

        all_records = []
        sample_id = 0
        run_id = 0
        timestamp = datetime.now().timestamp()

        for label, weight in weights.items():
            profile = self.profiles[label]
            n = int(n_samples * weight)
            if n == 0:
                continue

            records, sample_id, run_id, timestamp = self._generate_samples_for_profile(
                profile, n, samples_per_run, sample_id, run_id, timestamp,
            )
            all_records.extend(records)

        df = pd.DataFrame(all_records)
        # Shuffle while preserving run integrity
        df = df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        return df

    def save_dataset(self, df: pd.DataFrame, filepath: str):
        """Save dataset to CSV."""
        df.to_csv(filepath, index=False)
        print(f"Dataset saved to {filepath} ({len(df)} samples)")
