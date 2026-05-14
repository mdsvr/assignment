#!/usr/bin/env python3
"""
Standalone script to generate synthetic MDS detection datasets.

Usage:
    python generate_mds_dataset.py
    python generate_mds_dataset.py --n_samples 20000 --output my_dataset.csv
"""

import argparse
from synthetic_data_generator import SyntheticDataGenerator


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic MDS detection dataset'
    )
    parser.add_argument(
        '--n_samples', type=int, default=10000,
        help='Total number of samples (default: 10000)'
    )
    parser.add_argument(
        '--output', type=str, default='mds_synthetic_dataset.csv',
        help='Output CSV file path (default: mds_synthetic_dataset.csv)'
    )
    parser.add_argument(
        '--samples_per_run', type=int, default=100,
        help='Samples per experimental run (default: 100)'
    )
    parser.add_argument(
        '--sampling_interval', type=int, default=100,
        help='Sampling interval in milliseconds (default: 100)'
    )
    parser.add_argument(
        '--random_seed', type=int, default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    args = parser.parse_args()

    print(f"Generating {args.n_samples} samples...")
    generator = SyntheticDataGenerator(
        sampling_interval_ms=args.sampling_interval,
        random_seed=args.random_seed,
    )
    df = generator.generate_default_dataset(
        n_samples=args.n_samples,
        samples_per_run=args.samples_per_run,
    )

    generator.save_dataset(df, args.output)

    print(f"\nDataset shape: {df.shape}")
    print(f"\nClass distribution:")
    print(df['label'].value_counts().to_string())
    print(f"\nSample columns: {list(df.columns)}")


if __name__ == '__main__':
    main()
