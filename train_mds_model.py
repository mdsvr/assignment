#!/usr/bin/env python3
"""
End-to-End MDS Detection ML Pipeline

Generates synthetic data, engineers features, validates the dataset,
trains multiple ML models, and reports evaluation metrics.

Usage:
    python train_mds_model.py
    python train_mds_model.py --n_samples 20000 --model random_forest
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from synthetic_data_generator import SyntheticDataGenerator
from feature_engineering import MDSFeatureEngineer
from dataset_validator import DatasetValidator
from mds_detector import MDSDetector, MultiClassMDSDetector


def run_pipeline(
    n_samples: int = 10000,
    samples_per_run: int = 100,
    model_type: str = 'random_forest',
    output_dir: str = 'output',
    random_seed: int = 42,
    run_all_models: bool = False,
):
    """
    Run the complete MDS detection pipeline.

    Steps:
    1. Generate synthetic dataset
    2. Engineer features
    3. Validate dataset
    4. Train model(s)
    5. Evaluate and report
    """
    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()

    # ---- Step 1: Generate synthetic dataset ----
    print("\n" + "=" * 70)
    print("STEP 1: Generating Synthetic Dataset")
    print("=" * 70)
    generator = SyntheticDataGenerator(
        sampling_interval_ms=100,
        random_seed=random_seed,
    )
    df = generator.generate_default_dataset(
        n_samples=n_samples,
        samples_per_run=samples_per_run,
    )
    dataset_path = os.path.join(output_dir, 'mds_dataset.csv')
    generator.save_dataset(df, dataset_path)
    print(f"Generated {len(df)} samples across {df['run_id'].nunique()} runs")
    print(f"Class distribution:\n{df['label'].value_counts().to_string()}")

    # ---- Step 2: Feature engineering ----
    print("\n" + "=" * 70)
    print("STEP 2: Feature Engineering")
    print("=" * 70)
    engineer = MDSFeatureEngineer(window_size=10)
    df_features = engineer.extract_all_features(df)
    print(f"Features before engineering: {len(df.columns)}")
    print(f"Features after engineering:  {len(df_features.columns)}")

    # Save engineered dataset
    features_path = os.path.join(output_dir, 'mds_dataset_features.csv')
    df_features.to_csv(features_path, index=False)
    print(f"Feature-engineered dataset saved to {features_path}")

    # Feature importance by correlation
    importance = engineer.get_feature_importance_scores(df_features)
    print("\nTop 15 features by correlation with attack label:")
    for feat, score in list(importance.items())[:15]:
        print(f"  {feat:35s}: {score:.4f}")

    # ---- Step 3: Dataset validation ----
    print("\n" + "=" * 70)
    print("STEP 3: Dataset Validation")
    print("=" * 70)
    validator = DatasetValidator(df_features)
    validator.statistical_validation()
    validator.separability_analysis()

    report = validator.generate_validation_report()
    report_path = os.path.join(output_dir, 'validation_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nValidation report saved to {report_path}")

    # ---- Step 4: Train model(s) ----
    print("\n" + "=" * 70)
    print("STEP 4: Training ML Models")
    print("=" * 70)

    models_to_train = (
        ['random_forest', 'gradient_boosting', 'logistic_regression', 'svm', 'neural_network']
        if run_all_models
        else [model_type]
    )

    all_results = {}

    for mt in models_to_train:
        print(f"\n{'─' * 50}")
        print(f"Training: {mt}")
        print(f"{'─' * 50}")

        # Binary classifier
        detector = MDSDetector(model_type=mt)
        X_train, X_val, X_test, y_train, y_val, y_test = detector.split_by_run(df_features)
        detector.train(X_train, y_train, X_val, y_val)

        # Evaluate
        metrics = detector.evaluate(X_test, y_test)
        detector.print_evaluation_report(metrics)

        # Feature importance (for tree-based models)
        feat_imp = detector.get_feature_importance()
        if feat_imp:
            print(f"\nTop 10 Feature Importance ({mt}):")
            sorted_imp = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:10]
            for feat, imp in sorted_imp:
                print(f"  {feat:35s}: {imp:.4f}")

        all_results[mt] = {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'roc_auc': metrics.get('roc_auc'),
        }

    # ---- Step 5: Multi-class detector ----
    print("\n" + "=" * 70)
    print("STEP 5: Multi-Class MDS Variant Detection")
    print("=" * 70)
    mc_detector = MultiClassMDSDetector(model_type='random_forest')
    X_train, X_val, X_test, y_train, y_val, y_test = mc_detector.split_by_run(df_features)
    mc_detector.train(X_train, y_train, X_val, y_val)

    mc_pred = mc_detector.predict(X_test)
    from sklearn.metrics import classification_report, accuracy_score
    mc_acc = accuracy_score(y_test, mc_pred)
    print(f"\nMulti-class accuracy: {mc_acc:.4f}")
    print("\nMulti-class Classification Report:")
    print(classification_report(y_test, mc_pred))

    all_results['multi_class_random_forest'] = {'accuracy': mc_acc}

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    elapsed = time.time() - start_time
    print(f"Total time: {elapsed:.1f}s")
    print(f"Dataset: {len(df)} samples, {len(df_features.columns)} features")
    print(f"\nModel Comparison (Binary Classification):")
    print(f"{'Model':25s} {'Accuracy':>10s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'AUC':>10s}")
    print("-" * 70)
    for mt, res in all_results.items():
        if mt == 'multi_class_random_forest':
            continue
        auc_str = f"{res['roc_auc']:.4f}" if res.get('roc_auc') else 'N/A'
        print(f"{mt:25s} {res['accuracy']:10.4f} {res['precision']:10.4f} "
              f"{res['recall']:10.4f} {res['f1_score']:10.4f} {auc_str:>10s}")

    # Save results
    results_path = os.path.join(output_dir, 'results.json')
    serializable = {}
    for k, v in all_results.items():
        serializable[k] = {kk: float(vv) if vv is not None else None for kk, vv in v.items()}
    with open(results_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='End-to-end MDS detection ML pipeline'
    )
    parser.add_argument('--n_samples', type=int, default=10000)
    parser.add_argument('--samples_per_run', type=int, default=100)
    parser.add_argument('--model', type=str, default='random_forest',
                        choices=['random_forest', 'gradient_boosting',
                                 'logistic_regression', 'svm', 'neural_network'])
    parser.add_argument('--all_models', action='store_true',
                        help='Train and compare all model types')
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    run_pipeline(
        n_samples=args.n_samples,
        samples_per_run=args.samples_per_run,
        model_type=args.model,
        output_dir=args.output_dir,
        random_seed=args.seed,
        run_all_models=args.all_models,
    )


if __name__ == '__main__':
    main()
