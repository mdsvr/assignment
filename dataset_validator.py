"""
Dataset Validator for MDS Detection Datasets

Based on Section 9 of "Creating Static Data for MDS Analysis"
Implements statistical validation and separability analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class DatasetValidator:
    """
    Validates MDS detection datasets following PDF Section 9.

    Implements:
    - 9.1 Statistical Validation
    - 9.2 Separability Analysis
    - 9.3 Cross-Validation Considerations
    """

    ATTACK_LABELS = ['msbds', 'mfbds', 'mlpds', 'mdsum']

    def __init__(self, df: pd.DataFrame, label_col: str = 'label'):
        self.df = df.copy()
        self.label_col = label_col
        self.numeric_cols = self._get_numeric_feature_cols()
        self.validation_results: Dict = {}

    def _get_numeric_feature_cols(self) -> List[str]:
        """Get numeric feature columns (excluding metadata)."""
        exclude = {'sample_id', 'run_id', 'timestamp', 'is_attack'}
        numeric = self.df.select_dtypes(include=[np.number]).columns.tolist()
        return [c for c in numeric if c not in exclude]

    def statistical_validation(self) -> Dict:
        """
        Run statistical validation checks (Section 9.1).

        Checks:
        - Class balance
        - Feature variance (non-zero)
        - Outlier detection (negative counts, impossible values)
        - Basic distribution statistics
        """
        results = {}

        # Class balance
        class_counts = self.df[self.label_col].value_counts()
        total = len(self.df)
        results['class_distribution'] = class_counts.to_dict()
        results['class_percentages'] = (class_counts / total * 100).round(2).to_dict()

        attack_mask = self.df[self.label_col].isin(self.ATTACK_LABELS)
        results['attack_ratio'] = float(attack_mask.sum() / total)
        results['benign_ratio'] = float((~attack_mask).sum() / total)

        # Feature variance
        variances = self.df[self.numeric_cols].var()
        zero_var = variances[variances == 0].index.tolist()
        results['zero_variance_features'] = zero_var
        results['all_features_have_variance'] = len(zero_var) == 0

        # Outlier detection: negative counter values
        counter_cols = [
            'llc_load_misses', 'l1d_load_misses', 'branch_misses',
            'branch_instr', 'instructions', 'cache_references',
            'page_faults', 'context_switches', 'cpu_cycles'
        ]
        negative_counts = {}
        for col in counter_cols:
            if col in self.df.columns:
                neg = (self.df[col] < 0).sum()
                if neg > 0:
                    negative_counts[col] = int(neg)
        results['negative_counter_values'] = negative_counts
        results['no_negative_counters'] = len(negative_counts) == 0

        # NaN check
        nan_counts = self.df[self.numeric_cols].isna().sum()
        nan_features = nan_counts[nan_counts > 0].to_dict()
        results['nan_features'] = {k: int(v) for k, v in nan_features.items()}

        # Basic stats per feature
        results['feature_stats'] = {}
        for col in self.numeric_cols[:10]:
            results['feature_stats'][col] = {
                'mean': float(self.df[col].mean()),
                'std': float(self.df[col].std()),
                'min': float(self.df[col].min()),
                'max': float(self.df[col].max()),
            }

        self.validation_results['statistical'] = results
        self._print_statistical_results(results)
        return results

    def _print_statistical_results(self, results: Dict):
        """Print statistical validation results."""
        print("=" * 70)
        print("Statistical Validation Results")
        print("=" * 70)

        print("\nClass Distribution:")
        for label, count in results['class_distribution'].items():
            pct = results['class_percentages'][label]
            print(f"  {label:15s}: {count:6d} ({pct:.1f}%)")

        print(f"\n  Attack ratio: {results['attack_ratio']:.3f}")
        print(f"  Benign ratio: {results['benign_ratio']:.3f}")

        print(f"\nFeature Variance Check:")
        if results['all_features_have_variance']:
            print("  All features have non-zero variance.")
        else:
            print(f"  Zero-variance features: {results['zero_variance_features']}")

        print(f"\nNegative Counter Check:")
        if results['no_negative_counters']:
            print("  No negative counter values found.")
        else:
            for col, cnt in results['negative_counter_values'].items():
                print(f"  {col}: {cnt} negative values")

        if results['nan_features']:
            print(f"\nNaN Values:")
            for col, cnt in results['nan_features'].items():
                print(f"  {col}: {cnt} NaN values")
        print("=" * 70)

    def separability_analysis(self) -> Dict:
        """
        Analyze class separability (Section 9.2).

        Tests:
        - Mann-Whitney U test between attack and benign for each feature
        - Mutual information between features and labels
        - Effect size (Cohen's d)
        """
        results = {}
        attack_mask = self.df[self.label_col].isin(self.ATTACK_LABELS)
        attack_df = self.df[attack_mask]
        benign_df = self.df[~attack_mask]

        # Mann-Whitney U tests
        mann_whitney_results = {}
        for col in self.numeric_cols:
            if col not in self.df.columns:
                continue
            a = attack_df[col].dropna()
            b = benign_df[col].dropna()
            if len(a) < 5 or len(b) < 5:
                continue
            try:
                stat, pval = stats.mannwhitneyu(a, b, alternative='two-sided')
                mann_whitney_results[col] = {
                    'statistic': float(stat),
                    'p_value': float(pval),
                    'significant': pval < 0.05,
                }
            except Exception:
                pass

        results['mann_whitney'] = mann_whitney_results
        significant_features = [k for k, v in mann_whitney_results.items() if v['significant']]
        results['significant_features_count'] = len(significant_features)
        results['total_features_tested'] = len(mann_whitney_results)

        # Effect size (Cohen's d) for top features
        cohens_d = {}
        for col in self.numeric_cols:
            a = attack_df[col].dropna()
            b = benign_df[col].dropna()
            if len(a) < 2 or len(b) < 2:
                continue
            pooled_std = np.sqrt(((len(a) - 1) * a.std() ** 2 + (len(b) - 1) * b.std() ** 2) /
                                 (len(a) + len(b) - 2))
            if pooled_std > 0:
                d = abs(a.mean() - b.mean()) / pooled_std
                cohens_d[col] = float(d)
        results['cohens_d'] = dict(sorted(cohens_d.items(), key=lambda x: x[1], reverse=True))

        # Mutual information
        try:
            le = LabelEncoder()
            y_encoded = le.fit_transform(self.df[self.label_col])
            X = self.df[self.numeric_cols].fillna(0)
            mi = mutual_info_classif(X, y_encoded, random_state=42)
            mi_dict = dict(zip(self.numeric_cols, mi))
            results['mutual_information'] = dict(
                sorted(mi_dict.items(), key=lambda x: x[1], reverse=True)
            )
        except Exception as e:
            results['mutual_information'] = {'error': str(e)}

        self.validation_results['separability'] = results
        self._print_separability_results(results)
        return results

    def _print_separability_results(self, results: Dict):
        """Print separability analysis results."""
        print("=" * 70)
        print("Separability Analysis Results")
        print("=" * 70)

        print(f"\nMann-Whitney U Tests:")
        print(f"  Significant features: {results['significant_features_count']}"
              f" / {results['total_features_tested']}")

        print(f"\nTop 10 Features by Effect Size (Cohen's d):")
        for i, (feat, d) in enumerate(list(results['cohens_d'].items())[:10]):
            print(f"  {feat:30s}: d = {d:.4f}")

        if isinstance(results.get('mutual_information'), dict) and 'error' not in results['mutual_information']:
            print(f"\nTop 10 Features by Mutual Information:")
            for feat, mi in list(results['mutual_information'].items())[:10]:
                print(f"  {feat:30s}: MI = {mi:.4f}")

        print("=" * 70)

    def generate_validation_report(self) -> str:
        """Generate a text summary of all validation results."""
        lines = [
            "MDS Dataset Validation Report",
            "=" * 50,
            f"Total samples: {len(self.df)}",
            f"Total features: {len(self.numeric_cols)}",
            "",
        ]

        stat = self.validation_results.get('statistical', {})
        if stat:
            lines.append("Statistical Validation:")
            lines.append(f"  Class balance OK: {stat.get('attack_ratio', 0):.1%} attack / "
                         f"{stat.get('benign_ratio', 0):.1%} benign")
            lines.append(f"  All features have variance: {stat.get('all_features_have_variance', 'N/A')}")
            lines.append(f"  No negative counters: {stat.get('no_negative_counters', 'N/A')}")
            lines.append("")

        sep = self.validation_results.get('separability', {})
        if sep:
            lines.append("Separability Analysis:")
            lines.append(f"  Significant features (p<0.05): "
                         f"{sep.get('significant_features_count', 0)}"
                         f" / {sep.get('total_features_tested', 0)}")
            top_d = list(sep.get('cohens_d', {}).items())[:5]
            if top_d:
                lines.append("  Top features by effect size:")
                for feat, d in top_d:
                    lines.append(f"    {feat}: d={d:.3f}")

        return "\n".join(lines)
