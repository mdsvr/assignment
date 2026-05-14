"""
Feature Engineering for MDS Detection

Based on Section 4 of "Creating Static Data for MDS Analysis"
Implements raw, derived, MDS-specific, and temporal features.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')


class MDSFeatureEngineer:
    """
    Feature engineering for MDS detection following PDF Section 4.
    
    Implements:
    - 4.1 Raw Counter Features
    - 4.2 Derived Features (statistical)
    - 4.3 MDS-Specific Features
    - 4.4 Temporal Features
    """
    
    def __init__(self, window_size: int = 10):
        """
        Initialize the feature engineer.
        
        Args:
            window_size: Size of sliding window for derived features (default: 10 samples)
        """
        self.window_size = window_size
    
    def extract_raw_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract raw counter features (Section 4.1).
        
        Includes absolute values, rates, and basic ratios.
        
        Args:
            df: DataFrame with raw HPC data
            
        Returns:
            DataFrame with added raw features
        """
        df = df.copy()
        
        # Counter rates (events per second)
        # Assuming sampling_interval_ms is available or derived from timestamps
        if 'sampling_interval_ms' in df.columns:
            interval_sec = df['sampling_interval_ms'] / 1000.0
        else:
            # Estimate from timestamp differences
            interval_sec = df['timestamp'].diff().mean()
            if pd.isna(interval_sec):
                interval_sec = 0.1  # Default to 100ms
        
        counter_cols = ['llc_load_misses', 'l1d_load_misses', 'branch_misses',
                       'branch_instr', 'instructions', 'cache_references',
                       'page_faults', 'context_switches']
        
        for col in counter_cols:
            if col in df.columns:
                df[f'{col}_rate'] = df[col] / interval_sec
        
        # Basic ratios (already in main model, but ensure they exist)
        if 'cache_miss_ratio' not in df.columns:
            df['cache_miss_ratio'] = df['llc_load_misses'] / df['cache_references'].clip(lower=1)
        
        return df
    
    def extract_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract derived statistical features (Section 4.2).
        
        Computes mean, std, variance, min/max, rate of change,
        coefficient of variation, skewness, and kurtosis over sliding windows.
        
        Args:
            df: DataFrame with time-series HPC data
            
        Returns:
            DataFrame with added statistical features
        """
        df = df.copy()
        
        counter_cols = ['llc_load_misses', 'l1d_load_misses', 'branch_misses',
                       'instructions', 'cache_references', 'page_faults']
        
        for col in counter_cols:
            if col not in df.columns:
                continue
                
            # Rolling statistics
            df[f'{col}_rolling_mean'] = df[col].rolling(window=self.window_size, min_periods=1).mean()
            df[f'{col}_rolling_std'] = df[col].rolling(window=self.window_size, min_periods=1).std()
            df[f'{col}_rolling_var'] = df[col].rolling(window=self.window_size, min_periods=1).var()
            df[f'{col}_rolling_min'] = df[col].rolling(window=self.window_size, min_periods=1).min()
            df[f'{col}_rolling_max'] = df[col].rolling(window=self.window_size, min_periods=1).max()
            
            # Rate of change (first derivative)
            df[f'{col}_rate_of_change'] = df[col].diff()
            
            # Coefficient of variation (CV = std/mean)
            mean_val = df[f'{col}_rolling_mean']
            std_val = df[f'{col}_rolling_std']
            df[f'{col}_cv'] = std_val / mean_val.clip(lower=0.001)
            
            # Skewness and kurtosis (smaller window for stability)
            skew_window = min(self.window_size, 5)
            df[f'{col}_skew'] = df[col].rolling(window=skew_window, min_periods=3).apply(
                lambda x: stats.skew(x) if len(x) >= 3 else 0
            )
            df[f'{col}_kurtosis'] = df[col].rolling(window=skew_window, min_periods=4).apply(
                lambda x: stats.kurtosis(x) if len(x) >= 4 else 0
            )
        
        return df
    
    def extract_mds_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract MDS-specific features (Section 4.3).
        
        Features tailored to MDS detection:
        - Cache miss to instruction ratio
        - Page fault frequency
        - Branch misprediction rate
        - L1-miss to LLC-miss ratio
        - IPC anomalies
        - Store-to-load forwarding events (simulated)
        
        Args:
            df: DataFrame with HPC data
            
        Returns:
            DataFrame with added MDS-specific features
        """
        df = df.copy()
        
        # Cache miss to instruction ratio (high during Flush+Reload)
        df['cache_miss_to_instr_ratio'] = df['llc_load_misses'] / df['instructions'].clip(lower=1)
        
        # Page fault frequency (MDS triggers use page faults)
        if 'page_faults' in df.columns:
            # Assuming sampling interval is ~100ms
            df['page_fault_freq'] = df['page_faults'] * 10  # Convert to approximate per-second
        
        # Branch misprediction rate (speculative execution indicator)
        df['branch_misprediction_rate'] = df['branch_misses'] / df['branch_instr'].clip(lower=1)
        
        # L1-miss to LLC-miss ratio (indicates cache manipulation)
        df['l1_to_llc_miss_ratio'] = df['l1d_load_misses'] / df['llc_load_misses'].clip(lower=1)
        
        # IPC anomalies (Instructions Per Cycle)
        # IPC is already calculated, but let's add anomaly detection
        if 'ipc' in df.columns:
            ipc_mean = df['ipc'].rolling(window=self.window_size, min_periods=1).mean()
            ipc_std = df['ipc'].rolling(window=self.window_size, min_periods=1).std()
            df['ipc_zscore'] = (df['ipc'] - ipc_mean) / ipc_std.clip(lower=0.001)
            df['ipc_anomaly'] = (df['ipc_zscore'].abs() > 2).astype(int)
        
        # Store-to-load forwarding events (specific to MSBDS)
        # Since we can't measure this directly with standard HPCs,
        # we'll simulate it based on cache miss patterns
        df['store_to_load_fwd_sim'] = (
            (df['l1d_load_misses'] > df['llc_load_misses']) & 
            (df['branch_misses'] > df['branch_instr'] * 0.1)
        ).astype(int)
        
        return df
    
    def extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal features (Section 4.4).
        
        Time-series derived features:
        - Autocorrelation (attack periodicity)
        - Burst detection (sudden spikes)
        - Trend analysis
        - Cross-correlation between counters
        
        Args:
            df: DataFrame with time-series HPC data
            
        Returns:
            DataFrame with added temporal features
        """
        df = df.copy()
        
        counter_cols = ['llc_load_misses', 'l1d_load_misses', 'branch_misses',
                       'page_faults']
        
        # Autocorrelation (attack periodicity)
        for col in counter_cols:
            if col not in df.columns:
                continue
                
            # Calculate autocorrelation with lag 1-5
            for lag in [1, 2, 3, 5]:
                df[f'{col}_autocorr_lag{lag}'] = df[col].rolling(
                    window=20, min_periods=10
                ).apply(
                    lambda x: x.autocorr(lag=lag) if len(x) > lag else 0
                )
        
        # Burst detection (sudden spikes)
        for col in counter_cols:
            if col not in df.columns:
                continue
                
            rolling_mean = df[col].rolling(window=self.window_size, min_periods=1).mean()
            rolling_std = df[col].rolling(window=self.window_size, min_periods=1).std()
            
            # Detect bursts: values > mean + 2*std
            df[f'{col}_burst'] = (df[col] > rolling_mean + 2 * rolling_std).astype(int)
            df[f'{col}_burst_intensity'] = df[col] / rolling_mean.clip(lower=1)
        
        # Trend analysis (sustained elevation)
        for col in counter_cols:
            if col not in df.columns:
                continue
                
            # Linear trend over window
            df[f'{col}_trend'] = df[col].rolling(window=self.window_size, min_periods=2).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0
            )
            
            # Sustained elevation indicator
            rolling_mean = df[col].rolling(window=self.window_size, min_periods=1).mean()
            global_mean = df[col].mean()
            df[f'{col}_sustained_high'] = (rolling_mean > global_mean * 1.5).astype(int)
        
        # Cross-correlation between key counters
        if 'llc_load_misses' in df.columns and 'page_faults' in df.columns:
            df['llc_pf_crosscorr'] = df['llc_load_misses'].rolling(
                window=20, min_periods=10
            ).apply(
                lambda x: x.corr(df.loc[x.index, 'page_faults']) if len(x) >= 10 else 0
            )
        
        if 'l1d_load_misses' in df.columns and 'branch_misses' in df.columns:
            df['l1d_branch_crosscorr'] = df['l1d_load_misses'].rolling(
                window=20, min_periods=10
            ).apply(
                lambda x: x.corr(df.loc[x.index, 'branch_misses']) if len(x) >= 10 else 0
            )
        
        return df
    
    def extract_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all feature types in sequence.
        
        Args:
            df: DataFrame with raw HPC data
            
        Returns:
            DataFrame with all engineered features
        """
        df = self.extract_raw_features(df)
        df = self.extract_statistical_features(df)
        df = self.extract_mds_specific_features(df)
        df = self.extract_temporal_features(df)
        
        return df
    
    def get_feature_importance_scores(self, df: pd.DataFrame, label_col: str = 'label') -> Dict[str, float]:
        """
        Calculate feature importance based on correlation with labels.
        
        Args:
            df: DataFrame with engineered features
            label_col: Name of the label column
            
        Returns:
            Dictionary of feature names to importance scores
        """
        # Convert labels to binary (attack vs benign)
        attack_labels = ['msbds', 'mfbds', 'mlpds', 'mdsum']
        df_with_binary = df.copy()
        df_with_binary['is_attack'] = df_with_binary[label_col].isin(attack_labels).astype(int)
        
        # Calculate correlation with attack label for each feature
        numeric_cols = df_with_binary.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'is_attack']
        
        importance = {}
        for col in numeric_cols:
            corr = df_with_binary[col].corr(df_with_binary['is_attack'])
            if not pd.isna(corr):
                importance[col] = abs(corr)
        
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
