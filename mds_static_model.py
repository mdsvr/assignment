"""
MDS Static Model for Microarchitectural Data Sampling Analysis

Based on: "Creating Static Data for MDS Analysis" (May 2026)
This module implements a static dataset generation and analysis framework
for MDS attack detection using Hardware Performance Counters (HPCs).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum
import json
from datetime import datetime


class MDSVariant(Enum):
    """MDS attack variants as defined in the research document."""
    MSBDS = "msbds"  # Store Buffer Data Sampling
    MFBDS = "mfbds"  # Fill Buffer Data Sampling
    MLPDS = "mlpds"  # Load Port Data Sampling
    MDSUM = "mdsum"  # Uncacheable Memory Sampling


class BenignWorkload(Enum):
    """Benign workload types for dataset generation."""
    CPU_INTENSIVE = "benign_cpu"
    MEMORY_INTENSIVE = "benign_mem"
    IO_INTENSIVE = "benign_io"
    MIXED = "benign_mixed"


@dataclass
class HPCEvent:
    """Hardware Performance Counter event definition."""
    name: str
    description: str
    relevance: str


class HPCEvents:
    """Collection of HPC events for MDS detection as specified in the PDF."""
    
    EVENTS = [
        HPCEvent("LLC-load-misses", "Last-level cache misses", "Flush+Reload pattern"),
        HPCEvent("L1-dcache-load-misses", "L1 data cache misses", "Cache side-channel"),
        HPCEvent("branch-misses", "Branch mispredictions", "Speculative execution"),
        HPCEvent("branch-instructions", "Total branches", "Baseline activity"),
        HPCEvent("instructions", "Instructions retired", "Workload intensity"),
        HPCEvent("cache-references", "Cache accesses", "Memory access pattern"),
        HPCEvent("page-faults", "Page fault events", "MDS trigger mechanism"),
        HPCEvent("context-switches", "Context switch count", "Scheduling behavior"),
    ]
    
    @classmethod
    def get_event_names(cls) -> List[str]:
        """Get list of event names for data collection."""
        return [event.name for event in cls.EVENTS]
    
    @classmethod
    def get_event_dict(cls) -> Dict[str, HPCEvent]:
        """Get dictionary of events by name."""
        return {event.name: event for event in cls.EVENTS}


@dataclass
class HPCSample:
    """Single HPC measurement sample following the PDF schema."""
    timestamp: float
    sample_id: int
    run_id: int
    llc_load_misses: int
    l1d_load_misses: int
    branch_misses: int
    branch_instr: int
    instructions: int
    cache_references: int
    page_faults: int
    context_switches: int
    cpu_cycles: int
    cache_miss_ratio: float
    ipc: float
    branch_miss_rate: float
    label: str
    attack_variant: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert sample to dictionary."""
        return {
            'timestamp': self.timestamp,
            'sample_id': self.sample_id,
            'run_id': self.run_id,
            'llc_load_misses': self.llc_load_misses,
            'l1d_load_misses': self.l1d_load_misses,
            'branch_misses': self.branch_misses,
            'branch_instr': self.branch_instr,
            'instructions': self.instructions,
            'cache_references': self.cache_references,
            'page_faults': self.page_faults,
            'context_switches': self.context_switches,
            'cpu_cycles': self.cpu_cycles,
            'cache_miss_ratio': self.cache_miss_ratio,
            'ipc': self.ipc,
            'branch_miss_rate': self.branch_miss_rate,
            'label': self.label,
            'attack_variant': self.attack_variant
        }


class MDSStaticModel:
    """
    Main static model class for MDS analysis.
    
    Implements data generation, feature engineering, and analysis
    following the methodology in the research document.
    """
    
    def __init__(self, sampling_interval_ms: int = 100):
        """
        Initialize the MDS static model.
        
        Args:
            sampling_interval_ms: Sampling interval in milliseconds (default: 100ms)
                                 as recommended in the PDF for optimal tradeoff
        """
        self.sampling_interval_ms = sampling_interval_ms
        self.samples: List[HPCSample] = []
        self.run_counter = 0
        self.sample_counter = 0
        
    def _calculate_derived_features(self, 
                                    llc_misses: int,
                                    cache_refs: int,
                                    instructions: int,
                                    cycles: int,
                                    branch_misses: int,
                                    branch_instr: int) -> Tuple[float, float, float]:
        """
        Calculate derived features as specified in Section 4.1 of the PDF.
        
        Returns:
            Tuple of (cache_miss_ratio, ipc, branch_miss_rate)
        """
        cache_miss_ratio = llc_misses / max(cache_refs, 1)
        ipc = instructions / max(cycles, 1)
        branch_miss_rate = branch_misses / max(branch_instr, 1)
        
        return cache_miss_ratio, ipc, branch_miss_rate
    
    def add_sample(self, 
                   llc_load_misses: int,
                   l1d_load_misses: int,
                   branch_misses: int,
                   branch_instr: int,
                   instructions: int,
                   cache_references: int,
                   page_faults: int,
                   context_switches: int,
                   cpu_cycles: int,
                   label: str,
                   attack_variant: Optional[str] = None) -> HPCSample:
        """
        Add a single HPC sample to the model.
        
        Args:
            llc_load_misses: LLC load misses count
            l1d_load_misses: L1 data cache load misses
            branch_misses: Branch misprediction count
            branch_instr: Total branch instructions
            instructions: Instructions retired
            cache_references: Cache reference count
            page_faults: Page fault count
            context_switches: Context switch count
            cpu_cycles: CPU cycles elapsed
            label: Class label (e.g., 'benign_cpu', 'msbds')
            attack_variant: MDS variant if this is an attack sample
            
        Returns:
            The created HPCSample object
        """
        cache_miss_ratio, ipc, branch_miss_rate = self._calculate_derived_features(
            llc_load_misses, cache_references, instructions, 
            cpu_cycles, branch_misses, branch_instr
        )
        
        sample = HPCSample(
            timestamp=datetime.now().timestamp(),
            sample_id=self.sample_counter,
            run_id=self.run_counter,
            llc_load_misses=llc_load_misses,
            l1d_load_misses=l1d_load_misses,
            branch_misses=branch_misses,
            branch_instr=branch_instr,
            instructions=instructions,
            cache_references=cache_references,
            page_faults=page_faults,
            context_switches=context_switches,
            cpu_cycles=cpu_cycles,
            cache_miss_ratio=cache_miss_ratio,
            ipc=ipc,
            branch_miss_rate=branch_miss_rate,
            label=label,
            attack_variant=attack_variant
        )
        
        self.samples.append(sample)
        self.sample_counter += 1
        
        return sample
    
    def new_run(self):
        """Start a new experimental run (increments run_id)."""
        self.run_counter += 1
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert all samples to a pandas DataFrame.
        
        Returns:
            DataFrame with all samples following the PDF schema
        """
        data = [sample.to_dict() for sample in self.samples]
        return pd.DataFrame(data)
    
    def save_to_csv(self, filepath: str):
        """
        Save dataset to CSV file following the PDF schema.
        
        Args:
            filepath: Path to save the CSV file
        """
        df = self.to_dataframe()
        df.to_csv(filepath, index=False)
    
    def get_class_distribution(self) -> pd.Series:
        """Get distribution of samples by class label."""
        df = self.to_dataframe()
        return df['label'].value_counts()
    
    def get_statistics(self) -> Dict:
        """
        Get basic statistics about the dataset.
        
        Returns:
            Dictionary containing dataset statistics
        """
        df = self.to_dataframe()
        
        return {
            'total_samples': len(df),
            'total_runs': df['run_id'].nunique(),
            'sampling_interval_ms': self.sampling_interval_ms,
            'class_distribution': df['label'].value_counts().to_dict(),
            'attack_samples': len(df[df['label'].isin([v.value for v in MDSVariant])]),
            'benign_samples': len(df[df['label'].isin([w.value for w in BenignWorkload])])
        }
