
"""
ML Optimization Package

Provides advanced ML model performance optimization strategies:
- Quantization (dynamic, static, mixed-precision)
- Pruning (magnitude-based, structured)
- Batching (dynamic batching for inference)
- Parallelism (data/model/parameter parallelism)
- GPU optimizations (mixed precision, device placement)
- Caching (memoization & tensor caching)
- Monitoring (latency, throughput, GPU usage)
"""

from .quantization import Quantizer
from .pruning import Pruner
from .batching import DynamicBatcher
from .parallelism import ParallelTrainer
from .gpu_optimization import GPUOptimizer
from .caching import InferenceCache
from .monitoring import MLMetricsCollector

__all__ = [
    "Quantizer",
    "Pruner",
    "DynamicBatcher",
    "ParallelTrainer",
    "GPUOptimizer",
    "InferenceCache",
    "MLMetricsCollector",
]
