"""Experiment and evaluation pipelines."""
from .m3_benchmark import run_m3_benchmark
from .m4_benchmark import run_m4_benchmark
from .helformer_benchmark import run_helformer_benchmark
from .rollout_benchmark import run_rollout_benchmark
from .pretrain_benchmark import run_pretrain_benchmark

# Legacy pipelines import helformer (TensorFlow) — load lazily to avoid
# crashing when TF is absent or the models __init__ was cleaned up.
# Use direct imports when needed:
#   from hybridts.pipelines.m3_eval import evaluate_m3_hybrids
#   from hybridts.pipelines.m4_eval import evaluate_m4_hybrids
#   from hybridts.pipelines.synth_eval import evaluate_synth_hybrids

__all__ = [
    "run_m3_benchmark",
    "run_m4_benchmark",
    "run_helformer_benchmark",
    "run_rollout_benchmark",
    "run_pretrain_benchmark",
]
