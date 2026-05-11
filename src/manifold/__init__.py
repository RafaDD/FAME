"""Spatiotemporal manifold learning pipeline for research paper embeddings."""
from .dataset import ManifoldDataset, build_manifold_dataset_from_papers, load_manifold_dataset
from .model import LinearProjection, ManifoldSpine, NaiveModel, SpatiotemporalMapper
from .trainer import ManifoldTrainer, NaiveTrainer, NaiveTrainingConfig, TrainingConfig

__all__ = [
    "ManifoldDataset",
    "build_manifold_dataset_from_papers",
    "load_manifold_dataset",
    "SpatiotemporalMapper",
    "ManifoldSpine",
    "LinearProjection",
    "NaiveModel",
    "ManifoldTrainer",
    "NaiveTrainer",
    "TrainingConfig",
    "NaiveTrainingConfig",
]
