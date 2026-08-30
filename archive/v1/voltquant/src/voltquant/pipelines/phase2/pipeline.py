"""
VoltQuant — Phase 2 Pipeline Definition
GAN Augmentation + PCA Feature Extraction
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    ctgan_augment,
    merge_augmented,
    apply_pca,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=ctgan_augment,
            inputs="ev_cleaned",
            outputs="ev_synthetic",
            name="ctgan_augment_node",
        ),
        node(
            func=merge_augmented,
            inputs=["ev_cleaned", "ev_synthetic"],
            outputs="ev_augmented",
            name="merge_augmented_node",
        ),
        node(
            func=apply_pca,
            inputs="ev_augmented",
            outputs="ev_phase2",
            name="apply_pca_node",
        ),
    ])