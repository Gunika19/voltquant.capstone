"""
VoltQuant — Preprocessing Pipeline Definition
Wires all nodes into a Kedro pipeline.
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    drop_irrelevant_columns,
    rename_columns,
    handle_missing_values,
    engineer_features,
    create_labels,
    select_final_features,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=drop_irrelevant_columns,
            inputs="ev_raw",
            outputs="ev_dropped",
            name="drop_irrelevant_columns_node",
        ),
        node(
            func=rename_columns,
            inputs="ev_dropped",
            outputs="ev_renamed",
            name="rename_columns_node",
        ),
        node(
            func=handle_missing_values,
            inputs="ev_renamed",
            outputs="ev_no_nulls",
            name="handle_missing_values_node",
        ),
        node(
            func=engineer_features,
            inputs="ev_no_nulls",
            outputs="ev_engineered",
            name="engineer_features_node",
        ),
        node(
            func=create_labels,
            inputs="ev_engineered",
            outputs="ev_labeled",
            name="create_labels_node",
        ),
        node(
            func=select_final_features,
            inputs="ev_labeled",
            outputs="ev_cleaned",
            name="select_final_features_node",
        ),
    ])