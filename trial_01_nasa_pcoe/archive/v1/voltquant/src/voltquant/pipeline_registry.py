from voltquant.pipelines.phase2.pipeline import create_pipeline as phase2_pipeline
from kedro.pipeline import Pipeline
from voltquant.pipelines.preprocessing.pipeline import create_pipeline

def register_pipelines() -> dict[str, Pipeline]:
    return {
        "__default__": create_pipeline(),
        "preprocessing": create_pipeline(),
    }