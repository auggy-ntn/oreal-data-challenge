"""Train models pipeline.

Entry point for model training. Run via DVC:
    dvc repro train_models

Or directly:
    uv run python -m src.pipelines.models.train_models
"""

from src.pipelines.models.bayesian import create_model_from_params, train_bayesian_model
from src.pipelines.utils.load_data import load_gold_data
from src.pipelines.utils.model_inputs_loading import load_params
from src.utils.logger import logger


def train_models() -> None:
    """Train models based on params.yaml configuration."""
    params = load_params()

    # Load data
    dataset = params.get("dataset", "online")
    online = dataset == "online"
    X, y = load_gold_data(online=online)
    logger.info(f"Loaded {dataset} data: {len(X)} rows")

    # Create and train Bayesian model
    model_params = params.get("bayesian_model", {})
    run_name = model_params.get("run_name", "bayesian_mmm")
    sampler_config = model_params.get("sampler", {})

    # Add dataset info to run name
    full_run_name = f"{run_name}_{dataset}"

    logger.info(f"Creating Bayesian MMM model with config: {model_params}")
    model = create_model_from_params(params)

    logger.info(f"Training model: {full_run_name}")
    train_bayesian_model(
        model=model,
        X=X,
        y=y,
        run_name=full_run_name,
        target_type="online" if online else "offline",
        sampler_config=sampler_config,
    )

    logger.info("Training pipeline complete!")


if __name__ == "__main__":
    train_models()
