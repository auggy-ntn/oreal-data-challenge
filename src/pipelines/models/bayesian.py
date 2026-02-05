"""Bayesian MMM model."""

from pathlib import Path
from typing import Any
import warnings

import arviz as az
from dotenv import load_dotenv
import mlflow
import numpy as np
import pandas as pd
from pymc_extras.prior import Prior
from pymc_marketing.mmm import (
    MMM,
    AdstockTransformation,
    DelayedAdstock,
    GeometricAdstock,
    HillSaturation,
    LogisticSaturation,
    MichaelisMentenSaturation,
    SaturationTransformation,
    TanhSaturation,
    WeibullCDFAdstock,
    WeibullPDFAdstock,
)

from constants.column_names import gold as gold_cols
import constants.paths as pth
from src.pipelines.models.utils.contributions import compute_contribution
from src.pipelines.models.utils.plotting import (
    plot_contribution_decomposition,
    plot_yearly_incremental_units,
    plot_yearly_roi,
    plot_yearly_spend,
)
from src.utils.logger import logger

# Mapping from string names to transformation classes
ADSTOCK_MAPPING: dict[str, type[AdstockTransformation]] = {
    "geometric": GeometricAdstock,
    "delayed": DelayedAdstock,
    "weibull_pdf": WeibullPDFAdstock,
    "weibull_cdf": WeibullCDFAdstock,
}

SATURATION_MAPPING: dict[str, type[SaturationTransformation]] = {
    "logistic": LogisticSaturation,
    "tanh": TanhSaturation,
    "michaelis_menten": MichaelisMentenSaturation,
    "hill": HillSaturation,
}


def _parse_prior_value(value: Any) -> Any:
    """Parse a prior value, handling nested priors and arrays.

    Args:
        value: Value from YAML config. Can be a dict (for nested Prior),
            a list (converted to np.array), or a scalar.

    Returns:
        Parsed value: Prior object, np.array, or scalar.
    """
    if isinstance(value, dict) and "dist" in value:
        # Nested prior
        return _parse_prior_config(value)
    if isinstance(value, list):
        return np.array(value)
    return value


def _parse_prior_config(config: dict) -> Prior:
    """Parse a single prior configuration dict into a Prior object.

    Args:
        config: Dict with 'dist' (distribution name) and 'kwargs' (parameters).

    Returns:
        Prior object.
    """
    dist = config["dist"]
    kwargs = config.get("kwargs", {})
    # Parse kwargs, handling nested priors
    parsed_kwargs = {k: _parse_prior_value(v) for k, v in kwargs.items()}
    return Prior(dist, **parsed_kwargs)


def parse_model_config(config: dict | None) -> dict[str, Prior] | None:
    """Parse model_config from params.yaml into Prior objects.

    Args:
        config: Model config dict from params.yaml. Keys are parameter names,
            values are dicts with 'dist' and 'kwargs'.

    Returns:
        Dict mapping parameter names to Prior objects, or None if no config.

    Example:
        Input YAML:
            model_config:
              saturation_beta:
                dist: LogNormal
                kwargs:
                  mu: [2, 1]
                  sigma: 1
              likelihood:
                dist: Normal
                kwargs:
                  sigma:
                    dist: HalfNormal
                    kwargs:
                      sigma: 2

        Output:
            {
                "saturation_beta": Prior("LogNormal", mu=np.array([2, 1]), sigma=1),
                "likelihood": Prior("Normal", sigma=Prior("HalfNormal", sigma=2)),
            }
    """
    if not config:
        return None

    model_config = {}
    for param_name, prior_config in config.items():
        if prior_config is None:
            continue
        if not isinstance(prior_config, dict) or "dist" not in prior_config:
            logger.warning(
                f"Invalid prior config for {param_name}: missing 'dist'. Skipping."
            )
            continue
        model_config[param_name] = _parse_prior_config(prior_config)

    return model_config if model_config else None


def compute_r_squared(model: MMM, X: pd.DataFrame, y: pd.Series) -> float:
    """Compute R-squared on historical data.

    Args:
        model: Trained MMM model.
        X: Feature DataFrame.
        y: Actual target values.

    Returns:
        R-squared value.
    """
    # Get posterior predictive mean
    posterior_pred = model.predict(X)

    # Handle different return types from predict()
    if hasattr(posterior_pred, "dims"):
        # xarray DataArray - average over chain and draw dimensions
        y_pred = posterior_pred.mean(dim=["chain", "draw"]).values
    elif hasattr(posterior_pred, "values"):
        # pandas Series/DataFrame
        y_pred = posterior_pred.values
    else:
        # numpy array
        y_pred = np.array(posterior_pred)

    # Flatten if needed
    y_pred = np.ravel(y_pred)
    y_actual = np.ravel(y.values)

    # Compute R-squared: 1 - SS_res / SS_tot
    ss_res = np.sum((y_actual - y_pred) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    return float(r_squared)


def get_adstock_transformation(config: dict) -> AdstockTransformation:
    """Create adstock transformation from config.

    Args:
        config: Adstock configuration with 'type' and 'l_max' keys.

    Returns:
        AdstockTransformation instance.

    Raises:
        ValueError: If adstock type is not recognized.
    """
    adstock_type = config.get("type", "geometric")
    l_max = config.get("l_max", 8)

    if adstock_type not in ADSTOCK_MAPPING:
        raise ValueError(
            f"Unknown adstock type: {adstock_type}. "
            f"Available: {list(ADSTOCK_MAPPING.keys())}"
        )

    return ADSTOCK_MAPPING[adstock_type](l_max=l_max)


def get_saturation_transformation(config: dict) -> SaturationTransformation:
    """Create saturation transformation from config.

    Args:
        config: Saturation configuration with 'type' key.

    Returns:
        SaturationTransformation instance.

    Raises:
        ValueError: If saturation type is not recognized.
    """
    saturation_type = config.get("type", "logistic")

    if saturation_type not in SATURATION_MAPPING:
        raise ValueError(
            f"Unknown saturation type: {saturation_type}. "
            f"Available: {list(SATURATION_MAPPING.keys())}"
        )

    return SATURATION_MAPPING[saturation_type]()


def create_bayesian_model(
    adstock: AdstockTransformation,
    saturation: SaturationTransformation,
    yearly_seasonality: int,
    model_config: dict | None = None,
) -> MMM:
    """Create a Bayesian MMM model.

    Args:
        adstock: Adstock transformation.
        saturation: Saturation transformation.
        yearly_seasonality: Yearly seasonality (for Fourier harmonics).
        model_config: Model configuration. Prior distributions and other settings.

    Returns:
        MMM: Bayesian MMM model.

    Note:
        The model automatically scales the target and channel columns.
    """
    model = MMM(
        date_column=gold_cols.DATE,
        channel_columns=gold_cols.FEATURE_COLUMNS,
        control_columns=gold_cols.CONTROL_COLUMNS,
        adstock=adstock,
        saturation=saturation,
        yearly_seasonality=yearly_seasonality,
        model_config=model_config,
    )
    return model


def create_model_from_params(params: dict) -> MMM:
    """Create a Bayesian MMM model from params.yaml configuration.

    Args:
        params: Parameters dictionary from params.yaml.

    Returns:
        MMM: Configured Bayesian MMM model.
    """
    model_params = params.get("bayesian_model", {})

    adstock = get_adstock_transformation(model_params.get("adstock", {}))
    saturation = get_saturation_transformation(model_params.get("saturation", {}))
    yearly_seasonality = model_params.get("yearly_seasonality", 2)
    model_config = parse_model_config(model_params.get("model_config"))

    return create_bayesian_model(
        adstock=adstock,
        saturation=saturation,
        yearly_seasonality=yearly_seasonality,
        model_config=model_config,
    )


def train_bayesian_model(
    model: MMM,
    X: pd.DataFrame,
    y: pd.Series,
    run_name: str,
    sampler_config: dict | None = None,
) -> MMM:
    """Train the Bayesian MMM model and log to MLflow.

    Args:
        model: The MMM model to train.
        X: Feature DataFrame.
        y: Target Series.
        run_name: Name for the MLflow run and saved model.
        sampler_config: MCMC sampler configuration (draws, tune, chains, target_accept).

    Returns:
        The trained model.
    """
    load_dotenv()

    # Default sampler config
    sampler_config = sampler_config or {}
    fit_kwargs = {
        "draws": sampler_config.get("draws", 1000),
        "tune": sampler_config.get("tune", 1000),
        "chains": sampler_config.get("chains", 4),
        "target_accept": sampler_config.get("target_accept", 0.9),
    }

    # Ensure models directory exists
    model_dir = pth.MODELS_DIR / run_name
    model_dir.mkdir(exist_ok=True, parents=True)
    model_path = model_dir / "model.nc"

    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_params(
            {
                "adstock_type": type(model.adstock).__name__,
                "adstock_l_max": model.adstock.l_max,
                "saturation_type": type(model.saturation).__name__,
                "yearly_seasonality": model.yearly_seasonality,
                **{f"sampler_{k}": v for k, v in fit_kwargs.items()},
            }
        )

        logger.info("Fitting Bayesian MMM model...")
        idata = model.fit(X, y, **fit_kwargs)

        # Compute and log diagnostics (only on posterior samples to avoid NaN warnings)
        logger.info("Computing model diagnostics...")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            summary = az.summary(idata.posterior)

        # Compute diagnostic statistics
        ess_values = summary["ess_bulk"].dropna()
        rhat_values = summary["r_hat"].dropna()

        mean_ess = ess_values.mean()
        min_ess = ess_values.min()
        mean_rhat = rhat_values.mean()
        max_rhat = rhat_values.max()
        n_params_rhat_above_1_01 = (rhat_values > 1.01).sum()

        # Compute R-squared on historical data
        logger.info("Computing R-squared on historical data...")
        r_squared = compute_r_squared(model, X, y)

        mlflow.log_metrics(
            {
                "mean_ess_bulk": mean_ess,
                "min_ess_bulk": min_ess,
                "mean_rhat": mean_rhat,
                "max_rhat": max_rhat,
                "n_params_rhat_above_1_01": n_params_rhat_above_1_01,
                "r_squared": r_squared,
            }
        )

        logger.info(
            f"Mean ESS: {mean_ess:.0f}, Min ESS: {min_ess:.0f}\n"
            f"Mean R-hat: {mean_rhat:.4f}, Max R-hat: {max_rhat:.4f}\n"
            f"Params with R-hat > 1.01: {n_params_rhat_above_1_01}\n"
            f"R-squared: {r_squared:.4f}"
        )

        # Save model locally
        logger.info(f"Saving model to {model_path}")
        model.save(str(model_path))

        # Log model artifact to MLflow
        mlflow.log_artifact(str(model_path), artifact_path="models")

        logger.info(
            f"Model training complete. MLflow run: {mlflow.active_run().info.run_id}"
        )

        # Compute channel contributions and graphs, log them as artifacts
        artifacts_dir = model_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True, parents=True)

        contributions_df = compute_contribution(model)
        contributions_df.to_csv(artifacts_dir / "contributions.csv", index=False)

        # Generate plots
        logger.info("Generating performance plots...")
        for granularity in ["L2", "L3"]:
            # Contribution decomposition plot
            fig_decomp = plot_contribution_decomposition(
                contributions_df, granularity=granularity
            )
            fig_decomp.write_image(
                artifacts_dir / f"contribution_decomposition_{granularity}.png", scale=2
            )

            # Incremental units plot
            fig_units = plot_yearly_incremental_units(
                contributions_df, granularity=granularity
            )
            fig_units.write_image(
                artifacts_dir / f"incremental_units_{granularity}.png", scale=2
            )

            # Spend plot
            fig_spend = plot_yearly_spend(contributions_df, granularity=granularity)
            fig_spend.write_image(artifacts_dir / f"spend_{granularity}.png", scale=2)

            # ROI plot
            fig_roi = plot_yearly_roi(contributions_df, granularity=granularity)
            fig_roi.write_image(artifacts_dir / f"roi_{granularity}.png", scale=2)

        # Log all artifacts in the directory
        mlflow.log_artifacts(str(artifacts_dir), artifact_path="artifacts")

    return model


def load_model(model_path: Path | str) -> MMM:
    """Load a trained Bayesian MMM model.

    Args:
        model_path: Path to the saved model (.nc file).

    Returns:
        Loaded MMM model.
    """
    return MMM.load(str(model_path))
