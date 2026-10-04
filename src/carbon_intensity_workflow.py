"""Reproducible OWID electricity carbon-intensity regression workflow.

The module is intentionally shared by the command-line runner and the notebook so
that every number used by the paper comes from one executed implementation.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import shap
import sklearn
import xgboost
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


SEED = 42
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs" / "carbon_intensity"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
MODEL_DIR = OUTPUT_DIR / "models"
PAPER_DIR = ROOT / "paper"

ENERGY_PATH = RAW_DIR / "owid-grapher-electricity-inputs.csv"
TARGET_PATH = RAW_DIR / "owid-grapher-carbon-intensity.csv"
TARGET_METADATA_PATH = RAW_DIR / "owid-grapher-carbon-intensity.metadata.json"

GRAPHER_BASE_URL = "https://ourworldindata.org/grapher"
ELECTRICITY_MIX_URL = f"{GRAPHER_BASE_URL}/electricity-mix"
GRAPHER_CSV_OPTIONS = "v=1&csvType=full&useColumnShortNames=false"

SOURCE_URLS = {
    "electricity_by_source": (
        f"{ELECTRICITY_MIX_URL}.csv?{GRAPHER_CSV_OPTIONS}&source=total&"
        "metric=by_source&frequency=annual"
    ),
    "electricity_generation": (
        f"{ELECTRICITY_MIX_URL}.csv?{GRAPHER_CSV_OPTIONS}&source=total&"
        "metric=generation&frequency=annual"
    ),
    "electricity_demand": (
        f"{ELECTRICITY_MIX_URL}.csv?{GRAPHER_CSV_OPTIONS}&source=total&"
        "metric=demand&frequency=annual"
    ),
    "net_imports": (
        f"{ELECTRICITY_MIX_URL}.csv?{GRAPHER_CSV_OPTIONS}&source=total&"
        "metric=net_imports&frequency=annual"
    ),
    "electricity_per_capita": (
        f"{ELECTRICITY_MIX_URL}.csv?{GRAPHER_CSV_OPTIONS}&source=total&"
        "metric=per_capita&frequency=annual"
    ),
    "population": f"{GRAPHER_BASE_URL}/population.csv?{GRAPHER_CSV_OPTIONS}",
    "gdp_per_capita": f"{GRAPHER_BASE_URL}/gdp-per-capita-worldbank.csv?{GRAPHER_CSV_OPTIONS}",
    "target": (
        f"{ELECTRICITY_MIX_URL}.csv?{GRAPHER_CSV_OPTIONS}&source=total&"
        "metric=carbon_intensity&frequency=annual"
    ),
    "target_metadata": (
        f"{ELECTRICITY_MIX_URL}.metadata.json?{GRAPHER_CSV_OPTIONS}&"
        "source=total&metric=carbon_intensity&frequency=annual"
    ),
}

SOURCE_PATHS = {
    key: RAW_DIR / f"owid-grapher-{key.replace('_', '-')}.csv"
    for key in SOURCE_URLS
    if key not in {"target", "target_metadata"}
}
SOURCE_PATHS["target"] = TARGET_PATH
SOURCE_PATHS["target_metadata"] = TARGET_METADATA_PATH

BASE_FEATURES = [
    "coal_share_elec",
    "gas_share_elec",
    "oil_share_elec",
    "hydro_share_elec",
    "nuclear_share_elec",
    "wind_share_elec",
    "solar_share_elec",
    "biofuel_share_elec",
    "other_renewables_share_elec",
    "net_elec_imports_share_demand",
    "per_capita_electricity",
]

MODEL_FEATURES = [
    "coal_share_elec",
    "gas_share_elec",
    "oil_share_elec",
    "hydro_share_elec",
    "nuclear_share_elec",
    "wind_share_elec",
    "solar_share_elec",
    "biofuel_share_elec",
    "other_renewables_share_elec",
    "net_elec_imports_share_demand",
    "log_per_capita_electricity",
    "year",
]

MODEL_LABELS = {
    "linear_regression": "Linear Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}


def ensure_directories() -> None:
    for directory in (PROCESSED_DIR, FIGURE_DIR, TABLE_DIR, MODEL_DIR, PAPER_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def _read_grapher_csv(key: str, value_columns: list[str]) -> pd.DataFrame:
    """Read one downloaded Grapher CSV and enforce its expected schema."""
    frame = pd.read_csv(SOURCE_PATHS[key])
    required = {"Entity", "Code", "Year", *value_columns}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"OWID Grapher source {key!r} is missing columns: {sorted(missing)}"
        )
    frame = frame.loc[frame["Code"].notna()].copy()
    frame = frame.rename(
        columns={"Entity": "country", "Code": "iso_code", "Year": "year"}
    )
    frame["year"] = pd.to_numeric(frame["year"], errors="raise").astype(int)
    if frame.duplicated(["iso_code", "year"]).any():
        raise ValueError(f"OWID Grapher source {key!r} has duplicate Code-Year rows")
    return frame[["country", "iso_code", "year", *value_columns]]


def _build_energy_inputs() -> None:
    """Normalize official Grapher indicators to the workflow's input columns."""
    source_columns = [
        "Other renewables",
        "Bioenergy",
        "Solar",
        "Wind",
        "Hydropower",
        "Nuclear",
        "Gas",
        "Oil",
        "Coal",
    ]
    mix = _read_grapher_csv("electricity_by_source", source_columns)
    mix = mix.rename(
        columns={
            "Other renewables": "other_renewables_generation",
            "Bioenergy": "biofuel_generation",
            "Solar": "solar_generation",
            "Wind": "wind_generation",
            "Hydropower": "hydro_generation",
            "Nuclear": "nuclear_generation",
            "Gas": "gas_generation",
            "Oil": "oil_generation",
            "Coal": "coal_generation",
        }
    )
    keys = ["iso_code", "year"]

    indicators = {
        "electricity_generation": ("electricity_generation", "Total electricity"),
        "electricity_demand": ("electricity_demand", "Electricity demand"),
        "net_imports": ("net_imports", "Net imports"),
        "electricity_per_capita": ("electricity_per_capita", "Total electricity"),
        "population": ("population", "Population"),
        "gdp_per_capita": ("gdp_per_capita", "GDP per capita"),
    }
    for source, (key, value_column) in indicators.items():
        indicator = _read_grapher_csv(source, [value_column]).rename(
            columns={value_column: key}
        )
        mix = mix.merge(
            indicator[keys + [key]],
            on=keys,
            how="left",
            validate="one_to_one",
        )

    denominator = mix["electricity_generation"].replace(0, np.nan)
    generation_to_share = {
        "coal_generation": "coal_share_elec",
        "gas_generation": "gas_share_elec",
        "oil_generation": "oil_share_elec",
        "hydro_generation": "hydro_share_elec",
        "nuclear_generation": "nuclear_share_elec",
        "wind_generation": "wind_share_elec",
        "solar_generation": "solar_share_elec",
        "biofuel_generation": "biofuel_share_elec",
    }
    for generation, share in generation_to_share.items():
        mix[share] = mix[generation].div(denominator).mul(100)

    mix["other_renewables_share_elec"] = (
        mix["other_renewables_generation"] + mix["biofuel_generation"]
    ).div(denominator).mul(100)
    mix["renewables_share_elec"] = (
        mix["other_renewables_generation"]
        + mix["biofuel_generation"]
        + mix["solar_generation"]
        + mix["wind_generation"]
        + mix["hydro_generation"]
    ).div(denominator).mul(100)
    mix["net_elec_imports_share_demand"] = (
        mix["net_imports"].div(mix["electricity_demand"].replace(0, np.nan)).mul(100)
    )
    mix["other_renewables_electricity"] = mix["other_renewables_generation"]
    mix["per_capita_electricity"] = mix["electricity_per_capita"]
    mix["gdp"] = mix["gdp_per_capita"]

    normalized_columns = [
        "country",
        "year",
        "iso_code",
        "population",
        "gdp",
        "biofuel_share_elec",
        "coal_share_elec",
        "gas_share_elec",
        "hydro_share_elec",
        "nuclear_share_elec",
        "oil_share_elec",
        "other_renewables_share_elec",
        "per_capita_electricity",
        "renewables_share_elec",
        "solar_share_elec",
        "wind_share_elec",
        "electricity_generation",
        "electricity_demand",
        "net_elec_imports_share_demand",
        "other_renewables_electricity",
    ]
    mix[normalized_columns].to_csv(ENERGY_PATH, index=False)


def download_sources(force: bool = True) -> list[str]:
    """Download source CSVs directly from OWID Grapher and normalize them."""
    ensure_directories()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    downloaded: list[str] = []
    headers = {"User-Agent": "OWID-carbon-intensity-research/1.0"}
    for key, destination in SOURCE_PATHS.items():
        if destination.exists() and not force:
            continue
        temporary = destination.with_suffix(destination.suffix + ".part")
        try:
            with requests.get(
                SOURCE_URLS[key], headers=headers, timeout=120, stream=True
            ) as response:
                response.raise_for_status()
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        downloaded.append(str(destination))
    _build_energy_inputs()
    return downloaded


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_sources() -> dict[str, Any]:
    """Validate downloaded snapshots and write machine-readable provenance."""
    ensure_directories()
    required = [ENERGY_PATH, *SOURCE_PATHS.values()]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing raw OWID snapshots. Run the download commands in README.md: "
            + ", ".join(missing)
        )

    energy = pd.read_csv(ENERGY_PATH)
    target = pd.read_csv(TARGET_PATH)
    expected_target = {"Entity", "Code", "Year", "Carbon intensity"}
    if not expected_target.issubset(target.columns):
        raise ValueError(f"Unexpected target schema: {target.columns.tolist()}")
    expected_energy = {"country", "iso_code", "year", *BASE_FEATURES}
    absent = expected_energy.difference(energy.columns)
    if absent:
        raise ValueError(f"OWID Energy is missing required columns: {sorted(absent)}")
    country_target = target.loc[target["Code"].notna()]
    if country_target.duplicated(["Code", "Year"]).any():
        raise ValueError("Target snapshot has duplicate Code-Year rows")

    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_urls": SOURCE_URLS,
        "files": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "modified_utc": datetime.fromtimestamp(
                    path.stat().st_mtime, tz=timezone.utc
                ).isoformat(),
            }
            for path in required
        },
        "energy_shape": list(energy.shape),
        "energy_year_range": [int(energy["year"].min()), int(energy["year"].max())],
        "target_shape": list(target.shape),
        "target_year_range": [int(target["Year"].min()), int(target["Year"].max())],
        "target_unit": "gCO2e/kWh (lifecycle)",
        "target_source": "Ember (2026), processed by Our World in Data",
    }
    (TABLE_DIR / "data_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return provenance


def prepare_dataset() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Merge sources, enforce inclusion rules, and create the modeling table."""
    ensure_directories()
    energy = pd.read_csv(ENERGY_PATH, low_memory=False)
    target = pd.read_csv(TARGET_PATH).rename(
        columns={
            "Entity": "country",
            "Code": "iso_code",
            "Year": "year",
            "Carbon intensity": "carbon_intensity_gco2e_kwh",
        }
    )

    target = target.loc[
        target["iso_code"].notna() & target["year"].between(2000, 2025)
    ].copy()
    energy = energy.loc[energy["iso_code"].notna()].copy()
    merged = target.merge(
        energy[["iso_code", "year", *BASE_FEATURES]],
        how="left",
        on=["iso_code", "year"],
        validate="one_to_one",
    )
    merged["available_feature_count"] = merged[BASE_FEATURES].notna().sum(axis=1)
    before_feature_rule = len(merged)
    merged = merged.loc[merged["available_feature_count"] >= 8].copy()
    merged["log_per_capita_electricity"] = np.log1p(
        merged["per_capita_electricity"].clip(lower=0)
    )
    merged = merged.sort_values(["year", "iso_code"]).reset_index(drop=True)

    target_col = "carbon_intensity_gco2e_kwh"
    if merged[target_col].isna().any() or not np.isfinite(merged[target_col]).all():
        raise ValueError("Target contains missing or non-finite values after filtering")
    if (merged[target_col] < 0).any():
        raise ValueError("Target contains negative carbon-intensity values")

    def split_name(year: int) -> str:
        if year <= 2018:
            return "train"
        if year <= 2021:
            return "validation"
        return "test"

    merged["split"] = merged["year"].map(split_name)
    expected_splits = {
        "train": (2000, 2018),
        "validation": (2019, 2021),
        "test": (2022, 2025),
    }
    for name, (start, end) in expected_splits.items():
        years = merged.loc[merged["split"] == name, "year"]
        if years.empty or years.min() != start or years.max() != end:
            raise ValueError(f"Unexpected {name} years: {sorted(years.unique())}")

    keep_columns = [
        "country",
        "iso_code",
        "year",
        "split",
        target_col,
        "available_feature_count",
        *[feature for feature in MODEL_FEATURES if feature != "year"],
    ]
    model_data = merged[keep_columns]
    model_data.to_csv(PROCESSED_DIR / "carbon_intensity_modeling_data.csv", index=False)

    missingness = (
        model_data[MODEL_FEATURES]
        .isna()
        .mean()
        .mul(100)
        .rename("missing_percent")
        .rename_axis("feature")
        .reset_index()
    )
    missingness.to_csv(TABLE_DIR / "feature_missingness.csv", index=False)

    split_summary = (
        model_data.groupby("split", sort=False)
        .agg(
            rows=("iso_code", "size"),
            countries=("iso_code", "nunique"),
            year_min=("year", "min"),
            year_max=("year", "max"),
            target_mean=(target_col, "mean"),
            target_std=(target_col, "std"),
        )
        .reset_index()
    )
    split_summary.to_csv(TABLE_DIR / "split_summary.csv", index=False)

    audit = {
        "rows_before_feature_rule": int(before_feature_rule),
        "rows_after_feature_rule": int(len(model_data)),
        "excluded_for_sparse_features": int(before_feature_rule - len(model_data)),
        "countries": int(model_data["iso_code"].nunique()),
        "year_min": int(model_data["year"].min()),
        "year_max": int(model_data["year"].max()),
        "duplicate_country_year": int(
            model_data.duplicated(["iso_code", "year"]).sum()
        ),
        "split_rows": {
            row["split"]: int(row["rows"]) for _, row in split_summary.iterrows()
        },
    }
    (TABLE_DIR / "data_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _plot_dataset_overview(model_data)
    return model_data, audit


def _plot_dataset_overview(data: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    target_col = "carbon_intensity_gco2e_kwh"
    yearly = data.groupby("year").agg(
        observations=(target_col, "size"), mean_intensity=(target_col, "mean")
    )
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].bar(yearly.index, yearly["observations"], color="#4472C4")
    axes[0].set_ylabel("Country observations")
    axes[0].set_title("OWID electricity carbon-intensity panel coverage")
    axes[1].plot(yearly.index, yearly["mean_intensity"], marker="o", color="#C00000")
    axes[1].set_ylabel("Mean gCO$_2$e/kWh")
    axes[1].set_xlabel("Year")
    for boundary in (2018.5, 2021.5):
        for axis in axes:
            axis.axvline(boundary, color="black", linestyle="--", linewidth=1)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "dataset_coverage_and_trend.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _linear_pipeline() -> Pipeline:
    # Scaling is fit only on the training partition by the pipeline.
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )


def _tree_pipeline(model: Any) -> Pipeline:
    return Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("model", model)]
    )


def _candidate_models() -> dict[str, list[tuple[dict[str, Any], Pipeline]]]:
    rf_candidates: list[tuple[dict[str, Any], Pipeline]] = []
    for max_depth in (None, 16):
        for min_samples_leaf in (1, 3):
            params = {
                "n_estimators": 500,
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
                "max_features": 0.8,
            }
            model = RandomForestRegressor(
                **params, random_state=SEED, n_jobs=-1
            )
            rf_candidates.append((params, _tree_pipeline(model)))

    xgb_candidates: list[tuple[dict[str, Any], Pipeline]] = []
    for n_estimators in (400, 700):
        for max_depth in (3, 6):
            for learning_rate in (0.03, 0.07):
                params = {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "learning_rate": learning_rate,
                    "subsample": 0.9,
                    "colsample_bytree": 0.9,
                    "min_child_weight": 2,
                    "reg_lambda": 1.0,
                }
                model = XGBRegressor(
                    **params,
                    objective="reg:squarederror",
                    random_state=SEED,
                    n_jobs=-1,
                    tree_method="hist",
                )
                xgb_candidates.append((params, _tree_pipeline(model)))

    return {
        "linear_regression": [({}, _linear_pipeline())],
        "random_forest": rf_candidates,
        "xgboost": xgb_candidates,
    }


def train_evaluate_and_report(
    data: pd.DataFrame, audit: dict[str, Any]
) -> dict[str, Any]:
    """Tune on validation years, refit on train+validation, and evaluate once."""
    ensure_directories()
    target_col = "carbon_intensity_gco2e_kwh"
    partitions = {
        name: data.loc[data["split"] == name].copy()
        for name in ("train", "validation", "test")
    }
    train = partitions["train"]
    validation = partitions["validation"]
    test = partitions["test"]
    X_train, y_train = train[MODEL_FEATURES], train[target_col]
    X_val, y_val = validation[MODEL_FEATURES], validation[target_col]
    X_test, y_test = test[MODEL_FEATURES], test[target_col]

    tuning_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    selected: dict[str, tuple[dict[str, Any], Pipeline]] = {}

    for model_name, candidates in _candidate_models().items():
        best: tuple[float, dict[str, Any], Pipeline, dict[str, float]] | None = None
        for params, pipeline in candidates:
            fitted = clone(pipeline).fit(X_train, y_train)
            prediction = fitted.predict(X_val)
            scores = _metrics(y_val, prediction)
            tuning_rows.append(
                {
                    "model": MODEL_LABELS[model_name],
                    "parameters": json.dumps(params, sort_keys=True),
                    **scores,
                }
            )
            if best is None or scores["rmse"] < best[0]:
                best = (scores["rmse"], params, pipeline, scores)
        assert best is not None
        _, params, pipeline, scores = best
        selected[model_name] = (params, pipeline)
        validation_rows.append(
            {
                "split": "Validation (2019-2021)",
                "model": MODEL_LABELS[model_name],
                "n": len(validation),
                **scores,
                "parameters": json.dumps(params, sort_keys=True),
            }
        )

    pd.DataFrame(tuning_rows).to_csv(TABLE_DIR / "hyperparameter_search.csv", index=False)

    train_validation = pd.concat([train, validation], ignore_index=True)
    X_train_validation = train_validation[MODEL_FEATURES]
    y_train_validation = train_validation[target_col]
    final_models: dict[str, Pipeline] = {}
    test_rows: list[dict[str, Any]] = []
    prediction_frame = test[["country", "iso_code", "year", target_col]].copy()

    for model_name, (params, pipeline) in selected.items():
        fitted = clone(pipeline).fit(X_train_validation, y_train_validation)
        prediction = fitted.predict(X_test)
        scores = _metrics(y_test, prediction)
        test_rows.append(
            {
                "split": "Test (2022-2025)",
                "model": MODEL_LABELS[model_name],
                "n": len(test),
                **scores,
                "parameters": json.dumps(params, sort_keys=True),
            }
        )
        prediction_frame[f"prediction_{model_name}"] = prediction
        final_models[model_name] = fitted
        joblib.dump(fitted, MODEL_DIR / f"{model_name}.joblib")

    metrics = pd.DataFrame(validation_rows + test_rows)
    metrics.to_csv(TABLE_DIR / "model_metrics.csv", index=False)
    prediction_frame.to_csv(TABLE_DIR / "test_predictions.csv", index=False)

    yearly_rows: list[dict[str, Any]] = []
    for year, year_frame in prediction_frame.groupby("year"):
        for model_name in final_models:
            scores = _metrics(
                year_frame[target_col], year_frame[f"prediction_{model_name}"]
            )
            yearly_rows.append(
                {
                    "year": int(year),
                    "model": MODEL_LABELS[model_name],
                    "n": len(year_frame),
                    **scores,
                }
            )
    pd.DataFrame(yearly_rows).to_csv(TABLE_DIR / "test_metrics_by_year.csv", index=False)

    test_metrics = metrics.loc[metrics["split"].str.startswith("Test")]
    best_row = test_metrics.sort_values("rmse").iloc[0]
    best_label = str(best_row["model"])
    best_key = next(key for key, label in MODEL_LABELS.items() if label == best_label)

    _plot_model_results(metrics, prediction_frame, best_key, best_label)
    importance = _explain_models(final_models, X_test, y_test)
    _write_environment()
    _write_generated_latex(metrics, audit, data, importance, best_row)

    summary = {
        "best_test_model": best_label,
        "best_test_mae": float(best_row["mae"]),
        "best_test_rmse": float(best_row["rmse"]),
        "best_test_r2": float(best_row["r2"]),
        "test_rows": int(len(test)),
        "metrics_path": str(TABLE_DIR / "model_metrics.csv"),
        "paper_macros_path": str(PAPER_DIR / "generated_results.tex"),
    }
    (TABLE_DIR / "experiment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def _plot_model_results(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    best_key: str,
    best_label: str,
) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    test_metrics = metrics.loc[metrics["split"].str.startswith("Test")].copy()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for axis, metric, label in zip(
        axes, ("mae", "rmse", "r2"), ("MAE", "RMSE", "$R^2$")
    ):
        sns.barplot(data=test_metrics, x="model", y=metric, ax=axis, color="#4472C4")
        axis.set_title(label)
        axis.set_xlabel("")
        axis.tick_params(axis="x", rotation=20)
    fig.suptitle("Held-out temporal test performance (2022-2025)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "model_comparison_test.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    target_col = "carbon_intensity_gco2e_kwh"
    pred_col = f"prediction_{best_key}"
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.scatterplot(
        data=predictions,
        x=target_col,
        y=pred_col,
        hue="year",
        palette="viridis",
        alpha=0.72,
        ax=ax,
    )
    limits = [
        min(predictions[target_col].min(), predictions[pred_col].min()),
        max(predictions[target_col].max(), predictions[pred_col].max()),
    ]
    ax.plot(limits, limits, "k--", linewidth=1.5)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("Observed gCO$_2$e/kWh")
    ax.set_ylabel("Predicted gCO$_2$e/kWh")
    ax.set_title(f"Observed vs. predicted: {best_label}")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "actual_vs_predicted_best.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    residuals = predictions.copy()
    residuals["residual"] = residuals[target_col] - residuals[pred_col]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=residuals, x="year", y="residual", color="#A5A5A5", ax=ax)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("Observed - predicted (gCO$_2$e/kWh)")
    ax.set_title(f"Test residuals by year: {best_label}")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "residuals_by_year_best.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _explain_models(
    final_models: dict[str, Pipeline], X_test: pd.DataFrame, y_test: pd.Series
) -> pd.DataFrame:
    native_rows: list[dict[str, Any]] = []
    for model_name in ("random_forest", "xgboost"):
        estimator = final_models[model_name].named_steps["model"]
        for feature, value in zip(MODEL_FEATURES, estimator.feature_importances_):
            native_rows.append(
                {
                    "model": MODEL_LABELS[model_name],
                    "feature": feature,
                    "importance": float(value),
                }
            )
    native = pd.DataFrame(native_rows)
    native.to_csv(TABLE_DIR / "native_feature_importance.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
    for axis, model_name in zip(axes, ("Random Forest", "XGBoost")):
        subset = native.loc[native["model"] == model_name].nlargest(10, "importance")
        sns.barplot(
            data=subset,
            x="importance",
            y="feature",
            ax=axis,
            color="#70AD47",
        )
        axis.set_title(model_name)
        axis.set_xlabel("Native feature importance")
        axis.set_ylabel("")
    fig.suptitle("Tree-model feature importance")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "tree_feature_importance.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    xgb_pipeline = final_models["xgboost"]
    imputer = xgb_pipeline.named_steps["imputer"]
    estimator = xgb_pipeline.named_steps["model"]
    sample = X_test.sample(n=min(500, len(X_test)), random_state=SEED)
    transformed = imputer.transform(sample)
    explainer = shap.TreeExplainer(estimator)
    explanation = explainer(transformed)
    explanation.feature_names = MODEL_FEATURES
    shap.plots.beeswarm(explanation, max_display=12, show=False)
    plt.title("XGBoost SHAP values on temporal test sample")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "xgboost_shap_beeswarm.png", dpi=220, bbox_inches="tight")
    plt.close()

    shap_importance = pd.DataFrame(
        {
            "feature": MODEL_FEATURES,
            "mean_abs_shap": np.abs(explanation.values).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    shap_importance.to_csv(TABLE_DIR / "xgboost_shap_importance.csv", index=False)

    permutation = permutation_importance(
        xgb_pipeline,
        X_test,
        y_test,
        n_repeats=20,
        random_state=SEED,
        n_jobs=1,
        scoring="neg_root_mean_squared_error",
    )
    permutation_frame = pd.DataFrame(
        {
            "feature": MODEL_FEATURES,
            "permutation_importance_mean": permutation.importances_mean,
            "permutation_importance_std": permutation.importances_std,
        }
    ).sort_values("permutation_importance_mean", ascending=False)
    permutation_frame.to_csv(TABLE_DIR / "xgboost_permutation_importance.csv", index=False)
    return shap_importance


def _tex_escape(value: str) -> str:
    replacements = {
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }
    result = value
    for source, target in replacements.items():
        result = result.replace(source, target)
    return result


def _write_generated_latex(
    metrics: pd.DataFrame,
    audit: dict[str, Any],
    data: pd.DataFrame,
    shap_importance: pd.DataFrame,
    best_row: pd.Series,
) -> None:
    split_counts = data.groupby("split").size()
    macros = [
        "% Auto-generated by src/carbon_intensity_workflow.py. Do not edit manually.",
        rf"\newcommand{{\DatasetRows}}{{{len(data)}}}",
        rf"\newcommand{{\DatasetCountries}}{{{audit['countries']}}}",
        rf"\newcommand{{\TrainRows}}{{{int(split_counts['train'])}}}",
        rf"\newcommand{{\ValidationRows}}{{{int(split_counts['validation'])}}}",
        rf"\newcommand{{\TestRows}}{{{int(split_counts['test'])}}}",
        rf"\newcommand{{\ExcludedRows}}{{{audit['excluded_for_sparse_features']}}}",
        rf"\newcommand{{\BestModel}}{{{_tex_escape(str(best_row['model']))}}}",
        rf"\newcommand{{\BestTestMAE}}{{{best_row['mae']:.2f}}}",
        rf"\newcommand{{\BestTestRMSE}}{{{best_row['rmse']:.2f}}}",
        rf"\newcommand{{\BestTestRSquared}}{{{best_row['r2']:.3f}}}",
        rf"\newcommand{{\TopSHAPFeature}}{{\texttt{{{_tex_escape(str(shap_importance.iloc[0]['feature']))}}}}}",
    ]
    (PAPER_DIR / "generated_results.tex").write_text(
        "\n".join(macros) + "\n", encoding="utf-8"
    )

    ordered = metrics.sort_values(["split", "rmse"])
    lines = [
        "% Auto-generated experimental metrics.",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Tập dữ liệu & Mô hình & $n$ & MAE & RMSE & $R^2$ \\",
        r"\midrule",
    ]
    for _, row in ordered.iterrows():
        lines.append(
            f"{_tex_escape(str(row['split']))} & {_tex_escape(str(row['model']))} & "
            f"{int(row['n'])} & {row['mae']:.2f} & {row['rmse']:.2f} & {row['r2']:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (PAPER_DIR / "generated_metrics_table.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    feature_lines = [
        "% Auto-generated SHAP ranking.",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Đặc trưng & Mean $|$SHAP$|$ \\",
        r"\midrule",
    ]
    for _, row in shap_importance.head(8).iterrows():
        feature_lines.append(
            rf"\texttt{{{_tex_escape(str(row['feature']))}}} & {row['mean_abs_shap']:.2f} \\"
        )
    feature_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (PAPER_DIR / "generated_shap_table.tex").write_text(
        "\n".join(feature_lines) + "\n", encoding="utf-8"
    )


def _write_environment() -> None:
    versions = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "shap": shap.__version__,
        "seed": SEED,
        "features": MODEL_FEATURES,
        "split": {
            "train": "2000-2018",
            "validation": "2019-2021",
            "test": "2022-2025",
        },
    }
    (TABLE_DIR / "environment.json").write_text(
        json.dumps(versions, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_experiment() -> dict[str, Any]:
    download_sources(force=False)
    provenance = audit_sources()
    data, audit = prepare_dataset()
    summary = train_evaluate_and_report(data, audit)
    summary["target_snapshot_year_max"] = provenance["target_year_range"][1]
    return summary


if __name__ == "__main__":
    print(json.dumps(run_experiment(), ensure_ascii=False, indent=2))
