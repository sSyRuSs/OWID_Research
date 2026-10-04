"""End-to-end, evidence-producing workflow for the explainable carbon-intensity thesis.

All paper numbers are generated from artifacts written by this module. The three
evaluation protocols are fixed before model fitting: temporal, unseen-country,
and region-transfer.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import country_converter as coco
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import sklearn
import statsmodels.api as sm
import xgboost
from catboost import CatBoostRegressor
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import PartialDependenceDisplay, partial_dependence, permutation_importance
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from xgboost import XGBRegressor

from carbon_intensity_workflow import (
    ENERGY_PATH,
    SOURCE_PATHS,
    SOURCE_URLS,
    TARGET_PATH,
    audit_sources,
    download_sources,
)


SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs" / "thesis"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
MODEL_DIR = OUTPUT_DIR / "models"
PAPER_DIR = ROOT / "paper"
DOCS_DIR = ROOT / "docs"

TARGET = "carbon_intensity_gco2e_kwh"

# Complete panel variables retained for EDA, traceability and sensitivity work.
PANEL_FEATURES = [
    "coal_share_elec",
    "gas_share_elec",
    "oil_share_elec",
    "solar_share_elec",
    "wind_share_elec",
    "hydro_share_elec",
    "nuclear_share_elec",
    "renewables_share_elec",
    "biofuel_share_elec",
    "other_renewables_share_elec",
    "electricity_generation",
    "electricity_demand",
    "per_capita_electricity",
    "net_elec_imports_share_demand",
    "population",
    "gdp",
]

# Main specification: detailed mix with oil as reference category. This choice
# was made from a prespecified VIF audit over all possible single-share
# references; it produces the lowest maximum training VIF while retaining oil
# in the complete panel for descriptive and scenario analysis.
# Aggregate renewables, generation and per-capita electricity remain in the panel
# but are excluded here because they are algebraically or nearly algebraically
# derived from included variables.
MODEL_FEATURES = [
    "coal_share_elec",
    "gas_share_elec",
    "solar_share_elec",
    "wind_share_elec",
    "hydro_share_elec",
    "nuclear_share_elec",
    "biofuel_share_elec",
    "log_electricity_demand",
    "net_elec_imports_share_demand",
    "log_population",
    "year_centered",
]

SOURCE_SHARE_FEATURES = [
    "coal_share_elec",
    "gas_share_elec",
    "oil_share_elec",
    "solar_share_elec",
    "wind_share_elec",
    "hydro_share_elec",
    "nuclear_share_elec",
    "biofuel_share_elec",
    "other_renewables_share_elec",
]

REQUIRED_RAW_MODEL_FEATURES = [
    *SOURCE_SHARE_FEATURES[:-1],
    "electricity_demand",
    "net_elec_imports_share_demand",
    "population",
]

MODEL_LABELS = {
    "linear_regression": "Linear Regression",
    "ridge": "Ridge",
    "elastic_net": "Elastic Net",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
}


def ensure_directories() -> None:
    for path in (
        DATA_PROCESSED,
        FIGURE_DIR,
        TABLE_DIR,
        MODEL_DIR / "temporal",
        MODEL_DIR / "unseen_country",
        MODEL_DIR / "region_transfer",
        PAPER_DIR,
        DOCS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def _json_write(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _is_country_code(code: Any) -> bool:
    # OWID aggregates use OWID_* codes. Kosovo is the only country retained with
    # this representation in the downloaded data.
    return isinstance(code, str) and ((len(code) == 3 and code.isalpha()) or code == "OWID_KOS")


def _regions_for_codes(codes: pd.Series) -> pd.Series:
    converter = coco.CountryConverter()
    result: dict[str, str] = {"OWID_KOS": "Europe"}
    standard = sorted(code for code in codes.dropna().unique() if code != "OWID_KOS")
    converted = converter.convert(names=standard, src="ISO3", to="continent")
    result.update(dict(zip(standard, converted)))
    mapped = codes.map(result)
    if mapped.isna().any() or (mapped == "not found").any():
        bad = sorted(codes.loc[mapped.isna() | (mapped == "not found")].unique())
        raise ValueError(f"Unmapped country regions: {bad}")
    return mapped


def _temporal_split(year: int) -> str:
    if year <= 2018:
        return "train"
    if year <= 2021:
        return "validation"
    return "test"


def _assign_unseen_country_split(codes: pd.Series) -> dict[str, str]:
    unique = np.array(sorted(codes.unique()))
    rng = np.random.default_rng(SEED)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    n_test = max(1, round(len(shuffled) * 0.20))
    n_validation = max(1, round(len(shuffled) * 0.20))
    test = set(shuffled[:n_test])
    validation = set(shuffled[n_test : n_test + n_validation])
    return {
        code: "test" if code in test else "validation" if code in validation else "train"
        for code in unique
    }


def _dominant_source(frame: pd.DataFrame) -> pd.Series:
    labels = {
        "coal_share_elec": "Coal",
        "gas_share_elec": "Gas",
        "oil_share_elec": "Oil",
        "solar_share_elec": "Solar",
        "wind_share_elec": "Wind",
        "hydro_share_elec": "Hydro",
        "nuclear_share_elec": "Nuclear",
        "biofuel_share_elec": "Biofuel",
        "other_renewables_share_elec": "Other renewables",
    }
    return frame[SOURCE_SHARE_FEATURES].idxmax(axis=1).map(labels)


def prepare_panel() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create the country-year panel and all data-quality audit artifacts."""
    ensure_directories()
    energy = pd.read_csv(ENERGY_PATH, low_memory=False)
    target = pd.read_csv(TARGET_PATH).rename(
        columns={
            "Entity": "country",
            "Code": "iso_code",
            "Year": "year",
            "Carbon intensity": TARGET,
        }
    )
    target = target.loc[
        target["iso_code"].notna()
        & target["iso_code"].map(_is_country_code)
        & target["year"].between(2000, 2025)
    ].copy()
    energy = energy.loc[
        energy["iso_code"].notna() & energy["iso_code"].map(_is_country_code)
    ].copy()

    panel = target.merge(
        energy[["iso_code", "year", *PANEL_FEATURES]],
        how="left",
        on=["iso_code", "year"],
        validate="one_to_one",
    )
    panel["region"] = _regions_for_codes(panel["iso_code"])

    invalid_share_counts: dict[str, int] = {}
    for feature in SOURCE_SHARE_FEATURES + ["renewables_share_elec"]:
        invalid = panel[feature].notna() & ~panel[feature].between(0, 100)
        invalid_share_counts[feature] = int(invalid.sum())
        panel.loc[invalid, feature] = np.nan

    panel["available_model_inputs"] = panel[REQUIRED_RAW_MODEL_FEATURES].notna().sum(axis=1)
    before_sparse_rule = len(panel)
    panel = panel.loc[panel["available_model_inputs"] >= 9].copy()
    panel["log_electricity_demand"] = np.log1p(panel["electricity_demand"].clip(lower=0))
    panel["log_population"] = np.log1p(panel["population"].clip(lower=0))
    panel["year_centered"] = panel["year"] - 2000
    panel["dominant_source"] = _dominant_source(panel)
    panel["low_carbon_share_elec"] = (
        panel["renewables_share_elec"] + panel["nuclear_share_elec"]
    )
    panel["temporal_split"] = panel["year"].map(_temporal_split)
    country_map = _assign_unseen_country_split(panel["iso_code"])
    panel["unseen_country_split"] = panel["iso_code"].map(country_map)

    q1, q3 = panel[TARGET].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    panel["target_outlier_iqr"] = ~panel[TARGET].between(lower, upper)
    panel = panel.sort_values(["year", "iso_code"]).reset_index(drop=True)

    if panel.duplicated(["iso_code", "year"]).any():
        raise ValueError("Duplicate country-year rows remain")
    if panel[TARGET].isna().any() or (panel[TARGET] < 0).any():
        raise ValueError("Invalid target values remain")
    if set(panel["region"].unique()) != {"Africa", "America", "Asia", "Europe", "Oceania"}:
        raise ValueError(f"Unexpected regions: {sorted(panel['region'].unique())}")

    panel.to_csv(DATA_PROCESSED / "thesis_country_year_panel.csv", index=False)
    _write_data_audits(panel)
    audit = {
        "rows_before_sparse_rule": int(before_sparse_rule),
        "rows_after_sparse_rule": int(len(panel)),
        "rows_excluded_sparse": int(before_sparse_rule - len(panel)),
        "countries": int(panel["iso_code"].nunique()),
        "regions": int(panel["region"].nunique()),
        "year_min": int(panel["year"].min()),
        "year_max": int(panel["year"].max()),
        "target_outliers_iqr": int(panel["target_outlier_iqr"].sum()),
        "target_outlier_lower": float(lower),
        "target_outlier_upper": float(upper),
        "invalid_share_counts": invalid_share_counts,
        "temporal_rows": panel["temporal_split"].value_counts().to_dict(),
        "unseen_country_counts": panel.groupby("unseen_country_split")["iso_code"].nunique().to_dict(),
    }
    _json_write(TABLE_DIR / "data_audit.json", audit)
    return panel, audit


def _write_data_audits(panel: pd.DataFrame) -> None:
    missing = (
        panel[PANEL_FEATURES + [TARGET]]
        .isna()
        .mean()
        .mul(100)
        .rename("missing_percent")
        .rename_axis("variable")
        .reset_index()
    )
    missing.to_csv(TABLE_DIR / "panel_missingness.csv", index=False)

    split_summary = (
        panel.groupby("temporal_split", sort=False)
        .agg(
            rows=("iso_code", "size"),
            countries=("iso_code", "nunique"),
            year_min=("year", "min"),
            year_max=("year", "max"),
            target_mean=(TARGET, "mean"),
            target_std=(TARGET, "std"),
        )
        .reset_index()
    )
    split_summary.to_csv(TABLE_DIR / "temporal_split_summary.csv", index=False)

    country_split = (
        panel.groupby("unseen_country_split")
        .agg(rows=("iso_code", "size"), countries=("iso_code", "nunique"))
        .reset_index()
    )
    country_split.to_csv(TABLE_DIR / "unseen_country_split_summary.csv", index=False)

    leakage_rows = [
        ("carbon_intensity_elec", "excluded", "Alternative target definition / direct leakage risk"),
        ("greenhouse_gas_emissions", "excluded", "Emission total mechanically linked to target"),
        ("renewables_share_elec", "panel_only", "Aggregate of detailed renewable shares"),
        ("oil_share_elec", "reference", "Omitted compositional reference; this choice minimized maximum training VIF"),
        ("other_renewables_share_elec", "panel_only", "Overlaps strongly with biofuel in the OWID feature definitions"),
        ("electricity_generation", "panel_only", "Near-identity with demand and net imports"),
        ("per_capita_electricity", "panel_only", "Derived from generation and population"),
        ("gdp", "panel_only", "33% missing and unavailable for much of held-out period"),
    ]
    pd.DataFrame(leakage_rows, columns=["variable", "status", "reason"]).to_csv(
        TABLE_DIR / "leakage_and_redundancy_audit.csv", index=False
    )

    train = panel.loc[panel["temporal_split"] == "train", MODEL_FEATURES]
    imputed = SimpleImputer(strategy="median").fit_transform(train)
    vif_matrix = sm.add_constant(pd.DataFrame(imputed, columns=MODEL_FEATURES))
    vif = pd.DataFrame(
        {
            "feature": MODEL_FEATURES,
            "vif": [
                variance_inflation_factor(vif_matrix.values, index)
                for index in range(1, vif_matrix.shape[1])
            ],
        }
    ).sort_values("vif", ascending=False)
    vif.to_csv(TABLE_DIR / "vif_train.csv", index=False)

    outliers = panel.loc[panel["target_outlier_iqr"], ["country", "iso_code", "year", TARGET]]
    outliers.to_csv(TABLE_DIR / "target_outliers_iqr.csv", index=False)

    catalog = pd.DataFrame(
        [
            {
                "variable": variable,
                "role": "target" if variable == TARGET else "model_input" if variable in MODEL_FEATURES else "panel_only",
                "transformation": "log1p" if variable in {"log_electricity_demand", "log_population"} else "none",
            }
            for variable in [TARGET, *PANEL_FEATURES, *MODEL_FEATURES]
        ]
    ).drop_duplicates("variable")
    catalog.to_csv(TABLE_DIR / "feature_catalog.csv", index=False)
    _plot_data_audits(panel, vif)


def _plot_data_audits(panel: pd.DataFrame, vif: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    yearly = panel.groupby("year").agg(observations=(TARGET, "size"), mean_intensity=(TARGET, "mean"))
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].bar(yearly.index, yearly["observations"], color="#2F5597")
    axes[0].set_ylabel("Country observations")
    axes[0].set_title("Country-year coverage after quality filters")
    axes[1].plot(yearly.index, yearly["mean_intensity"], color="#C00000", marker="o")
    axes[1].set_ylabel("Mean gCO$_2$e/kWh")
    axes[1].set_xlabel("Year")
    for boundary in (2018.5, 2021.5):
        for axis in axes:
            axis.axvline(boundary, color="black", linestyle="--", linewidth=1)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "panel_coverage_trend.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    corr_data = panel.loc[panel["temporal_split"] == "train", MODEL_FEATURES].copy()
    corr_data = corr_data.fillna(corr_data.median(numeric_only=True))
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_data.corr(), cmap="vlag", center=0, square=True, ax=ax)
    ax.set_title("Training-feature correlation matrix")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "feature_correlation_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    vif_plot = vif.copy()
    finite_max = vif_plot.loc[np.isfinite(vif_plot["vif"]), "vif"].max()
    cap = max(100.0, float(finite_max) * 1.1 if pd.notna(finite_max) else 100.0)
    vif_plot["vif"] = vif_plot["vif"].replace([np.inf, -np.inf], cap)
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=vif_plot.sort_values("vif"), x="vif", y="feature", color="#A5A5A5", ax=ax)
    ax.axvline(10, color="#C00000", linestyle="--", label="VIF = 10")
    ax.set_title("Variance inflation factors on temporal training data")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "vif_audit.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=panel, x="region", y=TARGET, color="#5B9BD5", ax=ax)
    ax.set_title("Target distribution by region (outliers retained)")
    ax.set_ylabel("gCO$_2$e/kWh")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "target_outliers_by_region.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _metrics(y_true: pd.Series | np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, prediction)),
        "rmse": float(mean_squared_error(y_true, prediction) ** 0.5),
        "r2": float(r2_score(y_true, prediction)),
    }


def _scaled_pipeline(model: Any) -> Pipeline:
    return Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", model)]
    )


def _tree_pipeline(model: Any) -> Pipeline:
    return Pipeline([( "imputer", SimpleImputer(strategy="median")), ("model", model)])


def candidate_models() -> dict[str, list[tuple[dict[str, Any], Pipeline]]]:
    candidates: dict[str, list[tuple[dict[str, Any], Pipeline]]] = {
        "linear_regression": [({}, _scaled_pipeline(LinearRegression()))],
        "ridge": [],
        "elastic_net": [],
        "random_forest": [],
        "xgboost": [],
        "catboost": [],
    }
    for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
        candidates["ridge"].append(({"alpha": alpha}, _scaled_pipeline(Ridge(alpha=alpha))))
    for alpha in (0.001, 0.01, 0.1):
        for l1_ratio in (0.2, 0.7):
            params = {"alpha": alpha, "l1_ratio": l1_ratio}
            candidates["elastic_net"].append(
                (params, _scaled_pipeline(ElasticNet(**params, max_iter=30000, random_state=SEED)))
            )
    for max_depth in (None, 16):
        for min_samples_leaf in (1, 3):
            params = {
                "n_estimators": 450,
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
                "max_features": 0.8,
            }
            candidates["random_forest"].append(
                (params, _tree_pipeline(RandomForestRegressor(**params, random_state=SEED, n_jobs=-1)))
            )
    for n_estimators in (400, 700):
        for max_depth in (3, 6):
            params = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "min_child_weight": 2,
                "reg_lambda": 1.0,
            }
            candidates["xgboost"].append(
                (
                    params,
                    _tree_pipeline(
                        XGBRegressor(
                            **params,
                            objective="reg:squarederror",
                            random_state=SEED,
                            n_jobs=-1,
                            tree_method="hist",
                        )
                    ),
                )
            )
    for iterations in (500, 800):
        for depth in (4, 6):
            params = {"iterations": iterations, "depth": depth, "learning_rate": 0.05}
            candidates["catboost"].append(
                (
                    params,
                    _tree_pipeline(
                        CatBoostRegressor(
                            **params,
                            loss_function="RMSE",
                            random_seed=SEED,
                            verbose=False,
                            allow_writing_files=False,
                            thread_count=-1,
                        )
                    ),
                )
            )
    return candidates


def _fit_protocol(panel: pd.DataFrame, split_column: str, protocol: str) -> dict[str, Any]:
    """Tune on validation only, then refit on train+validation and score the test set."""
    splits = {name: panel.loc[panel[split_column] == name].copy() for name in ("train", "validation", "test")}
    if any(frame.empty for frame in splits.values()):
        raise ValueError(f"Protocol {protocol} has an empty split")
    x_train, y_train = splits["train"][MODEL_FEATURES], splits["train"][TARGET]
    x_val, y_val = splits["validation"][MODEL_FEATURES], splits["validation"][TARGET]
    x_test, y_test = splits["test"][MODEL_FEATURES], splits["test"][TARGET]

    search_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    fitted: dict[str, Pipeline] = {}
    selected_templates: dict[str, Pipeline] = {}
    selected_params: dict[str, dict[str, Any]] = {}
    predictions = splits["test"][["country", "iso_code", "year", "region", TARGET, "dominant_source"]].copy()

    for model_key, choices in candidate_models().items():
        scored: list[tuple[float, dict[str, Any], Pipeline]] = []
        for candidate_id, (params, estimator) in enumerate(choices, start=1):
            current = clone(estimator)
            current.fit(x_train, y_train)
            val_pred = current.predict(x_val)
            val_metrics = _metrics(y_val, val_pred)
            search_rows.append(
                {
                    "protocol": protocol,
                    "model": model_key,
                    "candidate_id": candidate_id,
                    "parameters": json.dumps(params, sort_keys=True),
                    **{f"validation_{key}": value for key, value in val_metrics.items()},
                }
            )
            scored.append((val_metrics["rmse"], params, estimator))
        _, best_params, best_template = min(scored, key=lambda item: item[0])
        selected_templates[model_key] = clone(best_template)
        selected_params[model_key] = best_params

        final_model = clone(best_template)
        x_development = pd.concat([x_train, x_val], ignore_index=True)
        y_development = pd.concat([y_train, y_val], ignore_index=True)
        final_model.fit(x_development, y_development)
        test_pred = final_model.predict(x_test)
        test_metrics = _metrics(y_test, test_pred)
        metric_rows.append(
            {
                "protocol": protocol,
                "split": "test",
                "model": model_key,
                "model_label": MODEL_LABELS[model_key],
                "n": len(y_test),
                "selected_parameters": json.dumps(best_params, sort_keys=True),
                **test_metrics,
            }
        )
        predictions[f"prediction_{model_key}"] = test_pred
        fitted[model_key] = final_model
        joblib.dump(final_model, MODEL_DIR / protocol / f"{model_key}.joblib")

    return {
        "search": pd.DataFrame(search_rows),
        "metrics": pd.DataFrame(metric_rows),
        "predictions": predictions,
        "models": fitted,
        "templates": selected_templates,
        "parameters": selected_params,
        "splits": splits,
    }


def _run_region_transfer(panel: pd.DataFrame, templates: dict[str, Pipeline]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for held_region in sorted(panel["region"].unique()):
        train = panel.loc[(panel["region"] != held_region) & (panel["year"] <= 2021)].copy()
        test = panel.loc[(panel["region"] == held_region) & (panel["year"] >= 2022)].copy()
        if train.empty or test.empty:
            continue
        for model_key, template in templates.items():
            model = clone(template)
            model.fit(train[MODEL_FEATURES], train[TARGET])
            prediction = model.predict(test[MODEL_FEATURES])
            rows.append(
                {
                    "protocol": "region_transfer",
                    "held_out_region": held_region,
                    "model": model_key,
                    "model_label": MODEL_LABELS[model_key],
                    "train_n": len(train),
                    "test_n": len(test),
                    "test_countries": test["iso_code"].nunique(),
                    **_metrics(test[TARGET], prediction),
                }
            )
            joblib.dump(model, MODEL_DIR / "region_transfer" / f"{held_region.lower()}_{model_key}.joblib")
    return pd.DataFrame(rows)


def _plot_model_results(metrics: pd.DataFrame, region_metrics: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, metric in zip(axes, ("mae", "rmse", "r2")):
        sns.barplot(data=metrics, x="model_label", y=metric, hue="protocol", ax=ax)
        ax.tick_params(axis="x", rotation=60)
        ax.set_xlabel("")
        ax.set_title(metric.upper())
    fig.suptitle("Hiệu năng ngoài mẫu theo giao thức đánh giá")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "model_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    pivot = region_metrics.pivot(index="model_label", columns="held_out_region", values="rmse")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax)
    ax.set_title("RMSE khi chuyển giao sang vùng chưa thấy")
    ax.set_xlabel("Vùng giữ lại")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "region_transfer_rmse.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _cluster_bootstrap_metrics(test: pd.DataFrame, prediction: np.ndarray, repetitions: int = 500) -> pd.DataFrame:
    evaluation = test[["iso_code", TARGET]].copy()
    evaluation["prediction"] = prediction
    countries = evaluation["iso_code"].unique()
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, float]] = []
    groups = {code: evaluation.loc[evaluation["iso_code"] == code] for code in countries}
    for _ in range(repetitions):
        sampled = rng.choice(countries, size=len(countries), replace=True)
        boot = pd.concat([groups[code] for code in sampled], ignore_index=True)
        rows.append(_metrics(boot[TARGET], boot["prediction"]))
    boot_df = pd.DataFrame(rows)
    return pd.DataFrame(
        {
            "metric": ["mae", "rmse", "r2"],
            "estimate": [_metrics(evaluation[TARGET], prediction)[key] for key in ("mae", "rmse", "r2")],
            "ci_low": [boot_df[key].quantile(0.025) for key in ("mae", "rmse", "r2")],
            "ci_high": [boot_df[key].quantile(0.975) for key in ("mae", "rmse", "r2")],
            "bootstrap_repetitions": repetitions,
        }
    )


def _native_importance(model: Pipeline) -> pd.DataFrame:
    estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_)
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.asarray(estimator.coef_).ravel())
    else:
        values = np.full(len(MODEL_FEATURES), np.nan)
    return pd.DataFrame({"feature": MODEL_FEATURES, "importance": values}).sort_values("importance", ascending=False)


def _explain_model(model: Pipeline, test: pd.DataFrame, model_key: str) -> dict[str, pd.DataFrame]:
    x_test = test[MODEL_FEATURES]
    y_test = test[TARGET]
    sample = x_test.sample(min(900, len(x_test)), random_state=SEED)

    native = _native_importance(model)
    native.to_csv(TABLE_DIR / "native_feature_importance.csv", index=False)
    permutation = permutation_importance(
        model,
        x_test,
        y_test,
        scoring="neg_root_mean_squared_error",
        n_repeats=20,
        random_state=SEED,
        n_jobs=1,
    )
    permutation_df = pd.DataFrame(
        {
            "feature": MODEL_FEATURES,
            "importance_mean": permutation.importances_mean,
            "importance_std": permutation.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    permutation_df.to_csv(TABLE_DIR / "permutation_importance.csv", index=False)

    transformed = model.named_steps["imputer"].transform(sample)
    estimator = model.named_steps["model"]
    if model_key in {"random_forest", "xgboost", "catboost"}:
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(transformed)
    else:
        if "scaler" in model.named_steps:
            transformed = model.named_steps["scaler"].transform(transformed)
        background = shap.sample(transformed, min(150, len(transformed)), random_state=SEED)
        explainer = shap.LinearExplainer(estimator, background)
        shap_values = explainer.shap_values(transformed)
    shap_values = np.asarray(shap_values)
    shap_df = pd.DataFrame(
        {
            "feature": MODEL_FEATURES,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            "mean_shap": shap_values.mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    shap_df.to_csv(TABLE_DIR / "shap_importance.csv", index=False)

    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values, transformed, feature_names=MODEL_FEATURES, show=False, max_display=12)
    plt.title(f"SHAP summary – {MODEL_LABELS[model_key]}")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "shap_summary.png", dpi=220, bbox_inches="tight")
    plt.close()

    plot_perm = permutation_df.head(12).sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(plot_perm["feature"], plot_perm["importance_mean"], xerr=plot_perm["importance_std"])
    ax.set_xlabel("Mức tăng RMSE khi hoán vị")
    ax.set_title("Permutation importance trên temporal test")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "permutation_importance.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    pdp_rows: list[dict[str, float | str]] = []
    pdp_features = [feature for feature in ("coal_share_elec", "solar_share_elec", "wind_share_elec", "hydro_share_elec")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for feature, ax in zip(pdp_features, axes.ravel()):
        PartialDependenceDisplay.from_estimator(model, x_test, [feature], grid_resolution=30, ax=ax)
        ax.set_title(feature)
        result = partial_dependence(model, x_test, [feature], grid_resolution=30)
        grid = result.get("grid_values", result.get("values"))[0]
        average = result["average"][0]
        pdp_rows.extend({"feature": feature, "grid_value": float(x), "partial_dependence": float(y)} for x, y in zip(grid, average))
    fig.suptitle("Partial dependence – temporal test")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "partial_dependence.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    pdp_df = pd.DataFrame(pdp_rows)
    pdp_df.to_csv(TABLE_DIR / "partial_dependence.csv", index=False)
    return {"native": native, "permutation": permutation_df, "shap": shap_df, "pdp": pdp_df, "shap_values": shap_values, "sample": sample}


def _heterogeneity_and_errors(test: pd.DataFrame, prediction: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scored = test.copy()
    scored["prediction"] = prediction
    scored["error"] = scored["prediction"] - scored[TARGET]
    scored["absolute_error"] = scored["error"].abs()
    rows: list[dict[str, Any]] = []
    for dimension in ("region", "dominant_source"):
        for group, frame in scored.groupby(dimension):
            if len(frame) < 5:
                continue
            rows.append({"dimension": dimension, "group": group, "n": len(frame), **_metrics(frame[TARGET], frame["prediction"])})
    heterogeneity = pd.DataFrame(rows)
    yearly = pd.DataFrame(
        [
            {"year": int(year), "n": len(frame), **_metrics(frame[TARGET], frame["prediction"])}
            for year, frame in scored.groupby("year")
        ]
    )
    errors = scored.sort_values("absolute_error", ascending=False).head(40)[
        ["country", "iso_code", "year", "region", "dominant_source", TARGET, "prediction", "error", "absolute_error"]
    ]
    heterogeneity.to_csv(TABLE_DIR / "heterogeneity_metrics.csv", index=False)
    yearly.to_csv(TABLE_DIR / "yearly_temporal_metrics.csv", index=False)
    errors.to_csv(TABLE_DIR / "largest_errors.csv", index=False)

    region = heterogeneity.loc[heterogeneity["dimension"] == "region"].sort_values("rmse")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=region, x="group", y="rmse", ax=ax, color="#4472c4")
    ax.set_title("Sai số theo vùng trên temporal test")
    ax.set_xlabel("")
    ax.set_ylabel("RMSE (gCO2e/kWh)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "heterogeneity_region.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return heterogeneity, yearly, errors


def _vietnam_case_and_scenarios(
    panel: pd.DataFrame, temporal_test: pd.DataFrame, model: Pipeline, prediction: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    vietnam = temporal_test.loc[temporal_test["iso_code"] == "VNM"].copy()
    if vietnam.empty:
        raise ValueError("Vietnam is absent from the temporal test set")
    pred_map = pd.Series(prediction, index=temporal_test.index)
    vietnam["prediction"] = pred_map.loc[vietnam.index]
    vietnam["error"] = vietnam["prediction"] - vietnam[TARGET]
    vietnam.to_csv(TABLE_DIR / "vietnam_temporal_case.csv", index=False)

    historical = panel.loc[panel["iso_code"] == "VNM"].sort_values("year")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(historical["year"], historical[TARGET], marker="o", ms=3, label="OWID quan sát")
    ax.plot(vietnam["year"], vietnam["prediction"], marker="s", linewidth=2, label="Dự báo ngoài mẫu")
    ax.axvline(2021.5, color="black", linestyle="--", linewidth=1, label="Bắt đầu test")
    ax.set_title("Nghiên cứu tình huống Việt Nam")
    ax.set_ylabel("gCO2e/kWh")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "vietnam_case.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    base = historical.iloc[-1].copy()
    scenarios = [
        ("Quan sát cơ sở", {}),
        ("Solar +5, coal -5", {"solar_share_elec": 5.0, "coal_share_elec": -5.0}),
        ("Wind +5, coal -5", {"wind_share_elec": 5.0, "coal_share_elec": -5.0}),
        ("Solar +5, wind +5, coal -10", {"solar_share_elec": 5.0, "wind_share_elec": 5.0, "coal_share_elec": -10.0}),
        ("Nuclear +5, coal -5", {"nuclear_share_elec": 5.0, "coal_share_elec": -5.0}),
        ("Hydro -5, gas +5", {"hydro_share_elec": -5.0, "gas_share_elec": 5.0}),
    ]
    scenario_rows: list[pd.Series] = []
    for name, changes in scenarios:
        row = base.copy()
        for feature, delta in changes.items():
            row[feature] = max(0.0, float(row[feature]) + delta)
        row["scenario"] = name
        scenario_rows.append(row)
    scenario_df = pd.DataFrame(scenario_rows)
    scenario_df["predicted_carbon_intensity"] = model.predict(scenario_df[MODEL_FEATURES])
    baseline = float(scenario_df.iloc[0]["predicted_carbon_intensity"])
    scenario_df["change_vs_model_baseline"] = scenario_df["predicted_carbon_intensity"] - baseline
    scenario_df["change_percent"] = 100 * scenario_df["change_vs_model_baseline"] / baseline
    keep = ["scenario", "country", "iso_code", "year", "predicted_carbon_intensity", "change_vs_model_baseline", "change_percent", *SOURCE_SHARE_FEATURES]
    scenario_df[keep].to_csv(TABLE_DIR / "vietnam_scenarios.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 5))
    plot = scenario_df.iloc[1:].sort_values("change_vs_model_baseline")
    colors = np.where(plot["change_vs_model_baseline"] <= 0, "#2e8b57", "#c44e52")
    ax.barh(plot["scenario"], plot["change_vs_model_baseline"], color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Thay đổi dự báo so với cơ sở (gCO2e/kWh)")
    ax.set_title(f"Kịch bản phản thực mô hình cho Việt Nam, năm cơ sở {int(base['year'])}")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "vietnam_scenarios.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return vietnam, scenario_df[keep]


def _latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#"}
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _write_latex_outputs(
    panel: pd.DataFrame,
    audit: dict[str, Any],
    metrics: pd.DataFrame,
    region_metrics: pd.DataFrame,
    best_key: str,
    shap_df: pd.DataFrame,
    heterogeneity: pd.DataFrame,
    vietnam: pd.DataFrame,
    scenarios: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> None:
    temporal = metrics.loc[(metrics["protocol"] == "temporal") & (metrics["model"] == best_key)].iloc[0]
    unseen = metrics.loc[(metrics["protocol"] == "unseen_country") & (metrics["model"] == best_key)].iloc[0]
    latest_vn = vietnam.sort_values("year").iloc[-1]
    best_scenario = scenarios.iloc[1:].sort_values("change_vs_model_baseline").iloc[0]
    macros = {
        "PanelRows": f"{len(panel):,}",
        "PanelCountries": panel["iso_code"].nunique(),
        "PanelStartYear": int(panel["year"].min()),
        "PanelEndYear": int(panel["year"].max()),
        "LatestSourceYear": int(audit["source_audit"]["target_year_range"][1]),
        "BestModel": MODEL_LABELS[best_key],
        "TemporalMAE": f"{temporal['mae']:.2f}",
        "TemporalRMSE": f"{temporal['rmse']:.2f}",
        "TemporalRsquared": f"{temporal['r2']:.3f}",
        "UnseenMAE": f"{unseen['mae']:.2f}",
        "UnseenRMSE": f"{unseen['rmse']:.2f}",
        "UnseenRsquared": f"{unseen['r2']:.3f}",
        "TopShapFeature": _latex_escape(shap_df.iloc[0]["feature"]),
        "VietnamLatestYear": int(latest_vn["year"]),
        "VietnamActual": f"{latest_vn[TARGET]:.2f}",
        "VietnamPredicted": f"{latest_vn['prediction']:.2f}",
        "BestScenarioName": _latex_escape(best_scenario["scenario"]),
        "BestScenarioDelta": f"{best_scenario['change_vs_model_baseline']:.2f}",
    }
    macro_lines = [f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macros.items()]
    (TABLE_DIR / "results_macros.tex").write_text("\n".join(macro_lines) + "\n", encoding="utf-8")

    display_metrics = metrics[["protocol", "model_label", "n", "mae", "rmse", "r2"]].copy()
    display_metrics[["mae", "rmse"]] = display_metrics[["mae", "rmse"]].round(2)
    display_metrics["r2"] = display_metrics["r2"].round(3)
    display_metrics.to_latex(TABLE_DIR / "model_metrics.tex", index=False, escape=True, longtable=True)
    region_metrics[["held_out_region", "model_label", "test_n", "mae", "rmse", "r2"]].round(3).to_latex(
        TABLE_DIR / "region_transfer_metrics.tex", index=False, escape=True, longtable=True
    )
    heterogeneity[["dimension", "group", "n", "mae", "rmse", "r2"]].round(3).to_latex(
        TABLE_DIR / "heterogeneity_metrics.tex", index=False, escape=True, longtable=True
    )
    scenario_table = scenarios[["scenario", "predicted_carbon_intensity", "change_vs_model_baseline", "change_percent"]].copy()
    scenario_table.columns = ["Kịch bản", "Dự báo", "Thay đổi", "Thay đổi pct"]
    scenario_table.round(2).to_latex(
        TABLE_DIR / "vietnam_scenarios.tex", index=False, escape=True, column_format=r"p{6.2cm}rrr"
    )
    bootstrap.round(3).to_latex(TABLE_DIR / "bootstrap_ci.tex", index=False, escape=True)


def _write_provenance() -> None:
    hashes: dict[str, str] = {}
    for path in (ENERGY_PATH, *SOURCE_PATHS.values()):
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "shap": shap.__version__,
        },
        "random_seed": SEED,
        "source_urls": SOURCE_URLS,
        "sha256": hashes,
        "model_features": MODEL_FEATURES,
    }
    _json_write(OUTPUT_DIR / "run_manifest.json", payload)


def run_thesis(force_download: bool = True) -> dict[str, Any]:
    ensure_directories()
    download_sources(force=force_download)
    source_audit = audit_sources()
    panel, audit = prepare_panel()
    audit["source_audit"] = source_audit

    temporal = _fit_protocol(panel, "temporal_split", "temporal")
    unseen = _fit_protocol(panel, "unseen_country_split", "unseen_country")
    search = pd.concat([temporal["search"], unseen["search"]], ignore_index=True)
    metrics = pd.concat([temporal["metrics"], unseen["metrics"]], ignore_index=True)
    search.to_csv(TABLE_DIR / "hyperparameter_search.csv", index=False)
    metrics.to_csv(TABLE_DIR / "model_metrics.csv", index=False)
    temporal["predictions"].to_csv(TABLE_DIR / "temporal_test_predictions.csv", index=False)
    unseen["predictions"].to_csv(TABLE_DIR / "unseen_country_test_predictions.csv", index=False)

    region_metrics = _run_region_transfer(panel, temporal["templates"])
    region_metrics.to_csv(TABLE_DIR / "region_transfer_metrics.csv", index=False)
    _plot_model_results(metrics, region_metrics)

    validation_summary = search.groupby("model", as_index=False)["validation_rmse"].min().sort_values("validation_rmse")
    best_key = str(validation_summary.iloc[0]["model"])
    best_model = temporal["models"][best_key]
    test = temporal["splits"]["test"]
    prediction = best_model.predict(test[MODEL_FEATURES])
    bootstrap = _cluster_bootstrap_metrics(test, prediction)
    bootstrap.to_csv(TABLE_DIR / "bootstrap_confidence_intervals.csv", index=False)
    explanations = _explain_model(best_model, test, best_key)
    heterogeneity, yearly, errors = _heterogeneity_and_errors(test, prediction)
    vietnam, scenarios = _vietnam_case_and_scenarios(panel, test, best_model, prediction)
    _write_latex_outputs(
        panel, audit, metrics, region_metrics, best_key, explanations["shap"], heterogeneity, vietnam, scenarios, bootstrap
    )
    _write_provenance()
    summary = {
        "panel_rows": len(panel),
        "countries": int(panel["iso_code"].nunique()),
        "years": [int(panel["year"].min()), int(panel["year"].max())],
        "best_model_selected_by_temporal_validation_rmse": best_key,
        "metrics_path": str(TABLE_DIR / "model_metrics.csv"),
        "figures": len(list(FIGURE_DIR.glob("*.png"))),
    }
    _json_write(OUTPUT_DIR / "run_summary.json", summary)
    return summary


if __name__ == "__main__":
    print(json.dumps(run_thesis(), ensure_ascii=False, indent=2))
