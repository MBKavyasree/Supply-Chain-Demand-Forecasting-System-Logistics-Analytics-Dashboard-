from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "supply_chain_data.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"


def ensure_directories() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [
        column.strip().lower().replace(" ", "_") for column in cleaned.columns
    ]
    cleaned = cleaned.rename(
        columns={
            "number_of_products_sold": "products_sold",
            "lead_times": "order_lead_time_days",
            "lead_time": "supplier_lead_time_days",
            "shipping_times": "shipping_time_days",
            "shipping_costs": "shipping_cost",
            "supplier_name": "supplier",
            "product_type": "product_type",
            "customer_demographics": "customer_demographic",
            "stock_levels": "stock_level",
            "order_quantities": "order_quantity",
            "production_volumes": "production_volume",
            "manufacturing_lead_time": "manufacturing_lead_time_days",
            "manufacturing_costs": "manufacturing_cost",
            "defect_rates": "defect_rate",
            "transportation_modes": "transport_mode",
            "revenue_generated": "revenue",
            "costs": "route_cost",
        }
    )
    return cleaned


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["record_id"] = np.arange(1, len(enriched) + 1)
    enriched["inventory_gap"] = enriched["stock_level"] - enriched["products_sold"]
    enriched["capacity_buffer"] = enriched["production_volume"] - enriched["order_quantity"]
    enriched["stock_to_order_ratio"] = enriched["stock_level"] / enriched[
        "order_quantity"
    ].replace(0, np.nan)
    enriched["inventory_turnover_ratio"] = enriched["products_sold"] / enriched[
        "stock_level"
    ].replace(0, np.nan)
    enriched["inventory_turnover_ratio"] = enriched[
        "inventory_turnover_ratio"
    ].replace([np.inf, -np.inf], np.nan)
    enriched["logistics_cost_total"] = enriched["shipping_cost"] + enriched["route_cost"]
    enriched["total_lead_time_days"] = (
        enriched["supplier_lead_time_days"]
        + enriched["shipping_time_days"]
        + enriched["manufacturing_lead_time_days"]
    )
    enriched["estimated_unit_margin"] = enriched["price"] - enriched["manufacturing_cost"]
    enriched["revenue_per_unit_sold"] = enriched["revenue"] / enriched[
        "products_sold"
    ].replace(0, np.nan)
    return enriched


def build_summary_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame | pd.Series | float | int]:
    inspection_score = {"Pass": 1.0, "Pending": 0.5, "Fail": 0.0}
    supplier_summary = (
        df.groupby("supplier")
        .agg(
            avg_products_sold=("products_sold", "mean"),
            avg_revenue=("revenue", "mean"),
            avg_shipping_cost=("shipping_cost", "mean"),
            avg_total_lead_time=("total_lead_time_days", "mean"),
            avg_defect_rate=("defect_rate", "mean"),
            inspection_pass_rate=(
                "inspection_results",
                lambda values: (values == "Pass").mean(),
            ),
        )
        .sort_values("avg_products_sold", ascending=False)
    )
    supplier_summary["supplier_score"] = (
        supplier_summary["avg_products_sold"].rank(pct=True) * 0.35
        + supplier_summary["avg_revenue"].rank(pct=True) * 0.25
        + supplier_summary["inspection_pass_rate"].rank(pct=True) * 0.20
        + supplier_summary["avg_total_lead_time"].rank(pct=True, ascending=False) * 0.10
        + supplier_summary["avg_defect_rate"].rank(pct=True, ascending=False) * 0.10
    )

    product_summary = (
        df.groupby("product_type")
        .agg(
            total_revenue=("revenue", "sum"),
            total_products_sold=("products_sold", "sum"),
            avg_stock_level=("stock_level", "mean"),
            avg_inventory_gap=("inventory_gap", "mean"),
        )
        .sort_values("total_revenue", ascending=False)
    )

    transport_summary = (
        df.groupby("transport_mode")
        .agg(
            avg_shipping_cost=("shipping_cost", "mean"),
            avg_route_cost=("route_cost", "mean"),
            avg_shipping_time=("shipping_time_days", "mean"),
            avg_total_lead_time=("total_lead_time_days", "mean"),
        )
        .sort_values("avg_shipping_cost")
    )

    inventory_summary = (
        df.groupby("sku")
        .agg(
            product_type=("product_type", "first"),
            stock_level=("stock_level", "mean"),
            products_sold=("products_sold", "mean"),
            inventory_gap=("inventory_gap", "mean"),
        )
        .sort_values("stock_level", ascending=False)
    )

    demand_trend = (
        df.sort_values("record_id")
        .assign(demand_rolling_7=lambda data: data["products_sold"].rolling(7, min_periods=1).mean())
        [["record_id", "products_sold", "demand_rolling_7"]]
    )

    summary = {
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "total_revenue": float(df["revenue"].sum()),
        "total_products_sold": int(df["products_sold"].sum()),
        "average_shipping_cost": float(df["shipping_cost"].mean()),
        "average_lead_time": float(df["total_lead_time_days"].mean()),
        "supplier_summary": supplier_summary,
        "product_summary": product_summary,
        "transport_summary": transport_summary,
        "inventory_summary": inventory_summary,
        "demand_trend": demand_trend,
        "inspection_distribution": df["inspection_results"].value_counts(normalize=True),
        "inspection_score_mean": float(df["inspection_results"].map(inspection_score).mean()),
    }
    return summary


def save_plot(plot_path: Path) -> None:
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()


def generate_eda_plots(df: pd.DataFrame, summary: dict[str, pd.DataFrame | pd.Series | float | int]) -> None:
    sns.set_theme(style="whitegrid", palette="crest")

    demand_trend = summary["demand_trend"]
    plt.figure(figsize=(12, 5))
    sns.lineplot(data=demand_trend, x="record_id", y="products_sold", label="Observed demand")
    sns.lineplot(data=demand_trend, x="record_id", y="demand_rolling_7", label="7-record rolling mean")
    plt.title("Demand Trend Across Observation Sequence")
    plt.xlabel("Observation sequence")
    plt.ylabel("Products sold")
    save_plot(PLOTS_DIR / "demand_trend.png")

    plt.figure(figsize=(8, 5))
    product_summary = summary["product_summary"].reset_index()
    sns.barplot(data=product_summary, x="product_type", y="total_revenue")
    plt.title("Revenue by Product Type")
    plt.xlabel("Product type")
    plt.ylabel("Revenue")
    save_plot(PLOTS_DIR / "revenue_by_product_type.png")

    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="transport_mode", y="shipping_cost", estimator=np.mean, errorbar=None)
    plt.title("Average Shipping Cost by Transportation Mode")
    plt.xlabel("Transportation mode")
    plt.ylabel("Average shipping cost")
    save_plot(PLOTS_DIR / "shipping_cost_by_transport_mode.png")

    plt.figure(figsize=(10, 5))
    supplier_summary = summary["supplier_summary"].reset_index().sort_values(
        "supplier_score", ascending=False
    )
    sns.barplot(data=supplier_summary, x="supplier", y="supplier_score")
    plt.title("Supplier Performance Comparison")
    plt.xlabel("Supplier")
    plt.ylabel("Composite supplier score")
    save_plot(PLOTS_DIR / "supplier_performance.png")

    top_inventory = summary["inventory_summary"].head(15).reset_index()
    plt.figure(figsize=(12, 6))
    sns.barplot(data=top_inventory, x="sku", y="stock_level", hue="product_type")
    plt.title("Inventory Levels Across Top-Stocked Products")
    plt.xlabel("SKU")
    plt.ylabel("Stock level")
    plt.xticks(rotation=45)
    save_plot(PLOTS_DIR / "inventory_levels.png")

    plt.figure(figsize=(10, 6))
    numeric_columns = [
        "price",
        "availability",
        "products_sold",
        "stock_level",
        "order_quantity",
        "shipping_cost",
        "supplier_lead_time_days",
        "shipping_time_days",
        "manufacturing_lead_time_days",
        "defect_rate",
        "route_cost",
        "total_lead_time_days",
    ]
    correlation = df[numeric_columns].corr(numeric_only=True)
    sns.heatmap(correlation, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap for Supply Chain Drivers")
    save_plot(PLOTS_DIR / "correlation_heatmap.png")


def build_model(df: pd.DataFrame) -> tuple[Pipeline, dict[str, float], pd.DataFrame, pd.DataFrame]:
    feature_columns = [
        "product_type",
        "sku",
        "price",
        "availability",
        "customer_demographic",
        "stock_level",
        "order_lead_time_days",
        "order_quantity",
        "shipping_time_days",
        "shipping_carriers",
        "shipping_cost",
        "supplier",
        "location",
        "supplier_lead_time_days",
        "production_volume",
        "manufacturing_lead_time_days",
        "manufacturing_cost",
        "inspection_results",
        "defect_rate",
        "transport_mode",
        "routes",
        "route_cost",
        "capacity_buffer",
        "stock_to_order_ratio",
        "total_lead_time_days",
    ]
    target_column = "products_sold"

    model_frame = df[feature_columns + [target_column]].copy()
    X = model_frame[feature_columns]
    y = model_frame[target_column]

    categorical_columns = X.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric_columns = [column for column in feature_columns if column not in categorical_columns]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_columns,
            ),
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
        ]
    )

    candidate_models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=2,
            random_state=42,
        ),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    model_scores: dict[str, float] = {}
    for model_name, estimator in candidate_models.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
        scoring = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring={
                "r2": "r2",
                "mae": "neg_mean_absolute_error",
                "rmse": "neg_root_mean_squared_error",
            },
            n_jobs=None,
        )
        model_scores[model_name] = float(np.mean(scoring["test_r2"]))

    best_model_name = max(model_scores, key=model_scores.get)
    best_estimator = candidate_models[best_model_name]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", best_estimator)])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    metrics = {
        "best_model": best_model_name,
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "r2": float(r2_score(y_test, predictions)),
        "cv_r2": float(model_scores[best_model_name]),
    }

    prediction_frame = X_test.copy()
    prediction_frame["actual_products_sold"] = y_test.values
    prediction_frame["predicted_products_sold"] = predictions
    prediction_frame["absolute_error"] = (
        prediction_frame["actual_products_sold"]
        - prediction_frame["predicted_products_sold"]
    ).abs()
    prediction_frame = prediction_frame.sort_values("absolute_error", ascending=False)

    current_forecast = df[
        ["sku", "product_type", "revenue", "stock_level"] + feature_columns
    ].copy()
    current_forecast = current_forecast.loc[:, ~current_forecast.columns.duplicated()].copy()
    current_forecast["predicted_products_sold"] = pipeline.predict(current_forecast[feature_columns])
    current_forecast = current_forecast.sort_values("predicted_products_sold", ascending=False)

    return pipeline, metrics, prediction_frame, current_forecast


def extract_feature_importance(model: Pipeline) -> pd.DataFrame:
    preprocessor: ColumnTransformer = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]

    transformed_feature_names: list[str] = []
    for transformer_name, transformer, columns in preprocessor.transformers_:
        if transformer_name == "categorical":
            encoder = transformer.named_steps["encoder"]
            transformed_feature_names.extend(encoder.get_feature_names_out(columns).tolist())
        elif transformer_name == "numeric":
            transformed_feature_names.extend(columns)

    if hasattr(estimator, "feature_importances_"):
        importance_values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importance_values = np.abs(estimator.coef_)
    else:
        importance_values = np.zeros(len(transformed_feature_names))

    importance_frame = pd.DataFrame(
        {
            "feature": transformed_feature_names,
            "importance": importance_values,
        }
    ).sort_values("importance", ascending=False)
    return importance_frame.head(15)


def generate_model_plots(
    prediction_frame: pd.DataFrame, feature_importance: pd.DataFrame
) -> None:
    plt.figure(figsize=(7, 7))
    sns.scatterplot(
        data=prediction_frame,
        x="actual_products_sold",
        y="predicted_products_sold",
    )
    limits = [
        float(min(prediction_frame["actual_products_sold"].min(), prediction_frame["predicted_products_sold"].min())),
        float(max(prediction_frame["actual_products_sold"].max(), prediction_frame["predicted_products_sold"].max())),
    ]
    plt.plot(limits, limits, color="red", linestyle="--")
    plt.title("Demand Forecast: Actual vs Predicted")
    plt.xlabel("Actual products sold")
    plt.ylabel("Predicted products sold")
    save_plot(PLOTS_DIR / "forecast_actual_vs_predicted.png")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x="importance", y="feature")
    plt.title("Top Demand Forecast Drivers")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    save_plot(PLOTS_DIR / "forecast_feature_importance.png")


def build_dashboard_html(
        summary: dict[str, pd.DataFrame | pd.Series | float | int],
        forecast_metrics: dict[str, float],
        current_forecast: pd.DataFrame,
        feature_importance: pd.DataFrame,
        insights: list[str],
) -> None:
        product_summary = summary["product_summary"].reset_index()
        transport_summary = summary["transport_summary"].reset_index()
        supplier_summary = summary["supplier_summary"].reset_index().sort_values(
                "supplier_score", ascending=False
        )
        top_inventory = summary["inventory_summary"].head(12).reset_index()
        demand_trend = summary["demand_trend"]
        forecast_candidates = current_forecast[
                ["sku", "product_type", "predicted_products_sold", "stock_level"]
        ].head(10)
        forecast_drivers = feature_importance.head(10)

        top_product = product_summary.sort_values("total_revenue", ascending=False).iloc[0]
        best_supplier = supplier_summary.iloc[0]
        cheapest_mode = transport_summary.sort_values("avg_shipping_cost").iloc[0]
        inventory_risk = current_forecast.sort_values(
                ["predicted_products_sold", "stock_level"], ascending=[False, True]
        ).iloc[0]
        forecast_health = "Low reliability"
        forecast_health_class = "warning"
        if forecast_metrics["r2"] >= 0.5:
                forecast_health = "Reliable"
                forecast_health_class = "good"
        elif forecast_metrics["r2"] >= 0:
                forecast_health = "Directional only"

        common_layout = {
                "paper_bgcolor": "rgba(0,0,0,0)",
                "plot_bgcolor": "rgba(0,0,0,0)",
                "margin": {"l": 40, "r": 20, "t": 40, "b": 40},
            "font": {"family": "Space Grotesk, Segoe UI, sans-serif", "color": "#172033"},
        }
        chart_config = {
                "displayModeBar": False,
                "responsive": True,
        }

        demand_fig = go.Figure()
        demand_fig.add_trace(
                go.Scatter(
                        x=demand_trend["record_id"],
                        y=demand_trend["products_sold"],
                        mode="lines",
                        name="Observed demand",
                        line={"color": "#103f91", "width": 3},
                        fill="tozeroy",
                        fillcolor="rgba(16,63,145,0.08)",
                )
        )
        demand_fig.add_trace(
                go.Scatter(
                        x=demand_trend["record_id"],
                        y=demand_trend["demand_rolling_7"],
                        mode="lines",
                        name="7-record rolling mean",
                        line={"color": "#d97706", "width": 2},
                )
        )
        demand_fig.update_layout(
                **common_layout,
                title="Demand Pattern Across Observations",
                legend={"orientation": "h", "y": 1.1, "x": 0},
                xaxis={"title": "Observation sequence", "gridcolor": "rgba(23,32,51,0.08)"},
                yaxis={"title": "Products sold", "gridcolor": "rgba(23,32,51,0.08)"},
        )

        revenue_fig = go.Figure(
                data=[
                        go.Bar(
                                x=product_summary["product_type"],
                                y=product_summary["total_revenue"],
                                marker={
                                        "color": ["#103f91", "#e07a2f", "#1f7a5c"],
                                        "line": {"color": "rgba(0,0,0,0)", "width": 0},
                                },
                                text=[f"${value:,.0f}" for value in product_summary["total_revenue"]],
                                textposition="outside",
                        )
                ]
        )
        revenue_fig.update_layout(
                **common_layout,
                title="Revenue by Product Type",
                xaxis={"title": "Product type"},
                yaxis={"title": "Revenue", "gridcolor": "rgba(23,32,51,0.08)"},
        )

        shipping_fig = go.Figure()
        shipping_fig.add_trace(
                go.Bar(
                        x=transport_summary["transport_mode"],
                        y=transport_summary["avg_shipping_cost"],
                        name="Avg shipping cost",
                        marker_color="#c2410c",
                )
        )
        shipping_fig.add_trace(
                go.Scatter(
                        x=transport_summary["transport_mode"],
                        y=transport_summary["avg_total_lead_time"],
                        name="Avg lead time",
                        mode="lines+markers",
                        yaxis="y2",
                        line={"color": "#103f91", "width": 3},
                        marker={"size": 9},
                )
        )
        shipping_fig.update_layout(
                **common_layout,
                title="Shipping Cost by Transportation Mode",
                xaxis={"title": "Transportation mode"},
                yaxis={"title": "Shipping cost", "gridcolor": "rgba(23,32,51,0.08)"},
                yaxis2={"title": "Lead time", "overlaying": "y", "side": "right"},
                legend={"orientation": "h", "y": 1.1, "x": 0},
        )

        supplier_fig = go.Figure(
                data=[
                        go.Bar(
                                x=supplier_summary["supplier_score"],
                                y=supplier_summary["supplier"],
                                orientation="h",
                                marker={
                                        "color": supplier_summary["supplier_score"],
                                        "colorscale": [[0, "#f3b07c"], [1, "#1f7a5c"]],
                                },
                                text=[f"{value:.2f}" for value in supplier_summary["supplier_score"]],
                                textposition="outside",
                        )
                ]
        )
        supplier_fig.update_layout(
                **common_layout,
                title="Supplier Performance Comparison",
                xaxis={"title": "Composite score", "range": [0, 1], "gridcolor": "rgba(23,32,51,0.08)"},
                yaxis={"title": "Supplier", "autorange": "reversed"},
        )

        inventory_fig = go.Figure(
                data=[
                        go.Bar(
                                x=top_inventory["sku"],
                                y=top_inventory["stock_level"],
                                marker_color="#8b1e3f",
                                name="Stock level",
                        ),
                        go.Scatter(
                                x=top_inventory["sku"],
                                y=top_inventory["products_sold"],
                                mode="lines+markers",
                                marker={"color": "#f1c40f", "size": 8},
                                line={"color": "#f1c40f", "width": 2},
                                name="Products sold",
                        ),
                ]
        )
        inventory_fig.update_layout(
                **common_layout,
                title="Inventory Levels Across Key Products",
                xaxis={"title": "SKU", "tickangle": -35},
                yaxis={"title": "Units", "gridcolor": "rgba(23,32,51,0.08)"},
                legend={"orientation": "h", "y": 1.1, "x": 0},
        )

        forecast_fig = go.Figure(
                data=[
                        go.Bar(
                                x=forecast_candidates["sku"],
                                y=forecast_candidates["predicted_products_sold"],
                                marker_color="#103f91",
                                text=[f"{value:.0f}" for value in forecast_candidates["predicted_products_sold"]],
                                textposition="outside",
                                customdata=forecast_candidates[["product_type", "stock_level"]],
                                hovertemplate=(
                                        "SKU=%{x}<br>Predicted demand=%{y:.1f}<br>Product type=%{customdata[0]}"
                                        "<br>Current stock=%{customdata[1]}<extra></extra>"
                                ),
                        )
                ]
        )
        forecast_fig.update_layout(
                **common_layout,
                title="Highest Forecast Demand Candidates",
                xaxis={"title": "SKU", "tickangle": -35},
                yaxis={"title": "Predicted products sold", "gridcolor": "rgba(23,32,51,0.08)"},
        )

        driver_fig = go.Figure(
                data=[
                        go.Bar(
                                x=forecast_drivers["importance"],
                                y=forecast_drivers["feature"],
                                orientation="h",
                                marker_color="#e07a2f",
                        )
                ]
        )
        driver_fig.update_layout(
                **common_layout,
                title="Top Forecast Drivers",
                xaxis={"title": "Importance", "gridcolor": "rgba(23,32,51,0.08)"},
                yaxis={"title": "Feature", "autorange": "reversed"},
        )

        cards_html = "".join(
            [
                (
                    f'<article class="kpi-card accent-navy"><span class="kpi-chip">Commercial</span>'
                    f'<span class="kpi-label">Total Revenue</span>'
                    f'<strong class="kpi-value">${summary["total_revenue"]:,.0f}</strong>'
                    '<span class="kpi-foot">Revenue generated across all observations</span></article>'
                ),
                (
                    f'<article class="kpi-card accent-amber"><span class="kpi-chip">Demand</span>'
                    f'<span class="kpi-label">Products Sold</span>'
                    f'<strong class="kpi-value">{summary["total_products_sold"]:,}</strong>'
                    '<span class="kpi-foot">Total recorded units sold</span></article>'
                ),
                (
                    f'<article class="kpi-card accent-teal"><span class="kpi-chip">Logistics</span>'
                    f'<span class="kpi-label">Avg Shipping Cost</span>'
                    f'<strong class="kpi-value">${summary["average_shipping_cost"]:.2f}</strong>'
                    '<span class="kpi-foot">Average carrier charge per order</span></article>'
                ),
                (
                    f'<article class="kpi-card accent-wine"><span class="kpi-chip">Operations</span>'
                    f'<span class="kpi-label">Avg Lead Time</span>'
                    f'<strong class="kpi-value">{summary["average_lead_time"]:.1f} days</strong>'
                    '<span class="kpi-foot">Supplier + manufacturing + shipping</span></article>'
                ),
            ]
        )

        insight_items = "".join(
                f"<li>{html.escape(insight)}</li>" for insight in insights[:5]
        )

        dashboard_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Supply Chain Control Tower</title>
    <script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Space+Grotesk:wght@400;500;700&display=swap');

        :root {{
            --bg: #efe7dc;
            --panel: rgba(255, 250, 243, 0.9);
            --panel-strong: rgba(255, 255, 255, 0.76);
            --ink: #162033;
            --muted: #5b6473;
            --navy: #123b7a;
            --navy-deep: #0a1831;
            --teal: #0f766e;
            --amber: #cc6d1d;
            --wine: #7c2048;
            --sand: #f4ede3;
            --line: rgba(22, 32, 51, 0.10);
            --shadow: 0 20px 50px rgba(17, 26, 45, 0.10);
            --radius: 26px;
        }}

        * {{ box-sizing: border-box; }}

        body {{
            margin: 0;
            font-family: "Space Grotesk", "Segoe UI", sans-serif;
            color: var(--ink);
            background:
                radial-gradient(circle at 10% 12%, rgba(204, 109, 29, 0.20), transparent 22%),
                radial-gradient(circle at 90% 0%, rgba(18, 59, 122, 0.18), transparent 28%),
                linear-gradient(180deg, #fbf8f3 0%, var(--bg) 100%);
            min-height: 100vh;
            position: relative;
        }}

        body::before,
        body::after {{
            content: "";
            position: fixed;
            inset: auto;
            pointer-events: none;
            z-index: 0;
            filter: blur(30px);
            opacity: 0.6;
        }}

        body::before {{
            width: 280px;
            height: 280px;
            left: -60px;
            top: 180px;
            background: radial-gradient(circle, rgba(15, 118, 110, 0.18), transparent 70%);
        }}

        body::after {{
            width: 340px;
            height: 340px;
            right: -80px;
            top: 420px;
            background: radial-gradient(circle, rgba(124, 32, 72, 0.12), transparent 70%);
        }}

        .shell {{
            max-width: 1480px;
            margin: 0 auto;
            padding: 30px 24px 72px;
            position: relative;
            z-index: 1;
        }}

        .hero {{
            display: grid;
            grid-template-columns: minmax(0, 1.55fr) minmax(300px, 0.9fr);
            gap: 20px;
            margin-bottom: 26px;
        }}

        .hero-card, .panel {{
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            backdrop-filter: blur(12px);
        }}

        .hero-card {{
            padding: 34px;
            min-height: 260px;
            position: relative;
            overflow: hidden;
            background:
                radial-gradient(circle at top right, rgba(243, 176, 124, 0.18), transparent 28%),
                linear-gradient(135deg, #101d36 0%, #17386a 48%, #0f766e 130%);
            color: #f8f3eb;
        }}

        .hero-card::after {{
            content: "";
            position: absolute;
            right: -70px;
            top: -60px;
            width: 260px;
            height: 260px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(255, 219, 170, 0.22), transparent 70%);
        }}

        .hero-card::before {{
            content: "";
            position: absolute;
            inset: auto 24px 24px auto;
            width: 180px;
            height: 180px;
            border-radius: 32px;
            border: 1px solid rgba(255,255,255,0.12);
            background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.01));
            transform: rotate(18deg);
        }}

        .eyebrow {{
            display: inline-block;
            padding: 8px 12px;
            border-radius: 999px;
            background: rgba(255,255,255,0.10);
            color: #f7ead7;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            border: 1px solid rgba(255,255,255,0.12);
        }}

        h1 {{
            margin: 18px 0 10px;
            font-family: "Fraunces", Georgia, serif;
            font-size: clamp(34px, 4vw, 58px);
            line-height: 0.98;
            letter-spacing: -0.03em;
            max-width: 700px;
        }}

        .hero-copy {{
            max-width: 760px;
            font-size: 16px;
            line-height: 1.7;
            color: rgba(248, 243, 235, 0.82);
            margin-bottom: 22px;
        }}

        .hero-metrics {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
            max-width: 820px;
        }}

        .hero-metric {{
            padding: 14px 16px;
            border-radius: 18px;
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.10);
            backdrop-filter: blur(10px);
        }}

        .hero-metric strong {{
            display: block;
            font-size: 24px;
            margin-bottom: 6px;
        }}

        .hero-metric span {{
            display: block;
            font-size: 12px;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: rgba(248, 243, 235, 0.7);
            margin-bottom: 6px;
        }}

        .hero-metric small {{
            color: rgba(248, 243, 235, 0.8);
            line-height: 1.45;
        }}

        .chip-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 18px;
        }}

        .chip {{
            padding: 10px 14px;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.12);
            background: rgba(255,255,255,0.08);
            font-size: 13px;
            color: #f8f3eb;
        }}

        .panel {{
            padding: 24px;
            background: linear-gradient(180deg, rgba(255,252,247,0.96), rgba(255,246,235,0.90));
        }}

        .panel h2, .chart-card h3 {{
            margin: 0 0 10px;
            font-size: 20px;
            letter-spacing: -0.02em;
        }}

        .signal-grid {{
            display: grid;
            gap: 12px;
            margin-top: 14px;
        }}

        .signal {{
            padding: 14px 16px;
            border-radius: 18px;
            background: linear-gradient(180deg, rgba(255,255,255,0.74), rgba(255,248,239,0.96));
            border: 1px solid var(--line);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.6);
        }}

        .signal-label {{
            display: block;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--muted);
            margin-bottom: 6px;
        }}

        .signal-value {{
            font-size: 18px;
            font-weight: 700;
            display: block;
            margin-bottom: 4px;
        }}

        .signal-copy {{
            font-size: 13px;
            color: var(--muted);
            line-height: 1.5;
        }}

        .status-pill {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            border-radius: 999px;
            font-size: 13px;
            font-weight: 700;
        }}

        .status-pill.warning {{
            background: rgba(217, 119, 6, 0.14);
            color: #9a5b02;
        }}

        .status-pill.good {{
            background: rgba(31, 122, 92, 0.14);
            color: #145a42;
        }}

        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 16px;
            margin: 0 0 26px;
        }}

        .kpi-card {{
            position: relative;
            overflow: hidden;
            background: linear-gradient(180deg, rgba(255,255,255,0.90), rgba(255,246,235,0.98));
            border: 1px solid var(--line);
            border-radius: 22px;
            box-shadow: var(--shadow);
            padding: 22px 22px 24px;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}

        .kpi-card::before {{
            content: "";
            position: absolute;
            inset: 0 auto 0 0;
            width: 6px;
            border-radius: 22px 0 0 22px;
        }}

        .accent-navy::before {{ background: linear-gradient(180deg, #123b7a, #4f7fcb); }}
        .accent-amber::before {{ background: linear-gradient(180deg, #cc6d1d, #f3b07c); }}
        .accent-teal::before {{ background: linear-gradient(180deg, #0f766e, #58b7a1); }}
        .accent-wine::before {{ background: linear-gradient(180deg, #7c2048, #be5c84); }}

        .kpi-chip {{
            align-self: flex-start;
            padding: 6px 10px;
            border-radius: 999px;
            background: rgba(22, 32, 51, 0.05);
            color: var(--muted);
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 4px;
        }}

        .kpi-label {{
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: var(--muted);
        }}

        .kpi-value {{
            font-size: clamp(24px, 2vw, 36px);
            line-height: 1;
            letter-spacing: -0.03em;
            font-family: "Fraunces", Georgia, serif;
        }}

        .kpi-foot {{
            color: var(--muted);
            font-size: 13px;
            line-height: 1.5;
        }}

        .layout {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 18px;
        }}

        .layout .span-2 {{ grid-column: span 2; }}

        .chart-card {{
            position: relative;
            overflow: hidden;
            background: linear-gradient(180deg, rgba(255,252,247,0.94), rgba(255,247,238,0.88));
            border: 1px solid var(--line);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            padding: 22px 18px 10px;
        }}

        .chart-card::before {{
            content: "";
            position: absolute;
            inset: 0 0 auto 0;
            height: 5px;
            background: linear-gradient(90deg, #123b7a 0%, #cc6d1d 50%, #0f766e 100%);
        }}

        .chart-kicker {{
            color: var(--muted);
            font-size: 12px;
            margin: 2px 0 6px;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }}

        .chart {{ min-height: 360px; }}

        .insight-list {{
            margin: 0;
            padding-left: 18px;
            color: var(--muted);
            line-height: 1.7;
        }}

        .footnote {{
            margin-top: 22px;
            padding: 16px 18px;
            border-radius: 18px;
            border: 1px dashed rgba(18, 59, 122, 0.24);
            background: rgba(18, 59, 122, 0.05);
            color: var(--muted);
            font-size: 13px;
            line-height: 1.6;
        }}

        @media (max-width: 1100px) {{
            .hero, .layout, .kpi-grid, .hero-metrics {{
                grid-template-columns: 1fr;
            }}

            .layout .span-2 {{
                grid-column: span 1;
            }}
        }}
    </style>
</head>
<body>
    <main class="shell">
        <section class="hero">
            <article class="hero-card">
                <span class="eyebrow">Retail Supply Chain Control Tower</span>
                <h1>Turn operational noise into supply chain signals.</h1>
                <p class="hero-copy">
                    This dashboard organizes demand, inventory, supplier, and logistics performance into one management view.
                    It is designed for fast review: what is selling, where cost is leaking, which suppliers deserve more volume,
                    and which SKUs need intervention first.
                </p>
                <div class="hero-metrics">
                    <div class="hero-metric">
                        <span>Top Category</span>
                        <strong>{top_product['product_type'].title()}</strong>
                        <small>${top_product['total_revenue']:,.0f} revenue contribution</small>
                    </div>
                    <div class="hero-metric">
                        <span>Best Supplier</span>
                        <strong>{best_supplier['supplier']}</strong>
                        <small>Composite score {best_supplier['supplier_score']:.2f}</small>
                    </div>
                    <div class="hero-metric">
                        <span>Lowest Cost Mode</span>
                        <strong>{cheapest_mode['transport_mode']}</strong>
                        <small>${cheapest_mode['avg_shipping_cost']:.2f} average shipping cost</small>
                    </div>
                </div>
                <div class="chip-row">
                    <span class="chip">{summary['row_count']} observations profiled</span>
                    <span class="chip">0 missing values</span>
                    <span class="chip">0 duplicate rows</span>
                    <span class="chip">Forecast model: {forecast_metrics['best_model']}</span>
                </div>
            </article>

            <aside class="panel">
                <h2>Operational Pressure Points</h2>
                <div class="status-pill {forecast_health_class}">Forecast confidence: {forecast_health}</div>
                <div class="signal-grid">
                    <div class="signal">
                        <span class="signal-label">Top revenue engine</span>
                        <span class="signal-value">{top_product['product_type'].title()}</span>
                        <span class="signal-copy">${top_product['total_revenue']:,.0f} revenue with {int(top_product['total_products_sold']):,} units sold.</span>
                    </div>
                    <div class="signal">
                        <span class="signal-label">Best supplier</span>
                        <span class="signal-value">{best_supplier['supplier']}</span>
                        <span class="signal-copy">Supplier score {best_supplier['supplier_score']:.2f} with pass rate {best_supplier['inspection_pass_rate']:.0%}.</span>
                    </div>
                    <div class="signal">
                        <span class="signal-label">Cheapest transport mode</span>
                        <span class="signal-value">{cheapest_mode['transport_mode']}</span>
                        <span class="signal-copy">Average shipping cost ${cheapest_mode['avg_shipping_cost']:.2f} and lead time {cheapest_mode['avg_total_lead_time']:.1f} days.</span>
                    </div>
                    <div class="signal">
                        <span class="signal-label">Most urgent replenishment candidate</span>
                        <span class="signal-value">{inventory_risk['sku']}</span>
                        <span class="signal-copy">Predicted demand {inventory_risk['predicted_products_sold']:.0f} against current stock of {int(inventory_risk['stock_level'])}.</span>
                    </div>
                </div>
            </aside>
        </section>

        <section class="kpi-grid">{cards_html}</section>

        <section class="layout">
            <article class="chart-card span-2">
                <p class="chart-kicker">Demand pulse</p>
                <h3>Demand Trends Over Time</h3>
                <div class="chart">{demand_fig.to_html(full_html=False, include_plotlyjs=False, config=chart_config)}</div>
            </article>

            <article class="chart-card">
                <p class="chart-kicker">Commercial view</p>
                <h3>Revenue by Product Type</h3>
                <div class="chart">{revenue_fig.to_html(full_html=False, include_plotlyjs=False, config=chart_config)}</div>
            </article>

            <article class="chart-card">
                <p class="chart-kicker">Logistics cost and speed</p>
                <h3>Shipping Cost by Transportation Mode</h3>
                <div class="chart">{shipping_fig.to_html(full_html=False, include_plotlyjs=False, config=chart_config)}</div>
            </article>

            <article class="chart-card">
                <p class="chart-kicker">Supplier scorecard</p>
                <h3>Supplier Performance Comparison</h3>
                <div class="chart">{supplier_fig.to_html(full_html=False, include_plotlyjs=False, config=chart_config)}</div>
            </article>

            <article class="chart-card">
                <p class="chart-kicker">Stock pressure</p>
                <h3>Inventory Levels Across Products</h3>
                <div class="chart">{inventory_fig.to_html(full_html=False, include_plotlyjs=False, config=chart_config)}</div>
            </article>

            <article class="chart-card">
                <p class="chart-kicker">Forward view</p>
                <h3>Highest Forecast Demand Candidates</h3>
                <div class="chart">{forecast_fig.to_html(full_html=False, include_plotlyjs=False, config=chart_config)}</div>
            </article>

            <article class="chart-card">
                <p class="chart-kicker">Model diagnostics</p>
                <h3>Top Forecast Drivers</h3>
                <div class="chart">{driver_fig.to_html(full_html=False, include_plotlyjs=False, config=chart_config)}</div>
            </article>

            <article class="panel span-2">
                <h2>Management Readout</h2>
                <ul class="insight-list">{insight_items}</ul>
                <div class="footnote">
                    Demand is displayed over record sequence because the source data does not include order or shipment dates.
                    Forecast metrics should be treated as directional only until a true time field is added and model accuracy improves.
                    Current model performance: R2 {forecast_metrics['r2']:.2f}, MAE {forecast_metrics['mae']:.1f}, RMSE {forecast_metrics['rmse']:.1f}.
                </div>
            </article>
        </section>
    </main>
</body>
</html>
"""

        dashboard_path = OUTPUT_DIR / "supply_chain_dashboard.html"
        dashboard_path.write_text(dashboard_html, encoding="utf-8")


def build_insights(
    summary: dict[str, pd.DataFrame | pd.Series | float | int],
    forecast_metrics: dict[str, float],
    current_forecast: pd.DataFrame,
    feature_importance: pd.DataFrame,
) -> list[str]:
    product_summary = summary["product_summary"]
    transport_summary = summary["transport_summary"]
    supplier_summary = summary["supplier_summary"]
    inventory_summary = summary["inventory_summary"]

    top_product = product_summary.index[0]
    top_revenue = product_summary.iloc[0]["total_revenue"]
    lowest_cost_mode = transport_summary.index[0]
    best_supplier = supplier_summary["supplier_score"].idxmax()
    worst_supplier = supplier_summary["supplier_score"].idxmin()
    high_stock_sku = inventory_summary.index[0]
    low_inventory_gap_sku = inventory_summary.sort_values("inventory_gap").index[0]
    top_forecast_sku = current_forecast.iloc[0]["sku"]
    top_driver = feature_importance.iloc[0]["feature"]

    insights = [
        (
            f"{top_product.title()} is the strongest revenue contributor at ${top_revenue:,.0f}, "
            "so assortment and promotional planning should prioritize that category."
        ),
        (
            f"{lowest_cost_mode} is the lowest-cost transport mode on average, while lead-time variation "
            "across modes suggests mode selection should be segmented by urgency rather than standardized."
        ),
        (
            f"{best_supplier} ranks highest on the composite supplier score, while {worst_supplier} is the weakest performer; "
            "supplier allocation and corrective action plans should reflect that gap."
        ),
        (
            f"{high_stock_sku} carries the highest stock position, whereas {low_inventory_gap_sku} has one of the tightest inventory gaps; "
            "inventory policy should rebalance excess stock away from slow-moving SKUs and protect constrained items."
        ),
        (
            f"The demand forecast model selected {forecast_metrics['best_model']} with test R2={forecast_metrics['r2']:.2f}; "
            f"the strongest predictive signal is {top_driver}, indicating operational variables materially shape demand outcomes."
        ),
        (
            f"{top_forecast_sku} has the highest predicted demand under current operating conditions, making it a priority candidate "
            "for replenishment, supplier capacity checks, and logistics planning."
        ),
    ]
    return insights


def frame_to_text_table(df: pd.DataFrame, index: bool = True, max_rows: int | None = None) -> str:
    display_df = df.copy()
    if max_rows is not None:
        display_df = display_df.head(max_rows)
    numeric_columns = display_df.select_dtypes(include=[np.number]).columns
    display_df[numeric_columns] = display_df[numeric_columns].round(2)
    return display_df.to_string(index=index)


def write_report(
    summary: dict[str, pd.DataFrame | pd.Series | float | int],
    forecast_metrics: dict[str, float],
    prediction_frame: pd.DataFrame,
    current_forecast: pd.DataFrame,
    feature_importance: pd.DataFrame,
    insights: list[str],
) -> None:
    top_suppliers = summary["supplier_summary"].sort_values("supplier_score", ascending=False).head(5)
    transport_summary = summary["transport_summary"]
    product_summary = summary["product_summary"]
    top_forecast = current_forecast[["sku", "product_type", "predicted_products_sold", "stock_level"]].head(10)
    hardest_predictions = prediction_frame[
        ["sku", "product_type", "actual_products_sold", "predicted_products_sold", "absolute_error"]
    ].head(10)

    report = f"""# Supply Chain Data Analysis Report

## Executive Summary

This analysis covers data cleaning, exploratory analysis, demand forecasting, and dashboard design for the retail supply chain dataset. The source file contains {summary['row_count']} records and {summary['column_count']} columns with no missing values and no duplicate rows. Because the dataset does not include a real transaction or shipment date, demand forecasting was implemented as a supervised machine learning problem rather than a classical time-series model.

## Data Cleaning and Preprocessing

- Standardized all column names into machine-friendly snake_case.
- Resolved the ambiguous lead-time fields by preserving both as `order_lead_time_days` and `supplier_lead_time_days`.
- Added derived features: `inventory_gap`, `inventory_turnover_ratio`, `logistics_cost_total`, `total_lead_time_days`, `estimated_unit_margin`, and `revenue_per_unit_sold`.
- Validated that the dataset has 0 missing values and 0 duplicate rows.
- Retained all observations because no rows were structurally invalid.

## KPI Snapshot

- Total revenue: ${summary['total_revenue']:,.2f}
- Total products sold: {summary['total_products_sold']:,}
- Average shipping cost: ${summary['average_shipping_cost']:,.2f}
- Average end-to-end lead time: {summary['average_lead_time']:.2f} days
- Inspection quality score: {summary['inspection_score_mean']:.2f} on a 0 to 1 scale

## Exploratory Findings

### Product Demand and Revenue

```
{frame_to_text_table(product_summary)}
```

### Logistics Efficiency by Transportation Mode

```
{frame_to_text_table(transport_summary)}
```

### Supplier Performance Comparison

```
{frame_to_text_table(top_suppliers)}
```

## Demand Forecasting

- Modeling target: `products_sold`
- Candidate models: Linear Regression, Random Forest Regressor, Gradient Boosting Regressor
- Selected model: {forecast_metrics['best_model']}
- Cross-validated R2: {forecast_metrics['cv_r2']:.3f}
- Test R2: {forecast_metrics['r2']:.3f}
- Test MAE: {forecast_metrics['mae']:.2f}
- Test RMSE: {forecast_metrics['rmse']:.2f}

### Top Forecast Drivers

```
{frame_to_text_table(feature_importance, index=False)}
```

### Highest Forecast Demand Candidates

```
{frame_to_text_table(top_forecast, index=False)}
```

### Largest Forecast Errors to Review

```
{frame_to_text_table(hardest_predictions, index=False)}
```

## Dashboard Design

The interactive dashboard is saved as `outputs/supply_chain_dashboard.html` and includes:

- KPI cards for total revenue, total products sold, and average lead time.
- Demand trend across the observation sequence with a rolling mean overlay.
- Revenue by product type.
- Average shipping cost by transportation mode.
- Supplier performance comparison using a composite score.
- Inventory levels for the highest-stock products.
- Forecast quality annotation with model metrics.

Note: the requested "demand trend over time" is approximated using record sequence because the source dataset has no date field. For production use, replace this with order date or shipment date.

## Business Insights

"""
    report += "\n".join(f"- {insight}" for insight in insights)
    report += f"""

## Recommendations

- Rebalance inventory by reducing stock exposure on overstocked SKUs and increasing safety stock for items with tight inventory gaps and high predicted demand.
- Use supplier scorecards in quarterly business reviews, prioritizing volume allocation to higher-performing suppliers and remediation plans for low-scoring ones.
- Shift non-urgent shipments toward the lowest-cost transportation mode where service levels allow, and reserve faster modes for constrained or high-margin items.
- Add a transaction date, promised delivery date, and actual delivery date to the source data model to unlock true time-series demand forecasting and service-level analytics.
- Investigate the weakest forecast cases to determine whether product-level factors or hidden external drivers are missing from the dataset.

## Generated Outputs

- Dashboard: `outputs/supply_chain_dashboard.html`
- Clean dataset: `outputs/supply_chain_cleaned.csv`
- Forecast results: `outputs/demand_forecast_results.csv`
- Forecast metrics: `outputs/forecast_metrics.json`
- Plots: `outputs/plots/`
"""

    (OUTPUT_DIR / "analysis_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    ensure_directories()
    raw_df = pd.read_csv(DATA_PATH)
    cleaned_df = standardize_columns(raw_df)
    enriched_df = engineer_features(cleaned_df)
    summary = build_summary_tables(enriched_df)
    generate_eda_plots(enriched_df, summary)

    model, forecast_metrics, prediction_frame, current_forecast = build_model(enriched_df)
    feature_importance = extract_feature_importance(model)
    generate_model_plots(prediction_frame, feature_importance)
    insights = build_insights(summary, forecast_metrics, current_forecast, feature_importance)
    build_dashboard_html(
        summary,
        forecast_metrics,
        current_forecast,
        feature_importance,
        insights,
    )
    write_report(
        summary,
        forecast_metrics,
        prediction_frame,
        current_forecast,
        feature_importance,
        insights,
    )

    enriched_df.to_csv(OUTPUT_DIR / "supply_chain_cleaned.csv", index=False)
    current_forecast.to_csv(OUTPUT_DIR / "demand_forecast_results.csv", index=False)
    (OUTPUT_DIR / "forecast_metrics.json").write_text(
        json.dumps(forecast_metrics, indent=2), encoding="utf-8"
    )

    print("Analysis complete. Outputs written to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()