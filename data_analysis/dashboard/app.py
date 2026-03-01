"""
Interactive Dash dashboard for BDD100K dataset statistics.

Displays class distributions, bounding box statistics, scene metadata,
and anomaly summaries loaded from pre-computed CSV files.

Usage:
    python dashboard/app.py --data_dir /path/to/precomputed_csvs

The data directory must contain:
    - class_dist.csv       (class, count)
    - images_per_class.csv (class, image_count)
    - bbox_stats.csv       (class, width, height, area, aspect_ratio, ...)
    - scene_dist.csv       (attribute, value, count)
    - train_val_compare.csv
"""

import argparse
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

BDD_DETECTION_CLASSES = [
    "pedestrian", "rider", "car", "truck", "bus",
    "train", "motorcycle", "bicycle", "traffic light", "traffic sign",
]


def load_data(data_dir: str) -> dict:
    """
    Load all pre-computed CSV files from the given directory.

    Args:
        data_dir: Path to directory containing CSV files.

    Returns:
        Dict mapping short name to pandas DataFrame.
    """
    files = {
        "class_dist": "class_dist.csv",
        "images_per_class": "images_per_class.csv",
        "bbox_stats": "bbox_stats.csv",
        "scene_dist": "scene_dist.csv",
        "train_val": "train_val_compare.csv",
    }
    data = {}
    for key, fname in files.items():
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            data[key] = pd.read_csv(path)
        else:
            data[key] = pd.DataFrame()
    return data


def build_app(data: dict) -> Dash:
    """
    Build and return the Dash application with all layout and callbacks.

    Args:
        data: Dict of DataFrames produced by load_data().

    Returns:
        Configured Dash application instance.
    """
    app = Dash(__name__, title="BDD100K Dataset Dashboard")

    class_options = [{"label": c, "value": c} for c in BDD_DETECTION_CLASSES]

    app.layout = html.Div(
        style={"fontFamily": "Arial, sans-serif", "backgroundColor": "#f4f6f9", "padding": "20px"},
        children=[
            html.H1(
                "BDD100K Dataset Analysis Dashboard",
                style={"textAlign": "center", "color": "#2c3e50", "marginBottom": "5px"},
            ),
            html.P(
                "Exploratory analysis of 100K driving images across 10 object detection classes.",
                style={"textAlign": "center", "color": "#7f8c8d", "marginBottom": "30px"},
            ),

            # ── Section 1: Class Distribution ──────────────────────────────
            html.Div(
                style={"backgroundColor": "white", "borderRadius": "8px",
                       "padding": "20px", "marginBottom": "20px",
                       "boxShadow": "0 1px 4px rgba(0,0,0,0.1)"},
                children=[
                    html.H2("1. Class Distribution", style={"color": "#2c3e50"}),
                    html.Div(
                        style={"display": "flex", "gap": "20px"},
                        children=[
                            dcc.Graph(
                                id="class-dist-bar",
                                figure=_make_class_dist_chart(data.get("class_dist", pd.DataFrame())),
                                style={"flex": "1"},
                            ),
                            dcc.Graph(
                                id="class-dist-pie",
                                figure=_make_class_dist_pie(data.get("class_dist", pd.DataFrame())),
                                style={"flex": "1"},
                            ),
                        ],
                    ),
                ],
            ),

            # ── Section 2: Train vs Val Comparison ─────────────────────────
            html.Div(
                style={"backgroundColor": "white", "borderRadius": "8px",
                       "padding": "20px", "marginBottom": "20px",
                       "boxShadow": "0 1px 4px rgba(0,0,0,0.1)"},
                children=[
                    html.H2("2. Train vs Validation Split", style={"color": "#2c3e50"}),
                    dcc.Graph(
                        id="train-val-chart",
                        figure=_make_train_val_chart(data.get("train_val", pd.DataFrame())),
                    ),
                ],
            ),

            # ── Section 3: Bounding Box Analysis ──────────────────────────
            html.Div(
                style={"backgroundColor": "white", "borderRadius": "8px",
                       "padding": "20px", "marginBottom": "20px",
                       "boxShadow": "0 1px 4px rgba(0,0,0,0.1)"},
                children=[
                    html.H2("3. Bounding Box Analysis", style={"color": "#2c3e50"}),
                    html.Label("Select Class:", style={"fontWeight": "bold"}),
                    dcc.Dropdown(
                        id="bbox-class-selector",
                        options=class_options,
                        value="car",
                        clearable=False,
                        style={"width": "300px", "marginBottom": "15px"},
                    ),
                    html.Div(
                        style={"display": "flex", "gap": "20px"},
                        children=[
                            dcc.Graph(id="bbox-area-hist", style={"flex": "1"}),
                            dcc.Graph(id="bbox-ar-hist", style={"flex": "1"}),
                        ],
                    ),
                ],
            ),

            # ── Section 4: Scene Metadata ──────────────────────────────────
            html.Div(
                style={"backgroundColor": "white", "borderRadius": "8px",
                       "padding": "20px", "marginBottom": "20px",
                       "boxShadow": "0 1px 4px rgba(0,0,0,0.1)"},
                children=[
                    html.H2("4. Scene Metadata Distribution", style={"color": "#2c3e50"}),
                    html.Div(
                        style={"display": "flex", "gap": "20px"},
                        children=[
                            dcc.Graph(
                                id="weather-chart",
                                figure=_make_scene_chart(
                                    data.get("scene_dist", pd.DataFrame()), "weather"
                                ),
                                style={"flex": "1"},
                            ),
                            dcc.Graph(
                                id="timeofday-chart",
                                figure=_make_scene_chart(
                                    data.get("scene_dist", pd.DataFrame()), "time_of_day"
                                ),
                                style={"flex": "1"},
                            ),
                            dcc.Graph(
                                id="scene-chart",
                                figure=_make_scene_chart(
                                    data.get("scene_dist", pd.DataFrame()), "scene"
                                ),
                                style={"flex": "1"},
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )

    @app.callback(
        Output("bbox-area-hist", "figure"),
        Output("bbox-ar-hist", "figure"),
        Input("bbox-class-selector", "value"),
    )
    def update_bbox_charts(selected_class: str):
        """Update bounding box histograms based on the selected class."""
        bbox_df = data.get("bbox_stats", pd.DataFrame())
        if bbox_df.empty or selected_class not in bbox_df["class"].values:
            empty = go.Figure()
            empty.update_layout(title="No data available")
            return empty, empty

        filtered = bbox_df[bbox_df["class"] == selected_class]

        fig_area = px.histogram(
            filtered,
            x="area",
            nbins=60,
            title=f"BBox Area Distribution — {selected_class}",
            color_discrete_sequence=["#3498db"],
            labels={"area": "Area (px²)"},
        )
        fig_area.update_layout(bargap=0.05)

        ar_clipped = filtered[filtered["aspect_ratio"].between(0.05, 8)]
        fig_ar = px.histogram(
            ar_clipped,
            x="aspect_ratio",
            nbins=60,
            title=f"Aspect Ratio Distribution — {selected_class}",
            color_discrete_sequence=["#e67e22"],
            labels={"aspect_ratio": "Width / Height"},
        )
        fig_ar.update_layout(bargap=0.05)

        return fig_area, fig_ar

    return app


# ── Private chart helpers ──────────────────────────────────────────────────────


def _make_class_dist_chart(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return go.Figure()
    return px.bar(
        df,
        x="class",
        y="count",
        title="Total Annotations per Class",
        color="class",
        labels={"count": "Annotation Count"},
        text_auto=True,
    )


def _make_class_dist_pie(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return go.Figure()
    return px.pie(
        df,
        names="class",
        values="count",
        title="Share of Annotations per Class",
        hole=0.35,
    )


def _make_train_val_chart(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Train %", x=df["class"], y=df["train_pct"]))
    fig.add_trace(go.Bar(name="Val %", x=df["class"], y=df["val_pct"]))
    fig.update_layout(
        barmode="group",
        title="Class Distribution: Train vs Validation (%)",
        xaxis_title="Class",
        yaxis_title="Percentage (%)",
    )
    return fig


def _make_scene_chart(df: pd.DataFrame, attribute: str) -> go.Figure:
    if df.empty:
        return go.Figure()
    filtered = df[df["attribute"] == attribute]
    return px.bar(
        filtered,
        x="value",
        y="count",
        title=attribute.replace("_", " ").title(),
        color="value",
        text_auto=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BDD100K Analysis Dashboard")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/data/precomputed",
        help="Directory containing pre-computed CSV files",
    )
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    data = load_data(args.data_dir)
    application = build_app(data)
    application.run(host="0.0.0.0", port=args.port, debug=args.debug)
