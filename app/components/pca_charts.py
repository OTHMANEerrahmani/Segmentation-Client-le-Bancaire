import reflex as rx
import plotly.graph_objects as go
from app.analysis_state import AnalysisState

TOOLTIP_PROPS = {
    "content_style": {
        "background": "white",
        "border_color": "#e5e7eb",
        "border_radius": "0.5rem",
        "font_family": "Poppins",
        "font_size": "14px",
    },
    "label_style": {"color": "#1f2937", "font_weight": "600"},
    "separator": ": ",
}


def variance_explained_chart() -> rx.Component:
    """Chart showing variance explained by each principal component."""
    return rx.el.div(
        rx.el.div(
            rx.el.h3(
                "Explained Variance per Component",
                class_name="text-lg font-semibold text-gray-800",
            ),
            rx.el.button(
                rx.icon("download", class_name="w-4 h-4"),
                on_click=lambda: AnalysisState.download_chart("variance"),
                class_name="p-2 rounded-md hover:bg-gray-100 transition-colors",
                title="Download as PNG",
            ),
            class_name="flex justify-between items-center mb-4",
        ),
        rx.recharts.composed_chart(
            rx.recharts.cartesian_grid(stroke_dasharray="3 3", stroke="#e5e7eb"),
            rx.recharts.graphing_tooltip(**TOOLTIP_PROPS),
            rx.recharts.x_axis(
                data_key="name",
                tick_line=False,
                axis_line=False,
                custom_attrs={"fontFamily": "Poppins"},
            ),
            rx.recharts.y_axis(
                y_axis_id="left",
                orientation="left",
                tick_line=False,
                axis_line=False,
                custom_attrs={"fontFamily": "Poppins"},
            ),
            rx.recharts.y_axis(
                y_axis_id="right",
                orientation="right",
                tick_line=False,
                axis_line=False,
                custom_attrs={"fontFamily": "Poppins"},
            ),
            rx.recharts.legend(),
            rx.recharts.bar(
                data_key="variance",
                bar_size=20,
                fill="#38bdf8",
                y_axis_id="left",
                name="Individual Variance",
            ),
            rx.recharts.line(
                type="monotone",
                data_key="cumulative",
                stroke="#ef4444",
                y_axis_id="right",
                name="Cumulative Variance",
            ),
            data=AnalysisState.pca_variance_data,
            height=300,
            width="100%",
        ),
        class_name="p-6 bg-white rounded-2xl shadow-[0_1px_3px_rgba(0,0,0,0.1)]",
    )


def pca_biplot() -> rx.Component:
    """Interactive biplot of the first two principal components."""
    return rx.el.div(
        rx.el.div(
            rx.el.h3("PCA Biplot", class_name="text-lg font-semibold text-gray-800"),
            rx.el.button(
                rx.icon("download", class_name="w-4 h-4"),
                on_click=lambda: AnalysisState.download_chart("biplot"),
                class_name="p-2 rounded-md hover:bg-gray-100 transition-colors",
                title="Download as PNG",
            ),
            class_name="flex justify-between items-center mb-4",
        ),
        rx.plotly(
            data=AnalysisState.pca_biplot,
            layout={"height": "500"},
            config={"displayModeBar": False},
        ),
        class_name="p-6 bg-white rounded-2xl shadow-[0_1px_3px_rgba(0,0,0,0.1)]",
    )


def correlation_heatmap() -> rx.Component:
    """Heatmap of variable correlations."""
    return rx.el.div(
        rx.el.div(
            rx.el.h3(
                "Variable Correlation Heatmap",
                class_name="text-lg font-semibold text-gray-800",
            ),
            rx.el.button(
                rx.icon("download", class_name="w-4 h-4"),
                on_click=lambda: AnalysisState.download_chart("heatmap"),
                class_name="p-2 rounded-md hover:bg-gray-100 transition-colors",
                title="Download as PNG",
            ),
            class_name="flex justify-between items-center mb-4",
        ),
        rx.plotly(
            data=AnalysisState.correlation_heatmap,
            layout={"height": "500"},
            config={"displayModeBar": False},
        ),
        class_name="p-6 bg-white rounded-2xl shadow-[0_1px_3px_rgba(0,0,0,0.1)]",
    )