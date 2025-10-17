import reflex as rx
from app.analysis_state import AnalysisState
from app.components.pca_charts import (
    variance_explained_chart,
    pca_biplot,
    correlation_heatmap,
)


def metric_card(title: str, value: rx.Var, icon: str) -> rx.Component:
    return rx.el.div(
        rx.icon(tag=icon, class_name="w-6 h-6 text-gray-500 mb-2"),
        rx.el.p(title, class_name="text-sm font-medium text-gray-600"),
        rx.el.h3(value, class_name="text-2xl font-bold text-gray-900"),
        class_name="p-6 bg-white rounded-2xl shadow-[0_1px_3px_rgba(0,0,0,0.1)] text-center",
    )


def index_page() -> rx.Component:
    """The main page for PCA analysis visualization."""
    return rx.el.div(
        rx.el.h1("PCA Analysis", class_name="text-3xl font-bold text-gray-900 mb-2"),
        rx.el.p(
            "Principal Component Analysis for dimensionality reduction.",
            class_name="text-gray-600 mb-8",
        ),
        rx.cond(
            AnalysisState.is_loading,
            rx.el.div(
                rx.spinner(size="3"),
                rx.el.p("Performing analysis..."),
                class_name="flex flex-col items-center justify-center h-96 gap-4 text-sky-600",
            ),
            rx.el.div(
                rx.el.div(
                    metric_card(
                        "Total Customers",
                        AnalysisState.total_customers.to_string(),
                        "users",
                    ),
                    metric_card(
                        "Quantitative Variables",
                        AnalysisState.quant_vars.length(),
                        "list-ordered",
                    ),
                    metric_card(
                        "Optimal Components (80% Var)",
                        AnalysisState.n_components_80.to_string(),
                        "bar-chart-big",
                    ),
                    class_name="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8",
                ),
                rx.el.div(
                    variance_explained_chart(),
                    pca_biplot(),
                    correlation_heatmap(),
                    class_name="grid grid-cols-1 lg:grid-cols-1 gap-6",
                ),
                class_name="flex flex-col gap-8",
            ),
        ),
        class_name="p-4 sm:p-6 md:p-8",
    )