import reflex as rx
from app.analysis_state import AnalysisState
from app.components.pca_charts import TOOLTIP_PROPS


def dendrogram_chart() -> rx.Component:
    return rx.el.div(
        rx.el.div(
            rx.el.h3(
                "Hierarchical Clustering Dendrogram",
                class_name="text-lg font-semibold text-gray-800",
            ),
            rx.el.button(
                rx.icon("download", class_name="w-4 h-4"),
                on_click=lambda: AnalysisState.download_chart("dendrogram"),
                class_name="p-2 rounded-md hover:bg-gray-100 transition-colors",
                title="Download as PNG",
            ),
            class_name="flex justify-between items-center mb-4",
        ),
        rx.plotly(
            data=AnalysisState.dendrogram_fig,
            layout={"height": "500"},
            config={"displayModeBar": False},
        ),
        class_name="p-6 bg-white rounded-2xl shadow-[0_1px_3px_rgba(0,0,0,0.1)]",
    )


def cluster_distribution_chart() -> rx.Component:
    return rx.el.div(
        rx.el.div(
            rx.el.h3(
                "Cluster Distribution", class_name="text-lg font-semibold text-gray-800"
            ),
            rx.el.button(
                rx.icon("download", class_name="w-4 h-4"),
                on_click=lambda: AnalysisState.download_chart("distribution"),
                class_name="p-2 rounded-md hover:bg-gray-100 transition-colors",
                title="Download as PNG",
            ),
            class_name="flex justify-between items-center mb-4",
        ),
        rx.recharts.pie_chart(
            rx.recharts.graphing_tooltip(**TOOLTIP_PROPS),
            rx.recharts.pie(
                data_key="value",
                name_key="name",
                data=AnalysisState.cluster_distribution_data,
                cx="50%",
                cy="50%",
            ),
            rx.recharts.legend(),
            height=300,
            width="100%",
        ),
        class_name="p-6 bg-white rounded-2xl shadow-[0_1px_3px_rgba(0,0,0,0.1)]",
    )


def segment_profile_card(profile: rx.Var, radar_chart: rx.Var) -> rx.Component:
    return rx.el.div(
        rx.el.div(
            rx.el.h4(
                profile["name"],
                class_name="text-xl font-bold",
                style={"color": profile["color"]},
            ),
            rx.el.span(
                f"{profile['size']} Customers",
                class_name="text-sm font-medium text-gray-500",
            ),
            class_name="flex justify-between items-center mb-4",
        ),
        rx.el.div(
            rx.plotly(
                data=radar_chart,
                config={"displayModeBar": False},
                layout={"height": 300},
            ),
            class_name="w-full",
        ),
        class_name="p-6 bg-white rounded-2xl shadow-[0_1px_3px_rgba(0,0,0,0.1)]",
    )


def clustering_page() -> rx.Component:
    """The page for visualizing clustering results."""
    return rx.el.div(
        rx.el.h1(
            "Customer Clustering", class_name="text-3xl font-bold text-gray-900 mb-2"
        ),
        rx.el.p(
            "Hierarchical clustering and segment profiling.",
            class_name="text-gray-600 mb-8",
        ),
        rx.cond(
            AnalysisState.is_loading,
            rx.el.div(
                rx.spinner(size="3"),
                rx.el.p("Performing analysis...", class_name="text-sky-600"),
                class_name="flex flex-col items-center justify-center h-96 gap-4",
            ),
            rx.el.div(
                rx.el.div(
                    dendrogram_chart(),
                    cluster_distribution_chart(),
                    class_name="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8",
                ),
                rx.el.h2(
                    "Segment Profiles",
                    class_name="text-2xl font-bold text-gray-800 my-8",
                ),
                rx.el.div(
                    rx.foreach(
                        AnalysisState.segment_profiles,
                        lambda profile, index: segment_profile_card(
                            profile, AnalysisState.segment_radar_charts[index]
                        ),
                    ),
                    class_name="grid grid-cols-1 md:grid-cols-2 gap-6",
                ),
                class_name="flex flex-col gap-8",
            ),
        ),
        class_name="p-4 sm:p-6 md:p-8",
    )