import reflex as rx
from app.components.base_layout import base_layout
from app.pages.index_page import index_page
from app.pages.clustering_page import clustering_page
from app.pages.profiles_page import profiles_page
from app.pages.insights_page import insights_page
from app.analysis_state import AnalysisState


def index() -> rx.Component:
    return base_layout(index_page())


def clustering() -> rx.Component:
    return base_layout(clustering_page())


def profiles() -> rx.Component:
    return base_layout(profiles_page())


def insights() -> rx.Component:
    return base_layout(insights_page())


app = rx.App(
    theme=rx.theme(appearance="light", accent_color="sky"),
    head_components=[
        rx.el.link(rel="preconnect", href="https://fonts.googleapis.com"),
        rx.el.link(rel="preconnect", href="https://fonts.gstatic.com", cross_origin=""),
        rx.el.link(
            href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap",
            rel="stylesheet",
        ),
    ],
)
app.add_page(index, route="/", on_load=AnalysisState.load_and_analyze_data)
app.add_page(
    clustering, route="/clustering", on_load=AnalysisState.load_and_analyze_data
)
app.add_page(profiles, route="/profiles", on_load=AnalysisState.load_and_analyze_data)
app.add_page(insights, route="/insights", on_load=AnalysisState.load_and_analyze_data)