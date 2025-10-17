import reflex as rx
from app.analysis_state import AnalysisState


def sidebar_item(text: str, icon: str, url: str) -> rx.Component:
    """A sidebar item with icon and text."""
    return rx.el.a(
        rx.el.div(
            rx.icon(tag=icon, class_name="w-5 h-5"),
            rx.el.span(text, class_name="font-medium"),
            class_name=rx.cond(
                AnalysisState.router.page.path == url,
                "flex items-center gap-3 rounded-lg bg-sky-100 px-3 py-2 text-sky-700 transition-all hover:text-sky-800",
                "flex items-center gap-3 rounded-lg px-3 py-2 text-gray-500 transition-all hover:text-gray-900",
            ),
        ),
        href=url,
    )


def sidebar() -> rx.Component:
    """The sidebar for navigation."""
    return rx.el.div(
        rx.el.div(
            rx.el.a(
                rx.icon(tag="area-chart", class_name="h-8 w-8 text-sky-600"),
                rx.el.span("Client-Segment", class_name="text-lg font-semibold"),
                href="/",
                class_name="flex items-center gap-2",
            ),
            rx.el.nav(
                sidebar_item("Data Cleaning", "sparkles", "/cleaning"),
                sidebar_item("PCA Analysis", "bar-chart-2", "/"),
                sidebar_item("Clustering", "users", "/clustering"),
                sidebar_item("Customer Profiles", "user-cog", "/profiles"),
                sidebar_item("Insights", "lightbulb", "/insights"),
                class_name="flex flex-col gap-1 mt-8",
            ),
            class_name="flex-1",
        ),
        class_name="hidden border-r bg-gray-50/50 md:flex flex-col gap-2 p-4 min-h-screen w-64 fixed",
    )