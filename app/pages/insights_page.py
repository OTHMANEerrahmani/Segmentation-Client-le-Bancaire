import reflex as rx
from app.analysis_state import AnalysisState
from app.pages.profiles_page import marketing_recommendation_card


def kpi_card(title: str, value: rx.Var, icon: str, color: str) -> rx.Component:
    """A card for displaying a key performance indicator."""
    return rx.el.div(
        rx.el.div(
            rx.icon(tag=icon, class_name="w-8 h-8"),
            class_name=f"p-3 rounded-full {color}",
        ),
        rx.el.div(
            rx.el.p(title, class_name="text-sm font-medium text-gray-500"),
            rx.el.h3(value, class_name="text-3xl font-bold text-gray-900"),
            class_name="flex flex-col",
        ),
        class_name="flex items-center gap-4 p-6 bg-white rounded-2xl shadow-[0_1px_3px_rgba(0,0,0,0.1)]",
    )


def insights_page() -> rx.Component:
    """The page for displaying summary insights and KPIs."""
    return rx.el.div(
        rx.el.h1(
            "Dashboard Insights", class_name="text-3xl font-bold text-gray-900 mb-2"
        ),
        rx.el.p(
            "A high-level overview of customer segments and key financial metrics.",
            class_name="text-gray-600 mb-8",
        ),
        rx.cond(
            AnalysisState.is_loading,
            rx.el.div(
                rx.spinner(size="3"),
                rx.el.p("Analyzing insights...", class_name="text-sky-600"),
                class_name="flex flex-col items-center justify-center h-96 gap-4",
            ),
            rx.el.div(
                rx.el.div(
                    kpi_card(
                        "Total Customers",
                        AnalysisState.total_customers.to_string(),
                        "users",
                        "bg-sky-100 text-sky-600",
                    ),
                    kpi_card(
                        "Average Income",
                        f"${AnalysisState.avg_income.to_string()}",
                        "dollar-sign",
                        "bg-emerald-100 text-emerald-600",
                    ),
                    kpi_card(
                        "Average Balance",
                        f"${AnalysisState.avg_balance.to_string()}",
                        "piggy-bank",
                        "bg-amber-100 text-amber-600",
                    ),
                    kpi_card(
                        "Average Credit Score",
                        AnalysisState.avg_credit_score.to_string(),
                        "circle_gauge",
                        "bg-indigo-100 text-indigo-600",
                    ),
                    class_name="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8",
                ),
                rx.el.div(
                    kpi_card(
                        "Largest Segment",
                        AnalysisState.largest_segment,
                        "crown",
                        "bg-rose-100 text-rose-600",
                    ),
                    kpi_card(
                        "Highest Income Segment",
                        AnalysisState.highest_income_segment,
                        "trending-up",
                        "bg-teal-100 text-teal-600",
                    ),
                    class_name="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8",
                ),
                rx.el.h2(
                    "AI-Powered Marketing Suggestions",
                    class_name="text-2xl font-bold text-gray-800 mt-12 mb-6",
                ),
                rx.el.div(
                    rx.foreach(
                        AnalysisState.marketing_recommendations.keys(),
                        lambda rec_name: marketing_recommendation_card(
                            AnalysisState.marketing_recommendations[rec_name], rec_name
                        ),
                    ),
                    class_name="grid grid-cols-1 md:grid-cols-2 gap-6",
                ),
            ),
        ),
        class_name="p-4 sm:p-6 md:p-8",
    )