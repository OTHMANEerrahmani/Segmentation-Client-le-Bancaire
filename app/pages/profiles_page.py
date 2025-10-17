import reflex as rx
from app.analysis_state import AnalysisState


def customer_table_header() -> rx.Component:
    """The header for the customer table."""
    headers = [
        ("Customer ID", "w-1/6"),
        ("Segment", "w-1/6"),
        ("Age", "w-1/12"),
        ("Income", "w-1/6"),
        ("Balance", "w-1/6"),
        ("Credit Score", "w-1/6"),
        ("Products", "w-1/12"),
    ]
    return rx.el.thead(
        rx.el.tr(
            rx.foreach(
                headers,
                lambda header: rx.el.th(
                    header[0],
                    class_name=f"px-4 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider {header[1]}",
                ),
            ),
            class_name="bg-gray-100",
        )
    )


def customer_table_row(customer: rx.Var) -> rx.Component:
    """A single row in the customer table."""
    return rx.el.tr(
        rx.el.td(customer["customer_id"], class_name="px-4 py-3"),
        rx.el.td(
            rx.el.span(
                customer["segment"],
                class_name="px-2.5 py-1 text-xs font-medium rounded-full",
                style={
                    "color": rx.cond(
                        customer["segment"] == "Premium",
                        "#b45309",
                        rx.cond(
                            customer["segment"] == "Standard",
                            "#0284c7",
                            rx.cond(
                                customer["segment"] == "Young Starters",
                                "#047857",
                                "#b91c1c",
                            ),
                        ),
                    ),
                    "backgroundColor": rx.cond(
                        customer["segment"] == "Premium",
                        "#fef3c7",
                        rx.cond(
                            customer["segment"] == "Standard",
                            "#e0f2fe",
                            rx.cond(
                                customer["segment"] == "Young Starters",
                                "#d1fae5",
                                "#fee2e2",
                            ),
                        ),
                    ),
                },
            ),
            class_name="px-4 py-3",
        ),
        rx.el.td(customer["age"], class_name="px-4 py-3"),
        rx.el.td(f"${customer['annual_income'].to(int)}", class_name="px-4 py-3"),
        rx.el.td(f"${customer['account_balance'].to(int)}", class_name="px-4 py-3"),
        rx.el.td(customer["credit_score"], class_name="px-4 py-3"),
        rx.el.td(customer["num_products"], class_name="px-4 py-3"),
        class_name="border-b border-gray-200 text-sm text-gray-800 hover:bg-gray-50",
    )


def filter_panel() -> rx.Component:
    """A panel with multiple filter controls for the customer table."""
    return rx.el.div(
        rx.el.div(
            rx.el.div(
                rx.el.label(
                    "Age Range",
                    class_name="block text-sm font-medium text-gray-700 mb-1",
                ),
                rx.el.div(
                    rx.el.input(
                        type="number",
                        default_value=AnalysisState.age_min,
                        on_change=AnalysisState.set_age_min,
                        class_name="w-full p-2 border border-gray-300 rounded-lg",
                    ),
                    rx.el.input(
                        type="number",
                        default_value=AnalysisState.age_max,
                        on_change=AnalysisState.set_age_max,
                        class_name="w-full p-2 border border-gray-300 rounded-lg",
                    ),
                    class_name="flex gap-2",
                ),
                class_name="flex-1 min-w-[120px]",
            ),
            rx.el.div(
                rx.el.label(
                    "Income Range ($k)",
                    class_name="block text-sm font-medium text-gray-700 mb-1",
                ),
                rx.el.div(
                    rx.el.input(
                        type="number",
                        default_value=AnalysisState.income_min,
                        on_change=AnalysisState.set_income_min,
                        step=1000,
                        class_name="w-full p-2 border border-gray-300 rounded-lg",
                    ),
                    rx.el.input(
                        type="number",
                        default_value=AnalysisState.income_max,
                        on_change=AnalysisState.set_income_max,
                        step=1000,
                        class_name="w-full p-2 border border-gray-300 rounded-lg",
                    ),
                    class_name="flex gap-2",
                ),
                class_name="flex-1 min-w-[150px]",
            ),
            rx.el.div(
                rx.el.label(
                    "Credit Score",
                    class_name="block text-sm font-medium text-gray-700 mb-1",
                ),
                rx.el.div(
                    rx.el.input(
                        type="number",
                        default_value=AnalysisState.credit_score_min,
                        on_change=AnalysisState.set_credit_score_min,
                        class_name="w-full p-2 border border-gray-300 rounded-lg",
                    ),
                    rx.el.input(
                        type="number",
                        default_value=AnalysisState.credit_score_max,
                        on_change=AnalysisState.set_credit_score_max,
                        class_name="w-full p-2 border border-gray-300 rounded-lg",
                    ),
                    class_name="flex gap-2",
                ),
                class_name="flex-1 min-w-[120px]",
            ),
            rx.el.div(
                rx.el.label(
                    "Products Owned",
                    class_name="block text-sm font-medium text-gray-700 mb-1",
                ),
                rx.el.select(
                    rx.el.option("Any", value=""),
                    rx.foreach(
                        ["1", "2", "3", "4", "5"], lambda x: rx.el.option(x, value=x)
                    ),
                    on_change=AnalysisState.set_products_filter,
                    value=AnalysisState.products_filter,
                    class_name="w-full p-2 border border-gray-300 rounded-lg bg-white",
                ),
                class_name="flex-1 min-w-[150px]",
            ),
            class_name="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4",
        ),
        class_name="p-4 bg-white rounded-2xl border border-gray-200 shadow-sm mb-6",
    )


def customer_profiles_table() -> rx.Component:
    """The main table component for displaying customer data."""
    return rx.el.div(
        rx.el.h2(
            "Customer Database", class_name="text-2xl font-bold text-gray-800 mb-6"
        ),
        filter_panel(),
        rx.el.div(
            rx.el.div(
                rx.icon(
                    "search",
                    class_name="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400",
                ),
                rx.el.input(
                    placeholder="Search by Customer ID...",
                    on_change=AnalysisState.set_search_query,
                    class_name="pl-10 pr-4 py-2 w-full max-w-sm border border-gray-300 rounded-lg focus:ring-2 focus:ring-sky-500 focus:border-sky-500",
                    default_value=AnalysisState.search_query,
                ),
                class_name="relative",
            ),
            rx.el.div(
                rx.el.select(
                    AnalysisState.cluster_options,
                    placeholder="Filter by Segment",
                    value=AnalysisState.filter_cluster,
                    on_change=AnalysisState.set_filter_cluster,
                    class_name="w-full md:w-48",
                ),
                rx.el.button(
                    rx.icon("filter-x", class_name="mr-2"),
                    "Clear Filters",
                    rx.el.span(
                        AnalysisState.active_filter_count,
                        class_name=rx.cond(
                            AnalysisState.active_filter_count > 0,
                            "ml-2 px-2 py-0.5 text-xs font-semibold text-sky-700 bg-sky-100 rounded-full",
                            "hidden",
                        ),
                    ),
                    on_click=AnalysisState.clear_all_filters,
                    class_name="flex items-center bg-gray-200 text-gray-700 px-4 py-2 rounded-lg hover:bg-gray-300 transition-colors",
                ),
                rx.el.button(
                    rx.icon("download", class_name="mr-2"),
                    "Export CSV",
                    on_click=AnalysisState.download_clustered_data,
                    class_name="flex items-center bg-sky-600 text-white px-4 py-2 rounded-lg hover:bg-sky-700 transition-colors",
                ),
                class_name="flex items-center gap-4 flex-wrap",
            ),
            class_name="flex flex-col md:flex-row justify-between items-center mb-4 gap-4",
        ),
        rx.el.div(
            rx.el.table(
                customer_table_header(),
                rx.el.tbody(
                    rx.foreach(AnalysisState.filtered_customers, customer_table_row)
                ),
                class_name="min-w-full bg-white",
            ),
            class_name="overflow-x-auto rounded-lg border border-gray-200 shadow-sm",
        ),
        class_name="flex flex-col gap-4",
    )


def marketing_recommendation_card(recommendation: rx.Var, name: rx.Var) -> rx.Component:
    return rx.el.div(
        rx.el.div(
            rx.icon(
                recommendation["icon"].to(str),
                class_name=recommendation["color"].to(str) + " w-8 h-8",
            ),
            rx.el.h3(name, class_name="text-xl font-bold text-gray-900"),
            class_name="flex items-center gap-3 pb-3 border-b border-gray-200",
        ),
        rx.el.p(recommendation["description"], class_name="text-gray-600 mt-3 mb-4"),
        rx.el.ul(
            rx.foreach(
                recommendation["recommendations"],
                lambda rec: rx.el.li(
                    rx.icon(
                        "check_check",
                        class_name="w-5 h-5 text-emerald-500 mr-3 shrink-0",
                    ),
                    rx.el.span(rec),
                    class_name="flex items-start text-gray-700",
                ),
            ),
            class_name="space-y-2",
        ),
        class_name="p-6 bg-white rounded-2xl shadow-[0_1px_3px_rgba(0,0,0,0.1)]",
    )


def profiles_page() -> rx.Component:
    """The page for displaying customer profiles and marketing recommendations."""
    return rx.el.div(
        rx.el.h1(
            "Customer Profiles & Recommendations",
            class_name="text-3xl font-bold text-gray-900 mb-2",
        ),
        rx.el.p(
            "Explore customer segments, filter data, and get targeted marketing strategies.",
            class_name="text-gray-600 mb-8",
        ),
        rx.cond(
            AnalysisState.is_loading,
            rx.el.div(
                rx.spinner(size="3"),
                rx.el.p("Loading customer data...", class_name="text-sky-600"),
                class_name="flex flex-col items-center justify-center h-96 gap-4",
            ),
            rx.el.div(
                customer_profiles_table(),
                rx.el.h2(
                    "Marketing Strategy Recommendations",
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
                class_name="flex flex-col gap-8",
            ),
        ),
        class_name="p-4 sm:p-6 md:p-8",
    )