import reflex as rx
from app.analysis_state import DataCleaningState


def step_badge(step_key: str, label: str) -> rx.Component:
    active = DataCleaningState.step == step_key
    classes = rx.cond(
        active,
        "px-3 py-1 rounded-full bg-sky-100 text-sky-700 text-sm",
        "px-3 py-1 rounded-full bg-gray-100 text-gray-600 text-sm",
    )
    return rx.el.span(label, class_name=classes)


def stat_card(name: str, value: str, icon: str) -> rx.Component:
    return rx.el.div(
        rx.icon(tag=icon, class_name="w-6 h-6 text-gray-500 mb-1"),
        rx.el.p(name, class_name="text-sm text-gray-600"),
        rx.el.h3(value, class_name="text-2xl font-bold text-gray-900"),
        class_name="p-5 bg-white rounded-2xl shadow-[0_1px_3px_rgba(0,0,0,0.1)] text-center",
    )


def cleaning_page() -> rx.Component:
    return rx.el.div(
        rx.el.h1("Data Cleaning", class_name="text-3xl font-bold text-gray-900 mb-2"),
        rx.el.p(
            "Upload a CSV, run the 4-step cleaning pipeline, and export results.",
            class_name="text-gray-600 mb-6",
        ),
        rx.el.div(
            step_badge("upload", "Upload"),
            rx.icon(tag="chevron-right"),
            step_badge("control", "Control"),
            rx.icon(tag="chevron-right"),
            step_badge("diagnose", "Diagnose"),
            rx.icon(tag="chevron-right"),
            step_badge("treat", "Treat"),
            rx.icon(tag="chevron-right"),
            step_badge("report", "Report"),
            class_name="flex items-center gap-2 mb-6",
        ),
        rx.el.div(
            rx.el.div(
                rx.el.div(
                    rx.el.p("1) Upload CSV", class_name="font-medium text-gray-800"),
                    rx.upload(
                        rx.button(
                            "Select bank_customers.csv",
                            class_name="px-4 py-2 rounded-lg bg-sky-600 text-white hover:bg-sky-700",
                        ),
                        accept=".csv,.xlsx",
                        on_drop=DataCleaningState.upload_csv(rx.upload_files()),
                    ),
                    rx.el.p(
                        "A backup of the raw file is kept in memory.",
                        class_name="text-sm text-gray-500 mt-2",
                    ),
                    class_name="p-5 bg-white rounded-2xl border border-gray-200",
                ),
                rx.el.div(
                    rx.el.p("2-4) Run Cleaning", class_name="font-medium text-gray-800"),
                    rx.button(
                        rx.cond(
                            DataCleaningState.is_processing,
                            rx.el.span("Processing..."),
                            rx.el.span("Run Cleaning Pipeline"),
                        ),
                        on_click=DataCleaningState.run_cleaning,
                        disabled=~DataCleaningState.has_upload | DataCleaningState.is_processing,
                        class_name="px-4 py-2 rounded-lg bg-gray-900 text-white hover:bg-gray-800",
                    ),
                    class_name="p-5 bg-white rounded-2xl border border-gray-200",
                ),
                class_name="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6",
            ),
            rx.el.div(
                rx.el.p("Summary", class_name="font-medium text-gray-800 mb-3"),
                rx.el.div(
                    rx.foreach(
                        DataCleaningState.summary_stats,
                        lambda item: stat_card(item["name"], item["value"], "bar-chart-2"),
                    ),
                    class_name="grid grid-cols-2 md:grid-cols-4 gap-4",
                ),
                class_name="p-5 bg-white rounded-2xl border border-gray-200",
            ),
            rx.el.div(
                rx.el.p("Actions", class_name="font-medium text-gray-800 mb-3"),
                rx.el.div(
                    rx.button(
                        "Preview Clean Data",
                        on_click=rx.window_alert("Preview shows first 5 rows in console."),
                        class_name="px-4 py-2 rounded-lg bg-sky-100 text-sky-700 hover:bg-sky-200",
                    ),
                    rx.button(
                        "Download Clean CSV",
                        on_click=DataCleaningState.download_clean_csv,
                        disabled=~DataCleaningState.has_clean,
                        class_name="px-4 py-2 rounded-lg bg-emerald-600 text-white hover:bg-emerald-700",
                    ),
                    rx.button(
                        "View Cleaning Log",
                        on_click=DataCleaningState.download_report_csv,
                        disabled=~DataCleaningState.has_report,
                        class_name="px-4 py-2 rounded-lg bg-amber-600 text-white hover:bg-amber-700",
                    ),
                    class_name="flex flex-wrap gap-3",
                ),
                class_name="p-5 bg-white rounded-2xl border border-gray-200",
            ),
            class_name="flex flex-col gap-6",
        ),
        class_name="p-4 sm:p-6 md:p-8",
    )


