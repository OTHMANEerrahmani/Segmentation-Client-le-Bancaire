import reflex as rx
from app.components.sidebar import sidebar


def base_layout(child: rx.Component, *args, **kwargs) -> rx.Component:
    """A shared layout with a persistent sidebar."""
    return rx.el.div(
        sidebar(),
        rx.el.main(child, class_name="md:ml-64"),
        class_name="font-['Poppins'] bg-gray-50 min-h-screen",
    )