"""Diffucore UI — entry point."""

from ui import build_ui, CSS, THEME

app = build_ui()
app.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False,
    theme=THEME,
    css=CSS,
)
