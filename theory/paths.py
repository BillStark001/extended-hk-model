"""Path registrations for theory tasks."""

from ehk.common.settings import register_path


MESOSCOPIC_OUTPUT = register_path(
    "EHK_THEORY_MESOSCOPIC_OUTPUT_DIR", "outputs/theory/mesoscopic"
)
FORCE_LANDSCAPE_OUTPUT = register_path(
    "EHK_THEORY_FORCE_LANDSCAPE_OUTPUT_DIR", "outputs/theory/force_landscape"
)
