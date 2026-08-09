"""Path registrations for paper experiments."""

from ehk.common.settings import register_path


STAT_INPUT = register_path(
    "SIMULATION_STAT_DIR", "outputs/experiments/paper"
)
MAIN_SCAN_OUTPUT = register_path(
    "EHK_MAIN_SCAN_OUTPUT_DIR", "outputs/experiments/paper/main_scan"
)
EPSILON_SCAN_OUTPUT = register_path(
    "EHK_EPSILON_SCAN_OUTPUT_DIR", "outputs/experiments/paper/epsilon_scan"
)
REPLICATE_OUTPUT = register_path(
    "EHK_REPLICATE_OUTPUT_DIR", "outputs/experiments/paper/replicate"
)
MECHANISM_OUTPUT = register_path(
    "EHK_MECHANISM_OUTPUT_DIR", "outputs/experiments/paper/mechanism_cases"
)
PATHWAY_OUTPUT = register_path(
    "EHK_PATHWAY_OUTPUT_DIR", "outputs/experiments/paper/pathway_index"
)
ILLUSTRATION_OUTPUT = register_path(
    "EHK_ILLUSTRATION_OUTPUT_DIR", "outputs/experiments/paper/illustrations"
)
