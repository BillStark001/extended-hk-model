import argparse

from ehk.common.io.files import init_logger
from ehk.micro.runner import run_scenarios
import experiments.paper.scenarios as cfg
from experiments.paths import SMP_BINARY
from experiments.workspaces import workspace_dir

scenarios = {
    "epsilon": cfg.all_scenarios_eps,
    "gradation": cfg.all_scenarios_grad,
    "mech": cfg.all_scenarios_mech,
    "replicate": cfg.all_scenarios_rep,
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run simulations for a given scenario."
    )
    parser.add_argument(
        "scenario",
        choices=scenarios.keys(),
        help="The scenario to run simulations for.",
    )
    parser.add_argument(
        "--workspace", help="The workspace name to store simulation results."
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Number of concurrent simulations to run.",
    )

    args = parser.parse_args()

    if not args.scenario:
        print("Please specify a scenario to run simulations for.")
        exit(1)
    if args.scenario not in scenarios:
        print(
            f'Invalid scenario specified. Available scenarios: {", ".join(scenarios.keys())}'
        )
        exit(1)

    simulation_result_dir = workspace_dir(args.workspace, args.scenario)

    simulation_result_dir.mkdir(parents=True, exist_ok=True)
    init_logger(None, str(simulation_result_dir / "logfile.log"))

    print(f"Result Directory: {simulation_result_dir}")

    run_scenarios(
        SMP_BINARY.resolve(),
        simulation_result_dir,
        scenarios[args.scenario],
        concurrency=args.concurrency,
    )
