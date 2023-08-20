import random
from pathlib import Path
from typing import Optional

import numpy as np
from speedrun import BaseExperiment, register_default_dispatch, IOMixin

from bazaar.py_utils import dump_dict
from bazaar.sim_builder import SimulationConfig, load
from bazaar.simulator import BazaarSimulator


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def parse_to_slice(s: str) -> slice:
    if s is None:
        return slice(0, None)
    else:
        start, stop = s.split(":")
        return slice(int(start), int(stop))


class SimulationRunner(BaseExperiment, IOMixin):
    def __init__(self, experiment_directory: Optional[str] = None):
        super().__init__(experiment_directory)
        self.auto_setup()

    def _build_simulation(self):
        self.sim_config = sim_config = SimulationConfig(**self.get("sim_config"))
        set_seed(sim_config.rng_seed)
        # Load the dataset
        principals = load(
            path=self.get("data/path"),
            config=sim_config,
        )
        buyer_principals = principals["buyers"][
            parse_to_slice(self.get("data/query_range"))
        ]
        vendor_principals = principals["institutions"] + principals["authors"]
        bulletin_board = principals["bulletin_board"]
        # Build the simulator
        self.sim = BazaarSimulator(
            bulletin_board=bulletin_board,
            buyer_principals=buyer_principals,
            vendor_principals=vendor_principals,
            seed=sim_config.rng_seed,
            buyer_agent_kwargs=sim_config.buyer_agent_kwargs,
            vendor_agent_kwargs=sim_config.vendor_agent_kwargs,
        )
        # Configure printing
        self.print_to_file(True, "prints.txt")

    def print_results(self) -> "SimulationRunner":
        # Get the buyer principals
        buyer_principals = [agent.principal for agent in self.sim.buyer_agents]
        for buyer_principal in buyer_principals:
            self.print("-" * 80)
            self.print(f"Question: {buyer_principal.query.text}")
            if buyer_principal.answer.success:
                self.print(f"Answer: {buyer_principal.answer.text}")
            else:
                self.print("No answer found.")
        return self

    def dump_simulation_summary(self) -> "SimulationRunner":
        dump_dict(
            self.sim.evaluation_summary(),
            Path(self.log_directory) / "baazar_summary.json",
        )
        return self

    @register_default_dispatch
    def simulate(self) -> "SimulationRunner":
        self._build_simulation()
        # Run the sim
        self.sim.run(self.get("runner/duration", 168))
        # Print the results and dump summary
        self.print_results().dump_simulation_summary()
        # Done
        self.print("Done.")
        return self
