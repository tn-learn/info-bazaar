import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
from speedrun import BaseExperiment, register_default_dispatch, IOMixin

from bazaar.lem_utils import (
    default_llm_name,
    global_embedding_manager,
    default_embedding_name,
)
from bazaar.py_utils import dump_dict, load_dict
from bazaar.schema import BulletinBoard
from bazaar.sim_builder import (
    load,
    build_buyers,
    build_authors_and_institutions,
)
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

    def _build(self):
        set_seed(self.get("rng_seed"))
        self._build_simulation()

    def _build_simulation(self):
        # Set the LLM and embedding names
        global_embedding_manager(init_from_path=self.get("embedding_manager_path"))
        default_llm_name(set_to=self.get("llm_name"))
        default_embedding_name(set_to=self.get("embedding_name"))

        # Load the dataset
        dataset = load_dict(self.get("dataset_path"))
        rng = np.random.RandomState(self.get("rng_seed"))
        buyers = build_buyers(
            dataset=dataset,
            buyer_max_budget_mean=self.get("buyer_max_budget_mean"),
            buyer_max_budget_sigma=self.get("buyer_max_budget_sigma"),
            buyer_urgency_min=self.get("buyer_urgency_min"),
            buyer_urgency_max=self.get("buyer_urgency_max"),
            query_creation_time_start=self.get("query_creation_time_start"),
            query_creation_time_end=self.get("query_creation_time_end"),
            rng=rng,
        )

        authors, institutions = build_authors_and_institutions(
            dataset=dataset,
            author_fraction_of_private_blocks=self.get(
                "author_fraction_of_private_blocks"
            ),
            author_response_time_mean=self.get("author_response_time_mean"),
            author_response_time_sigma=self.get("author_response_time_sigma"),
            rng=rng,
        )
        bulletin_board = BulletinBoard()

        results = {
            "buyers": buyers,
            "authors": authors,
            "institutions": institutions,
            "bulletin_board": bulletin_board,
        }

        # results = load(path=self.get("dataset_path"), config=config)
        results["buyers"] = results["buyers"][parse_to_slice(self.get("query_range"))]
        logging.info(f"Prepared simulation for {len(results['buyers'])} queries.")

        # Make a buyer agent for each principal
        buyer_principals = results["buyers"]
        vendor_principals = results["institutions"] + results["authors"]
        # Filter out the vendors that don't have a block to sell
        vendor_principals = [
            vendor_principal
            for vendor_principal in vendor_principals
            if vendor_principal.num_blocks_owned > 0
        ]

        # Build bulletin board
        bulletin_board = results["bulletin_board"]

        # Init the bazaar
        self.bazaar = BazaarSimulator(
            bulletin_board=bulletin_board,
            buyer_principals=buyer_principals,
            vendor_principals=vendor_principals,
            seed=self.get("rng_seed"),
            buyer_agent_kwargs=self.get("buyer_agent_kwargs"),
            vendor_agent_kwargs=self.get("vendor_agent_kwargs"),
        )

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
            self.bazaar.evaluation_summary(),
            Path(self.log_directory) / "baazar_summary.json",
        )
        return self

    @register_default_dispatch
    def simulate(self) -> "SimulationRunner":
        self._build_simulation()
        # Run the sim
        self.bazaar.run(self.get("runner/duration", 168))
        # Print the results and dump summary
        self.print_results().dump_simulation_summary()
        # Done
        self.print("Done.")
        return self
