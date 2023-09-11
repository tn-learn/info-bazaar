import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from speedrun import BaseExperiment, register_default_dispatch, IOMixin

from bazaar.lem_utils import (
    default_llm_name,
    global_embedding_manager,
    default_embedding_name, default_reranker_name,
)
from bazaar.py_utils import dump_dict, load_dict, root_dir_slash
from bazaar.schema import BulletinBoard
from bazaar.sim_builder import (
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
    def __init__(self, skip_setup: bool = False):
        super(SimulationRunner, self).__init__()
        if not skip_setup:
            self.auto_setup()

    def _build(self):
        set_seed(self.get("rng_seed"))

        # Set the LLM and embedding names
        global_embedding_manager(init_from_path=root_dir_slash(self.get("embedding_manager_path")))
        default_llm_name(set_to=self.get("llm_name"))
        default_embedding_name(set_to=self.get("embedding_name"))
        default_reranker_name(set_to=self.get("reranker_name"))

        # Load the dataset
        dataset = load_dict(root_dir_slash(self.get("dataset_path")))
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
        buyers = buyers[parse_to_slice(self.get("query_range"))]

        # Filter out the vendors that don't have a block to sell
        vendor_principals = [
            vendor_principal
            for vendor_principal in (institutions + authors)
            if vendor_principal.num_blocks_owned > 0
        ]

        # Remove vendors if required
        if self.get("fraction_active_vendors") < 1.0:
            num_vendors_to_keep = round(
                len(vendor_principals) * self.get("fraction_active_vendors")
            )
            vendor_indices_to_keep = rng.choice(
                len(vendor_principals), size=num_vendors_to_keep, replace=False
            )
            vendor_principals = [
                vendor_principals[idx]
                for idx in vendor_indices_to_keep
            ]

        # Init the bazaar
        self.bazaar = BazaarSimulator(
            bulletin_board=bulletin_board,
            buyer_principals=buyers,
            vendor_principals=vendor_principals,
            seed=self.get("rng_seed"),
            buyer_agent_kwargs=self.get("buyer_agent_kwargs"),
            vendor_agent_kwargs=self.get("vendor_agent_kwargs"),
        )

    def info(self, *message) -> "SimulationRunner":
        # Add datetime stamp to message and print
        message = " ".join([str(m) for m in message])
        self.print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")
        return self

    def print_results(self) -> "SimulationRunner":
        # Get the buyer principals
        buyer_principals = [agent.principal for agent in self.bazaar.buyer_agents]
        for buyer_principal in buyer_principals:
            self.print("-" * 80)
            self.print(f"Question: {buyer_principal.query.text}")
            if buyer_principal.answer.success:
                self.print(f"Answer: {buyer_principal.answer.text}")
                self.print("==== References ====")
                for block_idx, block in enumerate(buyer_principal.answer.blocks):
                    print(f"[Ref {block_idx + 1}] ", block.document_title)
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
        self._build()
        # Run the sim
        self.bazaar.run(self.get("runner/duration", 168), print_callback=self.info)
        # Print the results and dump summary
        self.print_results().dump_simulation_summary()
        # Done
        self.print("Done.")
        return self


if __name__ == '__main__':
    runner = SimulationRunner()
    runner.run()