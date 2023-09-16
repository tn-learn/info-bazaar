import random
from datetime import datetime
from itertools import zip_longest
from pathlib import Path

import numpy as np
from speedrun import BaseExperiment, register_default_dispatch, IOMixin

from bazaar.lem_utils import (
    default_llm_name,
    global_embedding_manager,
    default_embedding_name,
    default_reranker_name,
    get_closed_book_answer,
    get_open_book_answer,
)
from bazaar.py_utils import dump_dict, load_dict, root_dir_slash
from bazaar.schema import BulletinBoard, Answer
from bazaar.sim_builder import (
    build_buyers,
    build_authors_and_institutions, parse_questions_from_dataset,
)
from bazaar.simulator import BazaarSimulator


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


# fmt: off
specific_questions = [
    0, 4, 16, 17, 65, 83, 121, 195, 234, 247, 395, 526, 528, 727,
    804, 917, 952, 983, 1009, 1010, 1124, 1157, 1168, 1178, 1284,
    1573, 1723, 1794, 1801, 1915, 1954, 2000, 2062, 2201, 2281,
    2446, 2486, 2611, 2688, 2844, 3625, 3833, 3885, 3940, 3984,
    4129, 4165, 4188, 4289, 4369, 4427, 4630, 4764, 5108, 5246,
]

general_questions = [
    10, 113, 114, 140, 173, 183, 210, 356, 378, 514, 658, 691, 699,
    735, 840, 1144, 1210, 1241, 1318, 1837, 1863, 1971, 2005, 2230,
    2337, 2431, 2432, 2675, 2729, 2799, 2808, 2954, 2991, 3391, 3461,
    3596, 3600, 3601, 3655, 3691, 3729, 3732, 3858, 4025, 4146, 4259,
    4472, 4499, 4618, 4639, 4705, 4904, 4947, 5046, 5384
]
# fmt: on


def flush_print(*args, **kwargs):
    print(*args, flush=True, **kwargs)


class SimulationRunner(BaseExperiment, IOMixin):
    def __init__(self, skip_setup: bool = False):
        super(SimulationRunner, self).__init__()
        if not skip_setup:
            self.auto_setup()

    def _select_buyers(self, buyers: list) -> list:
        def slicey(list_: list, slice_str: str):
            if slice_str is None:
                pass
            elif ":" in slice_str:
                start, stop, *step = slice_str.split(":")
                start = int(start)
                stop = int(stop)
                if len(step) > 0:
                    step = int(step[0])
                    list_ = list_[start:stop:step]
                else:
                    list_ = list_[start:stop]
            elif "," in slice_str:
                list_ = [list_[int(idx)] for idx in slice_str.split(",")]
            elif slice_str.isdigit():
                list_ = [list_[int(slice_str)]]
            else:
                raise ValueError(f"Invalid slice string: {slice_str}")
            return list_

        if self.get("question_type") == "general":
            indices = slicey(general_questions, self.get("query_range"))
            buyers = [buyers[idx] for idx in indices]
        elif self.get("question_type") == "specific":
            indices = slicey(specific_questions, self.get("query_range"))
            buyers = [buyers[idx] for idx in indices]
        elif self.get("question_type") == "mixed":
            # Interleave general and specific question indices
            mixed_questions = [
                x
                for pair in zip_longest(general_questions, specific_questions)
                if pair is not None
                for x in pair
                if x is not None
            ]
            indices = slicey(mixed_questions, self.get("query_range"))
            buyers = [buyers[idx] for idx in indices]
        else:
            buyers = slicey(buyers, self.get("query_range"))
        return buyers

    def _build(self):
        self.set_printer(printer=flush_print)
        set_seed(self.get("rng_seed"))

        # Set the LLM and embedding names
        if self.get("embedding_manager_path") is not None:
            global_embedding_manager(
                init_from_path=root_dir_slash(self.get("embedding_manager_path"))
            )
        default_llm_name(set_to=self.get("llm_name"))
        default_embedding_name(set_to=self.get("embedding_name"))
        default_reranker_name(set_to=self.get("reranker_name"))

        # Load the dataset
        dataset = load_dict(root_dir_slash(self.get("dataset_path")))
        if self.get("questions_path") is not None:
            print(f"Loading questions from {self.get('questions_path')}...")
            questions = load_dict(root_dir_slash(self.get("questions_path")))
        else:
            print("Getting questions from dataset...")
            questions = parse_questions_from_dataset(dataset)
        print(f"Loaded {len(questions)} questions.")
        rng = np.random.RandomState(self.get("rng_seed"))

        buyers = build_buyers(
            questions=questions,
            buyer_max_budget_mean=self.get("buyer_max_budget_mean"),
            buyer_max_budget_sigma=self.get("buyer_max_budget_sigma"),
            buyer_urgency_min=self.get("buyer_urgency_min"),
            buyer_urgency_max=self.get("buyer_urgency_max"),
            query_creation_time_start=self.get("query_creation_time_start"),
            query_creation_time_end=self.get("query_creation_time_end"),
            rng=rng,
        )
        buyers = self._select_buyers(buyers)

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
                vendor_principals[idx] for idx in vendor_indices_to_keep
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

    def heartbeat(self, now: int) -> "SimulationRunner":
        # Touches a file to indicate that the simulation is still running
        (Path(self.log_directory) / "heartbeat.txt").touch()
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
            Path(self.log_directory) / "bazaar_summary.json",
        )
        return self

    @register_default_dispatch
    def simulate(self) -> "SimulationRunner":
        self._build()
        # Run the sim
        if self.get("run_type") == "retrieve":
            self.bazaar.run(
                self.get("runner/duration", 168),
                print_callback=self.info,
                step_callback=self.heartbeat,
            )
        elif self.get("run_type") == "closed_book":
            for buyer_agent in self.bazaar.buyer_agents:
                closed_book_answer = get_closed_book_answer(
                    question=buyer_agent.principal.query.text,
                    model_name=self.get("llm_name"),
                )
                answer = Answer(
                    success=True,
                    text=closed_book_answer,
                )
                buyer_agent.principal.answer = answer
        elif self.get("run_type") == "open_book":
            for buyer_agent in self.bazaar.buyer_agents:
                open_book_answer = get_open_book_answer(
                    question=buyer_agent.principal.query.text,
                    gold_passage=buyer_agent.principal.query._gold_block.content,
                    model_name=self.get("llm_name"),
                )
                answer = Answer(
                    success=True,
                    text=open_book_answer,
                )
                buyer_agent.principal.answer = answer

        # Print the results and dump summary
        self.print_results().dump_simulation_summary()
        # Done
        self.print("Done.")
        return self


if __name__ == "__main__":
    runner = SimulationRunner()
    runner.run()
