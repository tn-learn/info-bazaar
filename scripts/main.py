from pathlib import Path
from typing import Optional

import yaml
import argparse
import numpy as np
import random
import datetime

from bazaar.lem_utils import (
    default_llm_name,
    default_embedding_name,
    global_embedding_manager,
)
from bazaar.sim_builder import load, SimulationConfig
from bazaar.simulator import BazaarSimulator
from bazaar.py_utils import dump_dict, root_dir_slash

import logging

# Create a root logger
# Remove any default handlers if present
for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)

# Create a root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

# Add the handler only if no handlers are present
if not logger.hasHandlers():
    logger.addHandler(handler)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def parse_to_slice(s: str) -> slice:
    if s is None:
        return slice(0, None)
    else:
        start, stop, *step = s.split(":")
        if len(step) > 0:
            step = int(step[0])
        else:
            step = None
        return slice(int(start), int(stop), step)


def main(args: Optional[argparse.Namespace] = None):

    # Parse args
    if args is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        # make argparser
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--config",
            type=str,
            default=root_dir_slash("configs/default.yml"),
            help="path to config file",
        )
        parser.add_argument(
            "--dataset_path",
            type=str,
            default=root_dir_slash("data/final_dataset_with_metadata.json"),
        )
        parser.add_argument(
            "--output_path",
            type=str,
            default=root_dir_slash(f"runs/{timestamp}"),
        )
        parser.add_argument(
            "--query_range",
            type=str,
            default=None,
        )
        parser.add_argument(
            "--embedding_manager_path",
            type=str,
            default=root_dir_slash("data/final_dataset_embeddings.db"),
        )
        args = parser.parse_args()
    config = SimulationConfig(
        **yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    ).dump(path=Path(args.output_path) / "config.json")

    # Set the seed
    set_seed(config.rng_seed)
    logging.info(f"Seed set to {config.rng_seed}.")

    # Set the LLM and embedding names
    global_embedding_manager(init_from_path=args.embedding_manager_path)
    logging.info(
        f"Initialized embedding manager from path {args.embedding_manager_path}"
    )
    default_llm_name(set_to=config.llm_name)
    logging.info(f"Using LLM: {config.llm_name}")
    default_embedding_name(set_to=config.embedding_name)
    logging.info(f"Using embedding: {config.embedding_name}")

    # Load the dataset
    results = load(
        path=args.dataset_path,
        config=config,
    )
    results["buyers"] = results["buyers"][parse_to_slice(args.query_range)]
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
    bazaar = BazaarSimulator(
        bulletin_board=bulletin_board,
        buyer_principals=buyer_principals,
        vendor_principals=vendor_principals,
        seed=config.rng_seed,
        buyer_agent_kwargs=config.buyer_agent_kwargs,
        vendor_agent_kwargs=config.vendor_agent_kwargs,
    )

    # Run it for a week
    bazaar.run(168, print_callback=logging.info)

    # Print the answer
    for buyer in results["buyers"]:
        print("-" * 80)
        print("Question: ", buyer.query.text)
        try:
            print("Answer: ", buyer.answer.text)
            for block_idx, block in enumerate(buyer.answer.blocks):
                print(f"[Ref {block_idx + 1}] ", block.document_title)
        except Exception:
            print("No answer found.")
    dump_dict(
        bazaar.evaluation_summary(),
        str(Path(args.output_path) / "bazaar_summary.json"),
    )
    print("Done.")


if __name__ == "__main__":
    main()
