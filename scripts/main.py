from pathlib import Path

import yaml
import argparse
import numpy as np
import random

from coolname import generate_slug
import git

from bazaar.py_utils import dump_dict
from bazaar.sim_builder import load, SimulationConfig
from bazaar.simulator import BazaarSimulator


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def root_dir_slash(path: str) -> str:
    # Get the root dir of the repo where this file lives
    repo_root = git.Repo(__file__, search_parent_directories=True).working_tree_dir
    path = Path(repo_root) / path
    if not path.exists():
        path.mkdir(exist_ok=True, parents=True)
    return str(path)


def main():
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
        default=root_dir_slash("data/dataset_step_1.json"),
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=root_dir_slash(f"runs/{generate_slug(2)}"),
    )
    args = parser.parse_args()
    config = SimulationConfig(
        **yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    )

    # Set the seed
    set_seed(config.rng_seed)
    # Load the dataset
    results = load(
        path=args.dataset_path,
        config=config,
    )
    # FIXME: This is a temporary check
    results["buyers"] = results["buyers"][50:51]
    # Make a buyer agent for each principal
    buyer_principals = results["buyers"]
    vendor_principals = results["institutions"] + results["authors"]
    bulletin_board = results["bulletin_board"]
    bazaar = BazaarSimulator(
        bulletin_board=bulletin_board,
        buyer_principals=buyer_principals,
        vendor_principals=vendor_principals,
        seed=config.rng_seed,
        buyer_agent_kwargs=config.buyer_agent_kwargs,
    )
    # Run it for a week
    bazaar.run(168)

    # Print the answer
    for buyer in results["buyers"]:
        print("-" * 80)
        print("Question: ", buyer.query.text)
        if buyer.answer.success:
            print("Answer: ", buyer.answer.text)
        else:
            print("No answer found.")

    dump_dict(
        bazaar.evaluation_summary(),
        str(Path(args.output_path) / "bazaar_summary.json"),
    )
    print("Done.")


if __name__ == "__main__":
    main()
