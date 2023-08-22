from pathlib import Path
from typing import Optional

import yaml
import argparse
import numpy as np
import random
import datetime

from bazaar.sim_builder import load, SimulationConfig
from bazaar.simulator import BazaarSimulator
from bazaar.py_utils import dump_dict, root_dir_slash


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def parse_to_slice(s: str) -> slice:
    if s is None:
        return slice(0, None)
    else:
        start, stop = s.split(":")
        return slice(int(start), int(stop))


def main(args: Optional[argparse.Namespace] = None):
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
            default=root_dir_slash("data/machine-learning/dataset_step_1.pkl"),
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
        args = parser.parse_args()
    config = SimulationConfig(
        **yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    ).dump(path=Path(args.output_path) / "config.json")

    # Set the seed
    set_seed(config.rng_seed)
    # Load the dataset
    results = load(
        path=args.dataset_path,
        config=config,
    )
    results["buyers"] = results["buyers"][parse_to_slice(args.query_range)]
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
        vendor_agent_kwargs=config.vendor_agent_kwargs,
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
