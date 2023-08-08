import yaml
import argparse
import numpy as np
import random
from bazaar.sim_builder import load, SimulationConfig
from bazaar.simulator import BazaarSimulator


def main():
    # make argparser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="/Users/martinweiss/PycharmProjects/tn-learn/info-bazaar/configs/default.yml",
        help="path to config file",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/dataset_step_1.json",
    )
    args = parser.parse_args()
    config = SimulationConfig(**yaml.load(open(args.config, "r"), Loader=yaml.FullLoader))
    np.random.seed(42)
    random.seed(42)
    results = load(path=args.dataset_path, config=config)
    # FIXME: This is a temporary check
    results["buyers"] = results["buyers"][0:1]
    # Make a buyer agent for each principal
    buyer_principals = results["buyers"]
    vendor_principals = results["institutions"] + results["authors"]
    bulletin_board = results["bulletin_board"]
    bazaar = BazaarSimulator(
        bulletin_board=bulletin_board, buyer_principals=buyer_principals, vendor_principals=vendor_principals
    )
    # Run it for a week
    bazaar.run(168)


if __name__ == "__main__":
    main()
