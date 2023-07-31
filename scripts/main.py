from bazaar.sim_builder import load, SimulationConfig
from bazaar.simulator import BazaarSimulator


def main():

    config = SimulationConfig(
        rng_seed=0,
        author_block_price_mean=0,
        author_block_price_sigma=0,
        institution_block_price_mean=0,
        institution_block_price_sigma=0,
        author_fraction_of_private_blocks=0,
        institution_num_blocks=0,
        author_response_time_mean=0,
        author_response_time_sigma=0,
        buyer_max_budget_mean=0,
        buyer_max_budget_sigma=0,
        buyer_urgency_min=0,
        buyer_urgency_max=0,
        query_creation_time_start=0,
        query_creation_time_end=0,
    )

    principals = load(path="", config=config)
    # Make a buyer agent for each principal
    buyer_principals = principals["buyers"]
    vendor_principals = principals["institutions"] + principals["authors"]
    bazaar = BazaarSimulator(
        buyer_principals=buyer_principals, vendor_principals=vendor_principals
    )
    # Run it for a week
    bazaar.run(168)


if __name__ == "__main__":
    main()
