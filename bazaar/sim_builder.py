from dataclasses import dataclass
from typing import Tuple, List

import numpy as np

from bazaar.py_utils import dump_dict, PathType, load_dict, dataclass_from_dict
from bazaar.schema import (
    Block,
    Query,
    BuyerPrincipal,
    Institution,
    Author,
    BulletinBoard,
)


@dataclass
class SimulationConfig:
    rng_seed: int
    # Block price statistics
    author_block_price_mean: float
    author_block_price_sigma: float
    institution_block_price_mean: float
    institution_block_price_sigma: float
    # Fraction of blocks to keep
    author_fraction_of_private_blocks: float
    institution_num_blocks: int
    # Temporal parameters
    author_response_time_mean: float
    author_response_time_sigma: float
    # Buyer budget
    buyer_max_budget_mean: float
    buyer_max_budget_sigma: float
    # Buyer urgency
    buyer_urgency_min: int
    buyer_urgency_max: int
    # Vendor search params
    bulletin_board_retrieval_top_k: int
    bulletin_board_retrieval_score_threshold: float
    # Query creation times
    query_creation_time_start: int
    query_creation_time_end: int
    # Buyer agent kwargs
    buyer_agent_kwargs: dict
    vendor_agent_kwargs: dict
    # LLMs and embeddings
    llm_name: str
    embedding_name: str

    def dump(self, path: PathType) -> "SimulationConfig":
        dump_dict(self.__dict__, path)
        return self


def load(path: str, config: SimulationConfig):
    dataset = load_dict(path)
    rng = np.random.RandomState(config.rng_seed)
    buyers = build_buyers(dataset=dataset, config=config, rng=rng)
    authors, institutions = build_authors_and_institutions(
        dataset=dataset, config=config, rng=rng
    )
    bulletin_board = build_bulletin_board(config=config, rng=rng)
    return {
        "buyers": buyers,
        "authors": authors,
        "institutions": institutions,
        "bulletin_board": bulletin_board,
    }


def build_bulletin_board(config: SimulationConfig, rng: np.random.RandomState):
    return BulletinBoard(
        top_k=config.bulletin_board_retrieval_top_k,
        score_threshold=config.bulletin_board_retrieval_score_threshold,
    )


def build_buyers(
    dataset: dict, config: SimulationConfig, rng: np.random.RandomState
) -> List[BuyerPrincipal]:
    buyers = []
    for arxiv_id, data in dataset.items():
        for block in data["blocks"]:
            block = dataclass_from_dict(Block, block)
            for idx, question in enumerate(block.questions):
                max_budget = np.random.lognormal(
                    mean=config.buyer_max_budget_mean,
                    sigma=config.buyer_max_budget_sigma,
                    size=1,
                )
                urgency = rng.randint(
                    low=config.buyer_urgency_min, high=config.buyer_urgency_max
                )
                created_at_time = rng.randint(
                    low=config.query_creation_time_start,
                    high=config.query_creation_time_end,
                )
                query = Query(
                    text=question,
                    max_budget=max_budget,
                    urgency=urgency,
                    created_at_time=created_at_time,
                    _gold_block_id=block.block_id,
                    _gold_block=block,
                )
                buyer = BuyerPrincipal(
                    name=f"buyer-{block.block_id}-{idx}",
                    query=query,
                )
                buyers.append(buyer)
    return buyers


def sample_block_prices(mean: float, sigma: float, blocks: List["Block"]):
    return np.random.lognormal(mean=mean, sigma=sigma, size=len(blocks))


def allocate_block_prices(entity, mean, sigma, rng):
    # Sample block prices from a log normal distribution
    block_prices = sample_block_prices(
        mean=mean,
        sigma=sigma,
        blocks=entity.public_blocks.values(),
    )
    for idx, block_id in enumerate(entity.public_blocks.keys()):
        entity.block_prices[block_id] = block_prices[idx]
    return entity


def shuffle_blocks(entity, fraction_to_move, rng):
    num_blocks = len(entity.public_blocks)
    num_blocks_to_move = round(num_blocks * fraction_to_move)
    block_keys = list(entity.public_blocks.keys())
    # allocate blocks to private and public
    rng.shuffle(block_keys)
    for i, key in enumerate(block_keys):
        if i == num_blocks_to_move:
            break
        entity.private_blocks[key] = entity.public_blocks[key]
        del entity.public_blocks[key]
    return entity


def distribute_blocks(authors, institutions, config, rng):
    for author in authors.values():
        author = allocate_block_prices(
            author, config.author_block_price_mean, config.author_block_price_sigma, rng
        )
        author = shuffle_blocks(author, config.author_fraction_of_private_blocks, rng)
        # Sample response time from a log normal distribution
        author.mean_response_time = rng.lognormal(
            mean=config.author_response_time_mean,
            sigma=config.author_response_time_sigma,
            size=1,
        )

    for institution in institutions.values():
        institution = allocate_block_prices(
            institution,
            config.author_block_price_mean,
            config.author_block_price_sigma,
            rng,
        )
        institution = shuffle_blocks(
            institution, config.author_fraction_of_private_blocks, rng
        )
        institution.mean_response_time = rng.lognormal(
            mean=config.author_response_time_mean,
            sigma=config.author_response_time_sigma,
            size=1,
        )
    return authors, institutions


def build_authors_and_institutions(
    dataset: dict, config: SimulationConfig, rng: np.random.RandomState
) -> Tuple[List[Author], List[Institution]]:
    # Build authors and institutions
    authors = {}
    institutions = {}

    for arxiv_id in dataset:
        data = dataset[arxiv_id]["metadata"]
        for authorship in data["authorships"]:
            for institution in authorship["institutions"]:
                if institution["id"] not in institutions:
                    institution["name"] = institution["display_name"]
                    del institution["display_name"]
                    institutions[institution["id"]] = dataclass_from_dict(
                        Institution, institution
                    )
            author = authorship["author"]
            author["name"] = author["display_name"]
            del author["display_name"]
            author = dataclass_from_dict(Author, author)
            if author not in authors:
                authors[author.id] = author

    # Assign blocks to institutions and authors
    for arxiv_id in dataset:
        for block in dataset[arxiv_id]["blocks"]:
            block = dataclass_from_dict(Block, block)
            for authorship in dataset[arxiv_id]["metadata"]["authorships"]:
                if authorship.get("author_position") == "first":
                    institution = authorship.get("institutions", [])
                    if not institution:
                        continue
                    institutions[institution[0]["id"]].public_blocks[
                        block.block_id
                    ] = block
                    authors[authorship["author"]["id"]].public_blocks[
                        block.block_id
                    ] = block

    # Distribute blocks in authors and institutions
    authors, institutions = distribute_blocks(authors, institutions, config, rng)
    # Done
    return list(authors.values()), list(institutions.values())
