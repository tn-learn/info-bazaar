import json
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np

from bazaar.schema import Block, Query, BuyerPrincipal, Institution, Author


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
    vendor_top_k: int
    vendor_score_threshold: int
    # Query creation times
    query_creation_time_start: int
    query_creation_time_end: int


def load(path: str, config: SimulationConfig):
    with open(path, "r") as f:
        dataset = json.load(f)

    rng = np.random.RandomState(config.rng_seed)
    buyers = build_buyers(dataset=dataset, config=config, rng=rng)
    authors, institutions = build_authors_and_institutions(
        dataset=dataset, config=config, rng=rng
    )
    return {
        "buyers": buyers,
        "authors": authors,
        "institutions": institutions,
    }


def build_buyers(
    dataset: dict, config: SimulationConfig, rng: np.random.RandomState
) -> List[BuyerPrincipal]:
    buyers = []
    for arxiv_id, data in dataset.items():
        for block in data["blocks"]:
            for idx, nugget in enumerate(block.get("nuggets", [])):
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
                    text=nugget["question"],
                    max_budget=max_budget,
                    urgency=urgency,
                    created_at_time=created_at_time,
                )
                buyer = BuyerPrincipal(
                    name=f"buyer-{block['block_id']}-{idx}",
                    query=query,
                )
                buyers.append(buyer)
    return buyers


def _configure_author_blocks_(
    author: Author, config: SimulationConfig, rng: np.random.RandomState
) -> Author:
    # Move a fraction of the blocks from public to private
    num_blocks = len(author.public_blocks)
    num_blocks_to_move = round(num_blocks * config.author_fraction_of_private_blocks)
    block_keys = list(author.public_blocks.keys())

    # Set author's preference on blocks
    block_prices = np.random.lognormal(
        mean=config.author_block_price_mean,
        sigma=config.author_block_price_sigma,
        size=num_blocks,
    )
    for idx, block_id in enumerate(author.public_blocks.keys()):
        author.block_prices[block_id] = block_prices[idx]

    # allocate blocks to private and public
    rng.shuffle(block_keys)
    for i, key in enumerate(block_keys):
        if i == num_blocks_to_move:
            break
        author.private_blocks[key] = author.public_blocks[key]
        del author.public_blocks[key]

    # Sample response time from a log normal distribution
    author.mean_response_time = rng.lognormal(
        mean=config.author_response_time_mean,
        sigma=config.author_response_time_sigma,
        size=1,
    )

    return author


def _configure_institution_blocks_(
    institution: Institution, config: SimulationConfig, rng: np.random.RandomState
) -> Institution:
    block_ids = list(institution.blocks.keys())
    # Delete the rest
    block_ids_to_delete = block_ids[config.institution_num_blocks :]
    for block_id in block_ids_to_delete:
        del institution.blocks[block_id]
    return institution


def build_authors_and_institutions(
    dataset: dict, config: SimulationConfig, rng: np.random.RandomState
) -> Tuple[List[Author], List[Institution]]:
    # Build authors and institutions
    authors = {}
    institutions = {}

    for arxiv_id, data in dataset.items():
        for authorship in data["authorships"]:
            for ins in authorship["institutions"]:
                if ins["id"] not in institutions:
                    institution = Institution(**ins, blocks={})
                    institutions[institution.id] = institution
            author = Author(**authorship["author"], blocks={})
            if author not in authors:
                authors[author.id] = author

    # Assign blocks to institutions and authors
    for arxiv_id, data in dataset.items():
        for block in data["blocks"]:
            if block.get("nuggets") is None:
                block["nuggets"] = []
            block_obj = Block(
                document_id=arxiv_id,
                document_title=data["title"],
                publication_date=data["publication_date"],
                block_id=block["block_id"],
                content=block["content"],
                num_tokens=block["num_tokens"],
                embedding=block["embedding"],
            )

            for authorship in data["authorships"]:
                for ins in authorship["institutions"]:
                    institutions[ins["id"]].blocks[block_obj.block_id] = block_obj
                authors[authorship["author"]["id"]].public_blocks[
                    block_obj.block_id
                ] = block_obj

    # Distribute blocks in authors and institutions
    for author in authors.values():
        _configure_author_blocks_(author, config, rng)
    for institution in institutions.values():
        _configure_institution_blocks_(institution, config, rng)
    # Done
    return list(authors.values()), list(institutions.values())
