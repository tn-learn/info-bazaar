from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Any

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
    reranker_name: str

    def dump(self, path: PathType) -> "SimulationConfig":
        dump_dict(self.__dict__, path)
        return self


def build_buyer(
    question: str,
    gold_block_id: Optional[str],
    gold_block_text: Optional[str],
    buyer_name: str,
    buyer_max_budget_mean: float,
    buyer_max_budget_sigma: float,
    buyer_urgency_min: int,
    buyer_urgency_max: int,
    query_creation_time_start: int,
    query_creation_time_end: int,
    rng: np.random.RandomState,
):
    max_budget = np.random.lognormal(
        mean=buyer_max_budget_mean, sigma=buyer_max_budget_sigma, size=1,
    )
    urgency = rng.randint(low=buyer_urgency_min, high=buyer_urgency_max)
    created_at_time = rng.randint(
        low=query_creation_time_start, high=query_creation_time_end,
    )
    query = Query(
        text=question,
        max_budget=max_budget,
        urgency=urgency,
        created_at_time=created_at_time,
        _gold_block_id=gold_block_id,
        _gold_block=gold_block_text,
    )
    buyer = BuyerPrincipal(name=buyer_name, query=query,)
    return buyer


def parse_questions_from_dataset(dataset: dict) -> List[Dict[str, Any]]:
    questions = []
    for arxiv_id, data in dataset.items():
        for block in data["blocks"]:
            block = dataclass_from_dict(Block, block)
            for idx, question in enumerate(block.questions):
                buyer_name = f"buyer-{block.block_id}-{idx}"
                questions.append(
                    {
                        "buyer_name": buyer_name,
                        "question": question,
                        "gold_block_id": block.block_id,
                        "gold_block_text": block.content,
                    }
                )
    return questions


def build_buyers(
    questions: List[Dict[str, Any]],
    buyer_max_budget_mean: float,
    buyer_max_budget_sigma: float,
    buyer_urgency_min: int,
    buyer_urgency_max: int,
    query_creation_time_start: int,
    query_creation_time_end: int,
    rng: np.random.RandomState,
) -> List[BuyerPrincipal]:
    buyers = []
    for question in questions:
        buyer = build_buyer(
            question=question["question"],
            gold_block_id=question["gold_block_id"],
            gold_block_text=question["gold_block_text"],
            buyer_name=question["buyer_name"],
            buyer_max_budget_mean=buyer_max_budget_mean,
            buyer_max_budget_sigma=buyer_max_budget_sigma,
            buyer_urgency_min=buyer_urgency_min,
            buyer_urgency_max=buyer_urgency_max,
            query_creation_time_start=query_creation_time_start,
            query_creation_time_end=query_creation_time_end,
            rng=rng,
        )
        buyers.append(buyer)
    return buyers


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


def build_authors_and_institutions(
    dataset: dict,
    author_fraction_of_private_blocks: float,
    author_response_time_mean: float,
    author_response_time_sigma: float,
    rng: np.random.RandomState,
) -> Tuple[List[Author], List[Institution]]:
    authors = {}
    institutions = {}
    paper_prices = {}

    for arxiv_id in dataset:
        data = dataset[arxiv_id]["metadata"]
        for authorship in data["authorships"]:
            if authorship.get("author_position") == "first":
                paper_prices[arxiv_id] = authorship["author"].get(
                    "cited_by_count", 0
                ) / (authorship["author"].get("works_count", 1) + 1)

                for institution in authorship["institutions"]:
                    if (
                        institution is not None
                        and institution["id"] not in institutions
                    ):
                        institution["name"] = institution["display_name"]
                        del institution["display_name"]
                        institutions[institution["id"]] = dataclass_from_dict(
                            Institution, institution
                        )
                author = authorship["author"]
                author["name"] = author["display_name"]
                del author["display_name"]
                author = dataclass_from_dict(Author, author)
                if author.id not in authors:
                    authors[author.id] = author

    for arxiv_id in dataset:
        for block in dataset[arxiv_id]["blocks"]:
            block = dataclass_from_dict(Block, block)
            for authorship in dataset[arxiv_id]["metadata"]["authorships"]:
                if authorship.get("author_position") == "first":
                    institution = authorship.get("institutions", [])
                    if not institution or institution[0] is None:
                        continue
                    institutions[institution[0]["id"]].public_blocks[
                        block.block_id
                    ] = block
                    institutions[institution[0]["id"]].block_prices[
                        block.block_id
                    ] = paper_prices[arxiv_id]
                    authors[authorship["author"]["id"]].public_blocks[
                        block.block_id
                    ] = block
                    authors[authorship["author"]["id"]].block_prices[
                        block.block_id
                    ] = paper_prices[arxiv_id]

    for author in authors.values():
        author = shuffle_blocks(author, author_fraction_of_private_blocks, rng)
        author.mean_response_time = rng.lognormal(
            mean=author_response_time_mean, sigma=author_response_time_sigma, size=1,
        )

    for institution in institutions.values():
        institution = shuffle_blocks(
            institution, author_fraction_of_private_blocks, rng
        )
        institution.mean_response_time = rng.lognormal(
            mean=author_response_time_mean, sigma=author_response_time_sigma, size=1,
        )

    return list(authors.values()), list(institutions.values())
