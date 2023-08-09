from typing import List, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from bazaar.schema import Block, Query
from dataclasses import dataclass


@dataclass
class RetrievalOutput:
    query: Query
    blocks: List[Block]
    scores: List[float]


def retrieve_blocks(
    queries: List[Query],
    blocks: List[Block],
    use_hyde: bool = True,
    top_k: Optional[int] = None,
    score_threshold: Optional[float] = None,
) -> List[RetrievalOutput]:

    if len(blocks) == 0:
        # No blocks to retrieve
        return []

    if use_hyde:
        query_embeddings = [query.hyde_embedding for query in queries]
    else:
        query_embeddings = [query.text_embedding for query in queries]

    # query_embeddings.shape = (num_queries, embedding_dim)
    query_embeddings = np.array(query_embeddings)
    # block_embeddings.shape = (num_blocks, embedding_dim)
    block_embeddings = np.array([block.embedding for block in blocks])
    # cosine_similarities.shape = (num_queries, num_blocks)
    cosine_similarities = cosine_similarity(query_embeddings, block_embeddings)
    # Now gather the blocks for each query
    retrieval_outputs = []
    block_indices = np.arange(len(blocks))

    for query_idx, query in enumerate(queries):
        if score_threshold is not None:
            mask = cosine_similarities[query_idx] > score_threshold
            block_indices_for_query = block_indices[mask]
            cosine_similarities_for_query = cosine_similarities[query_idx, mask]
        else:
            block_indices_for_query = block_indices
            cosine_similarities_for_query = cosine_similarities[query_idx]
        # Sort the blocks by cosine similarity
        sorted_indices = np.argsort(cosine_similarities_for_query)[::-1]
        sorted_scores = cosine_similarities_for_query[sorted_indices]
        sorted_block_indices = block_indices_for_query[sorted_indices]
        sorted_blocks = [blocks[idx] for idx in sorted_block_indices]
        if top_k is not None:
            sorted_blocks = sorted_blocks[:top_k]
            sorted_scores = sorted_scores[:top_k]
        output = RetrievalOutput(
            query=query, blocks=sorted_blocks, scores=sorted_scores.tolist()
        )
        retrieval_outputs.append(output)

    return retrieval_outputs
