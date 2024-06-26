import hashlib
from collections import defaultdict
from typing import List, Optional, Dict

import numpy as np
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity

from bazaar.schema import Block, Query
from dataclasses import dataclass, field


@dataclass
class RetrievalOutput:
    query: Query
    blocks: List[Block]
    scores: List[float]


@dataclass
class ScoredBlock:
    block: Block
    # For each filter type, we store the score of this block for all queries
    scores_at_filters: Dict[Query, Dict[str, float]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    score_weights_at_filters: Dict[Query, Dict[str, float]] = field(
        default_factory=lambda: defaultdict(dict)
    )

    def record_score(
        self, filter: "Filter", query: Query, score: float, weight: float = 1.0
    ) -> "ScoredBlock":
        self.scores_at_filters[query][filter.__class__.__name__] = score
        self.score_weights_at_filters[query][filter.__class__.__name__] = weight
        return self

    def get_final_scores(self) -> Dict[Query, float]:
        final_scores = {}
        for query, scores_for_query in self.scores_at_filters.items():
            scores_for_query = list(scores_for_query.values())
            score_weights_for_query = list(
                self.score_weights_at_filters[query].values()
            )

            # No scores
            if len(scores_for_query) == 0:
                final_scores[query] = None
                continue

            # Single score
            if len(scores_for_query) == 1:
                final_scores[query] = scores_for_query[0]
                continue

            # Weighted average for multiple scores
            final_scores[query] = np.average(
                scores_for_query, weights=score_weights_for_query
            )
        return final_scores

    def as_retrieval_outputs(self) -> List["RetrievalOutput"]:
        # If there are multiple queries, it's possible we'll have multiple retrieval outputs
        outputs = []
        final_scores = self.get_final_scores()
        for query in final_scores.keys():
            outputs.append(
                RetrievalOutput(
                    query=query, blocks=[self.block], scores=[final_scores[query]]
                )
            )
        return outputs


class Filter:
    def apply(
        self, queries: List[Query], scored_blocks: List[ScoredBlock]
    ) -> List[ScoredBlock]:
        raise NotImplementedError()

    def __call__(
        self,
        queries: List[Query],
        blocks: Optional[List[Block]] = None,
        scored_blocks: Optional[List[ScoredBlock]] = None,
    ) -> List[RetrievalOutput]:
        if blocks is None:
            assert scored_blocks is not None
        if scored_blocks is None:
            assert blocks is not None
            scored_blocks = [ScoredBlock(block) for block in blocks]
        return [
            retrieval_output
            for scored_block in self.apply(queries, scored_blocks)
            for retrieval_output in scored_block.as_retrieval_outputs()
        ]


class FilterChain(Filter):
    def __init__(self, filters: List[Filter]):
        self.filters = filters

    def apply(
        self, queries: List[Query], scored_blocks: List[ScoredBlock]
    ) -> List[ScoredBlock]:
        for filter in self.filters:
            scored_blocks = filter.apply(queries, scored_blocks)
        return scored_blocks


class BM25(Filter):
    def __init__(
        self,
        use_query_keywords: bool = True,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        use_caching: bool = True,
        weight: float = 1.0,
    ):
        self.use_query_keywords = use_query_keywords
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.use_caching = use_caching
        self.weight = weight
        # Privates
        self._cached_bm25 = {}

    def _create_bm25(self, scored_blocks: List["ScoredBlock"]) -> BM25Okapi:
        # Tokenize the corpus
        tokenized_corpus = [
            word_tokenize(block.block.content) for block in scored_blocks
        ]
        return BM25Okapi(tokenized_corpus)

    def _get_bm25(self, scored_blocks: List["ScoredBlock"]) -> BM25Okapi:
        if not self.use_caching:
            return self._create_bm25(scored_blocks)
        hash_value = hashlib.sha256(
            " +++ ".join([block.block.content for block in scored_blocks]).encode()
        ).hexdigest()
        if hash_value not in self._cached_bm25:
            self._cached_bm25[hash_value] = self._create_bm25(scored_blocks)
        return self._cached_bm25[hash_value]

    def apply(
        self, queries: List[Query], scored_blocks: List[ScoredBlock]
    ) -> List[ScoredBlock]:
        if len(scored_blocks) == 0:
            # No blocks to retrieve
            return []
        block_indices = np.arange(len(scored_blocks))
        # Get the BM25 model for this set of blocks
        bm25 = self._get_bm25(scored_blocks)
        # Prepare output
        output_blocks = []
        for query in queries:
            # Get the scores
            if self.use_query_keywords:
                query_tokens = [
                    sub_keyword
                    for keyword in query.keywords
                    for sub_keyword in word_tokenize(keyword)
                ]
            else:
                query_tokens = word_tokenize(query.text)
            scores = bm25.get_scores(query_tokens)
            # Cutoff the scores
            if self.score_threshold is not None:
                mask = scores > self.score_threshold
                block_indices_for_query = block_indices[mask]
                scores_for_query = scores[mask]
            else:
                block_indices_for_query = block_indices
                scores_for_query = scores
            # Sort the blocks by score
            sorted_indices = np.argsort(scores_for_query)[::-1]
            sorted_scores = scores_for_query[sorted_indices]
            sorted_block_indices = block_indices_for_query[sorted_indices]
            sorted_blocks = [scored_blocks[idx] for idx in sorted_block_indices]
            if self.top_k is not None:
                sorted_blocks = sorted_blocks[: self.top_k]
                sorted_scores = sorted_scores[: self.top_k]
            # Record the scores
            for block, score in zip(sorted_blocks, sorted_scores):
                block.record_score(self, query, score, weight=self.weight)
                if block not in output_blocks:
                    output_blocks.append(block)
        # Done
        return output_blocks


class MIPS(Filter):
    def __init__(
        self,
        use_hyde: bool = True,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        half_sphere: bool = True,
        embed_block_content_with_metadata: bool = False,
        weight: float = 1.0,
    ):
        self.use_hyde = use_hyde
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.half_sphere = half_sphere
        self.embed_block_content_with_metadata = embed_block_content_with_metadata
        self.weight = weight

    def apply(
        self, queries: List[Query], scored_blocks: List[ScoredBlock]
    ) -> List[ScoredBlock]:
        if len(scored_blocks) == 0:
            # No blocks to retrieve
            return []

        if self.use_hyde:
            query_embeddings = [query.hyde_embedding for query in queries]
        else:
            query_embeddings = [query.text_embedding for query in queries]

        # query_embeddings.shape = (num_queries, embedding_dim)
        query_embeddings = np.array(query_embeddings)
        # block_embeddings.shape = (num_blocks, embedding_dim)
        block_embeddings = np.array(
            [
                scored_block.block.get_content_embedding(
                    embed_metadata=self.embed_block_content_with_metadata
                )
                for scored_block in scored_blocks
            ]
        )
        # cosine_similarities.shape = (num_queries, num_blocks)
        cosine_similarities = cosine_similarity(query_embeddings, block_embeddings)
        # Form the scores
        if self.half_sphere:
            # Clamp the scores to [0, 1]
            scores = np.clip(cosine_similarities, 0, 1)
        else:
            scores = (cosine_similarities + 1) / 2
        # Now gather the blocks for each query
        output_blocks = []
        block_indices = np.arange(len(scored_blocks))

        for query_idx, query in enumerate(queries):
            if self.score_threshold is not None:
                mask = scores[query_idx] > self.score_threshold
                block_indices_for_query = block_indices[mask]
                scores_for_query = scores[query_idx, mask]
            else:
                block_indices_for_query = block_indices
                scores_for_query = scores[query_idx]
            # Sort the blocks by cosine similarity
            sorted_indices = np.argsort(scores_for_query)[::-1]
            sorted_scores = scores_for_query[sorted_indices]
            sorted_block_indices = block_indices_for_query[sorted_indices]
            sorted_blocks = [scored_blocks[idx] for idx in sorted_block_indices]
            if self.top_k is not None:
                sorted_blocks = sorted_blocks[: self.top_k]
                sorted_scores = sorted_scores[: self.top_k]
            # Record the scores
            for block, score in zip(sorted_blocks, sorted_scores):
                block.record_score(self, query, score, weight=self.weight)
                if block not in output_blocks:
                    output_blocks.append(block)
        # Done
        return output_blocks


def build_retriever(
    # BM25
    filter_with_bm25: bool = True,
    use_query_keywords: bool = True,
    bm25_top_k: Optional[int] = None,
    bm25_score_threshold: Optional[float] = None,
    bm25_use_caching: bool = True,
    bm25_weight: float = 1.0,
    # MIPS
    filter_with_mips: bool = True,
    mips_use_hyde: bool = True,
    mips_top_k: Optional[int] = None,
    mips_score_threshold: Optional[float] = None,
    mips_half_sphere: bool = True,
    mips_embed_block_content_with_metadata: bool = False,
    mips_weight: float = 1.0,
) -> "Filter":
    filters = []
    if filter_with_bm25:
        filters.append(
            BM25(
                use_query_keywords=use_query_keywords,
                top_k=bm25_top_k,
                score_threshold=bm25_score_threshold,
                use_caching=bm25_use_caching,
                weight=bm25_weight,
            )
        )
    if filter_with_mips:
        filters.append(
            MIPS(
                use_hyde=mips_use_hyde,
                top_k=mips_top_k,
                score_threshold=mips_score_threshold,
                half_sphere=mips_half_sphere,
                embed_block_content_with_metadata=mips_embed_block_content_with_metadata,
                weight=mips_weight,
            )
        )
    return FilterChain(filters)
