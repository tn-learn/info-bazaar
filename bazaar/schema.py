import hashlib
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, List, Union, Optional, Dict, Any, Tuple

import tiktoken

from bazaar.lem_utils import default_llm_name, default_embedding_name
from bazaar.py_utils import ensure_number

if TYPE_CHECKING:
    from bazaar.simulator import BuyerAgent, VendorAgent


def repr_factory(obj, **attributes):
    class_name = obj.__class__.__name__
    attribute_strings = []
    for attribute_name, attribute_value in attributes.items():
        attribute_strings.append(f"{attribute_name}={attribute_value}")
    return f"{class_name}({', '.join(attribute_strings)})"


@dataclass
class Query:
    text: str
    max_budget: int
    created_at_time: int
    issued_by: Optional["BuyerAgent"] = None
    urgency: Optional[int] = None
    required_by_time: Optional[int] = None
    processor_model: Optional[str] = None
    embedding_model: Optional[str] = None
    query_type: Optional[str] = None
    # Containers for hyde and text
    _text_embedding: Optional[List[float]] = None
    _hyde_text: Optional[str] = None
    _hyde_embedding: Optional[List[float]] = None
    _keywords: Optional[List[str]] = None
    _gold_block_id: Optional[str] = None
    _gold_block: Optional["Block"] = None

    def __post_init__(self):
        self.max_budget = ensure_number(self.max_budget)
        self.created_at_time = ensure_number(self.created_at_time)
        self.urgency = ensure_number(self.urgency, allow_none=True)
        self.required_by_time = ensure_number(self.required_by_time, allow_none=True)
        if self.urgency is not None:
            assert self.required_by_time is None
            self.required_by_time = self.created_at_time + self.urgency
        elif self.required_by_time is not None:
            assert self.urgency is None
            self.urgency = self.required_by_time - self.created_at_time
        else:
            raise ValueError("Must specify either urgency or required_by_time.")
        if self.processor_model is None:
            self.processor_model = default_llm_name()
        if self.embedding_model is None:
            self.embedding_model = default_embedding_name()

    def time_till_deadline(self, now):
        time_remaining = self.required_by_time - now
        return time_remaining

    def deadline_passed(self, now):
        return self.required_by_time < now

    def register_issuer(self, issuer: "BuyerAgent") -> "Query":
        assert self.issued_by is None, "Cannot overwrite issuer."
        self.issued_by = issuer
        return self

    def ensure_issued_by(self, issuer: "BuyerAgent") -> "Query":
        if self.issued_by is None:
            self.register_issuer(issuer)
        else:
            assert self.issued_by is issuer
        return self

    @property
    def hyde_text(self):
        from bazaar.lem_utils import generate_hyde_passage

        if self._hyde_text is None:
            self._hyde_text = generate_hyde_passage(
                self.text, model=self.processor_model
            )
        return self._hyde_text

    @property
    def hyde_embedding(self):
        from bazaar.lem_utils import generate_embedding

        if self._hyde_embedding is None:
            self._hyde_embedding = generate_embedding(
                self.hyde_text, model=self.embedding_model
            )
        return self._hyde_embedding

    @property
    def text_embedding(self):
        from bazaar.lem_utils import generate_embedding

        if self._text_embedding is None:
            self._text_embedding = generate_embedding(
                self.text, model=self.embedding_model, as_query=True
            )
        return self._text_embedding

    @property
    def keywords(self):
        from bazaar.lem_utils import generate_keywords

        if self._keywords is None:
            self._keywords = generate_keywords(self.text, num_keywords=3)
        return self._keywords

    def get_content_prehash(self):
        return (
            self.text,
            ensure_number(self.max_budget),
            ensure_number(self.created_at_time),
            ensure_number(self.issued_by.unique_id),
            ensure_number(self.urgency),
            ensure_number(self.required_by_time),
            self.processor_model,
            self.embedding_model,
        )

    def __hash__(self):
        return hash(self.get_content_prehash())

    def compare_content(self, other: "Query") -> bool:
        return self.get_content_prehash() == other.get_content_prehash()

    def __repr__(self):
        return repr_factory(
            self,
            text=self.text,
            max_budget=self.max_budget,
            created_at_time=self.created_at_time,
            urgency=self.urgency,
            required_by_time=self.required_by_time,
            issued_by=(
                self.issued_by.unique_id if self.issued_by is not None else None
            ),
        )

    def evaluation_summary(self) -> Dict[str, Any]:
        return dict(
            text=self.text,
            max_budget=float(self.max_budget),
            created_at_time=self.created_at_time,
            urgency=self.urgency,
            required_by_time=self.required_by_time,
            issued_by=(
                self.issued_by.unique_id if self.issued_by is not None else None
            ),
            gold_block=(
                self._gold_block.evaluation_summary()
                if self._gold_block is not None
                else None
            ),
            gold_block_id=self._gold_block_id,
            hyde_text=self._hyde_text,
            keywords=self._keywords,
        )


class QuoteStatus(Enum):
    NOT_ISSUED = "not_issued"
    WAITING = "waiting"
    PENDING = "pending"
    CLAIMED = "claimed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


@dataclass
class Quote:
    query: Query
    price: float
    relevance_scores: List[float]
    created_at_time: int
    issued_by: "VendorAgent"
    answer_blocks: List["Block"] = field(default_factory=list)
    eta: Union[int, None] = None
    quote_status: QuoteStatus = QuoteStatus.NOT_ISSUED
    quote_progression: int = 0

    def claim_quote(self) -> "Quote":
        self.quote_status = QuoteStatus.CLAIMED
        return self

    def accept_quote(self) -> "Quote":
        self.quote_status = QuoteStatus.ACCEPTED
        return self

    def reject_quote(self) -> "Quote":
        self.quote_status = QuoteStatus.REJECTED
        return self

    def issue(self) -> "Quote":
        self.quote_status = QuoteStatus.PENDING
        return self

    def progress_quote(self) -> "Quote":
        self.quote_progression += 1
        return self

    def __repr__(self):
        return repr_factory(
            self,
            query=self.query,
            price=self.price,
            answer_blocks=self.answer_blocks,
            relevance_scores=self.relevance_scores,
            created_at_time=self.created_at_time,
            issued_by=(
                self.issued_by.unique_id if self.issued_by is not None else None
            ),
            eta=self.eta,
            quote_status=self.quote_status,
            quote_progression=self.quote_progression,
        )

    def evaluation_summary(self) -> Dict[str, Any]:
        return dict(
            query=self.query.evaluation_summary(),
            price=float(self.price),
            answer_blocks=[block.evaluation_summary() for block in self.answer_blocks],
            relevance_scores=[float(s) for s in self.relevance_scores],
            created_at_time=self.created_at_time,
            issued_by=(
                self.issued_by.unique_id if self.issued_by is not None else None
            ),
            eta=self.eta,
            quote_status=str(self.quote_status),
            quote_progression=self.quote_progression,
        )

    def get_content_prehash(self):
        return (
            self.query.get_content_prehash(),
            tuple(block.get_content_prehash() for block in self.answer_blocks),
            self.created_at_time,
            self.issued_by.unique_id,
            self.eta,
        )

    def __hash__(self):
        return hash(self.get_content_prehash())

    def compare_block_content(self, other: "Quote") -> bool:
        if len(self.answer_blocks) != len(other.answer_blocks):
            return False
        return all(
            [
                block1.compare_content(block2)
                for block1, block2 in zip(self.answer_blocks, other.answer_blocks)
            ]
        )

    def get_block_content_hash(self) -> str:
        return "+".join([block.get_block_content_hash() for block in self.answer_blocks])


@dataclass
class BulletinBoard:
    queries: List[Query] = field(default_factory=list)
    top_k: int = 1
    score_threshold: float = 0.7

    def post(self, query: Query):
        assert query.issued_by is not None, "Query must have an issuer."
        self.queries.append(query)

    def maintain(self, now: int) -> "BulletinBoard":
        # Remove queries that have passed their deadline
        self.queries = [q for q in self.queries if not q.deadline_passed(now)]
        return self

    def new_queries_since(self, time):
        return [query for query in self.queries if query.created_at_time >= time]

    def mark_query_as_processed(self, query: Query) -> "BulletinBoard":
        while query in self.queries:
            self.queries.remove(query)
        return self


@dataclass
class Principal:
    name: str

    def evaluation_summary(self) -> Dict[str, Any]:
        return dict(name=self.name)


@dataclass
class Answer:
    success: bool
    text: Optional[str] = None
    blocks: List["Block"] = field(default_factory=list)
    relevance_scores: List[float] = field(default_factory=list)

    def __post_init__(self):
        assert len(self.blocks) == len(
            self.relevance_scores
        ), "Must have same number of blocks and relevance scores."

    def get_content_prehash(self):
        return (
            self.success,
            self.text,
            tuple([b.get_content_prehash() for b in self.blocks]),
            tuple(self.relevance_scores),
        )

    def evaluation_summary(self) -> Dict[str, Any]:
        return dict(
            success=self.success,
            text=self.text,
            blocks=[block.evaluation_summary() for block in self.blocks],
            relevance_scores=[float(s) for s in self.relevance_scores],
        )


@dataclass
class BuyerPrincipal(Principal):
    query: Optional[Query] = None
    answer: Optional[Answer] = None

    def bind_query(self, query: Query):
        assert self.query is None, "Cannot overwrite query."
        self.query = query

    @property
    def budget(self):
        return self.query.max_budget

    def time_left(self, now) -> Union[float, int]:
        if self.query.required_by_time is None:
            return float("inf")
        return self.query.required_by_time - now

    def submit_final_response(self, answer: Answer) -> "BuyerPrincipal":
        self.answer = answer
        return self

    def evaluation_summary(self) -> Dict[str, Any]:
        super_summary = super().evaluation_summary()
        answer = self.answer.evaluation_summary() if self.answer is not None else None
        return dict(
            **super_summary,
            query=self.query.evaluation_summary(),
            answer=answer,
        )


@dataclass
class Block:
    document_id: str
    document_title: str
    section_title: str
    publication_date: str
    token_start: int
    token_end: int
    content: str
    questions: List[str] = field(default_factory=list)

    def __post_init__(self):
        self._content_embedding_cache: Dict[Optional[str], List[float]] = {}

    def __repr__(self):
        return repr_factory(
            self,
            document_id=self.document_id,
            document_title=self.document_title,
            block_id=self.block_id,
        )

    def compare_content(self, other: "Block") -> bool:
        return self.get_block_content_prehash() == other.get_block_content_prehash()

    def get_block_content_prehash(self) -> Tuple[str, str, str, str, int, int, str]:
        return (
            self.document_id,
            self.document_title,
            self.section_title,
            self.publication_date,
            self.token_start,
            self.token_end,
            self.content,
        )

    def get_block_content_hash(self) -> str:
        return hashlib.sha256(
            str(self.get_block_content_prehash()).encode("utf-8")
        ).hexdigest()

    def evaluation_summary(self) -> Dict[str, Any]:
        return dict(block_id=self.block_id, content=self.content)

    @property
    def block_id(self) -> str:
        return f"{self.document_id}/{self.section_title}/{self.token_start}/{self.token_end}"

    @staticmethod
    def num_tokens_in_content(content: str, model_name: str = "gpt-3.5-turbo") -> int:
        tiktoken_enc = tiktoken.encoding_for_model(model_name)
        return len(tiktoken_enc.encode(content))

    @property
    def num_tokens(self) -> int:
        return self.num_tokens_in_content(self.content)

    def get_content_prehash(self):
        return (
            self.document_id,
            self.document_title,
            self.publication_date,
            self.block_id,
            self.content,
            tuple(self.questions),
        )

    def __hash__(self):
        return hash(self.get_content_prehash())

    @property
    def content_with_metadata(self) -> str:
        content = (
            f"Paper Title: {self.document_title}\n"
            f"Section Title: {self.section_title}\n"
            f"---\n"
            f"Content: {self.content}"
        )
        return content

    @property
    def metadata(self) -> str:
        content = (
            f"Paper Title: {self.document_title}\n"
            f"Section Title: {self.section_title}\n"
        )
        return content

    def get_content_embedding(
        self,
        model: Optional[str] = None,
        embed_metadata: bool = False,
        **embedding_kwargs,
    ) -> List[float]:
        if model is None:
            model = default_embedding_name()
        if model not in self._content_embedding_cache:
            from bazaar.lem_utils import generate_embedding

            self._content_embedding_cache[model] = generate_embedding(
                self.content_with_metadata if embed_metadata else self.content,
                model=model,
                **embedding_kwargs,
            )
        return self._content_embedding_cache[model]


@dataclass
class Institution(Principal):
    id: str
    ror: str
    country_code: str
    type: str
    public_blocks: Dict[str, Block] = field(default_factory=dict)
    private_blocks: Dict[str, Block] = field(default_factory=dict)
    block_prices: Dict[str, float] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.id)

    @property
    def num_blocks_owned(self):
        return len(self.public_blocks) + len(self.private_blocks)

    def evaluation_summary(self) -> Dict[str, Any]:
        return dict(
            **super().evaluation_summary(),
            id=self.id,
            type=self.type,
            ror=self.ror,
            public_blocks=[
                block.evaluation_summary() for block in self.public_blocks.values()
            ],
            private_blocks=[
                block.evaluation_summary() for block in self.private_blocks.values()
            ],
            block_prices=self.block_prices,
        )


@dataclass
class Author(Principal):
    id: str
    orcid: str
    last_known_institution: Optional[Institution] = None
    related_concepts: Optional[List[str]] = None
    public_blocks: Dict[str, Block] = field(default_factory=dict)
    private_blocks: Dict[str, Block] = field(default_factory=dict)
    block_prices: Dict[str, float] = field(default_factory=dict)
    mean_response_time: Optional[int] = None

    def __hash__(self):
        return hash(self.id)

    @property
    def num_blocks_owned(self):
        return len(self.public_blocks) + len(self.private_blocks)

    def evaluation_summary(self) -> Dict[str, Any]:
        return dict(
            **super().evaluation_summary(),
            id=self.id,
            last_known_institution=(
                self.last_known_institution.get("id")
                if self.last_known_institution is not None
                else None
            ),
            related_concepts=self.related_concepts,
            public_blocks=[
                block.evaluation_summary() for block in self.public_blocks.values()
            ],
            private_blocks=[
                block.evaluation_summary() for block in self.private_blocks.values()
            ],
            block_prices={k: float(v) for k, v in self.block_prices.items()},
        )


class AgentStatus(Enum):
    ACTIVE = "active"
    TERMINATED = "terminated"
