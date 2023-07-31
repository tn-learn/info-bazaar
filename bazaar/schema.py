from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, List, Union, Optional, Dict

from bazaar.lem_utils import generate_hyde_passage, generate_embedding

if TYPE_CHECKING:
    from bazaar.simulator import BuyerAgent, VendorAgent


@dataclass
class Query:
    text: str
    max_budget: int
    created_at_time: int
    issued_by: Optional["BuyerAgent"] = None
    urgency: Optional[int] = None
    required_by_time: Optional[int] = None
    processor_model: str = "gpt-3.5-turbo"
    embedding_model: str = "text-embedding-ada-002"
    # Containers for hyde and text
    _text_embedding: Optional[List[float]] = None
    _hyde_text: Optional[str] = None
    _hyde_embedding: Optional[List[float]] = None

    def __post_init__(self):
        if self.urgency is not None:
            assert self.required_by_time is None
            self.required_by_time = self.created_at_time + self.urgency

    def time_till_deadline(self, now):
        time_remaining = self.required_by_time - now
        return time_remaining

    def deadline_passed(self, now):
        return self.required_by_time < now

    def register_issuer(self, issuer: "BuyerAgent") -> "Query":
        assert self.issued_by is None, "Cannot overwrite issuer."
        self.issued_by = issuer
        return self

    @property
    def hyde_text(self):
        if self._hyde_text is None:
            self._hyde_text = generate_hyde_passage(
                self.text, model=self.processor_model
            )
        return self._hyde_text

    @property
    def hyde_embedding(self):
        if self._hyde_embedding is None:
            self._hyde_embedding = generate_embedding(
                self.hyde_text, model=self.embedding_model
            )
        return self._hyde_embedding

    @property
    def text_embedding(self):
        if self._text_embedding is None:
            self._text_embedding = generate_embedding(
                self.text, model=self.embedding_model
            )
        return self._text_embedding


class QuoteStatus(Enum):
    NOT_ISSUED = "not_issued"
    WAITING = "waiting"
    PENDING = "pending"
    AUTHORIZED = "authorized"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


@dataclass
class Quote:
    query: Query
    price: float
    created_at_time: int
    issued_by: "VendorAgent"
    answer_blocks: List["Block"] = field(default_factory=list)
    eta: Union[int, None] = None
    quote_status: QuoteStatus = QuoteStatus.NOT_ISSUED

    def authorize_quote(self) -> "Quote":
        self.quote_status = QuoteStatus.AUTHORIZED
        return self

    def accept_quote(self) -> "Quote":
        self.quote_status = QuoteStatus.ACCEPTED
        return self

    def reject_quote(self) -> "Quote":
        self.quote_status = QuoteStatus.REJECTED
        return self


@dataclass
class BulletinBoard:
    queries: List[Query] = field(default_factory=list)

    def post(self, query: Query):
        assert query.issued_by is not None, "Query must have an issuer."
        self.queries.append(query)

    def maintain(self, now: int) -> "BulletinBoard":
        # Remove queries that have passed their deadline
        self.queries = [q for q in self.queries if not q.deadline_passed(now)]
        return self

    def new_queries_since(self, time):
        return [query for query in self.queries if query.created_at_time >= time]


@dataclass
class Principal:
    name: str


@dataclass
class BuyerPrincipal(Principal):
    query: Optional[Query] = None

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


@dataclass(frozen=True)
class Nugget:
    question: str
    answer: str
    embedding: str


@dataclass(frozen=True)
class Block:
    document_id: str
    document_title: str
    publication_date: str
    block_id: str
    content: str
    num_tokens: int
    embedding: List[float]
    nuggets: List[Nugget] = field(default_factory=list)


@dataclass
class Institution(Principal):
    id: str
    display_name: str
    ror: str
    country_code: str
    type: str
    blocks: Dict[str, Block] = field(default_factory=dict)
    block_prices: Dict[str, float] = field(default_factory=dict)

    def __init__(self, name=None, *args, **kwargs):
        if not name:
            name = kwargs.get("display_name", "")
        super().__init__(name)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __hash__(self):
        return hash(self.id)


@dataclass
class Author(Principal):
    id: str
    display_name: str
    orcid: str
    last_known_institution: Optional[Institution] = None
    related_concepts: Optional[List[str]] = None
    public_blocks: Dict[str, Block] = field(default_factory=dict)
    private_blocks: Dict[str, Block] = field(default_factory=dict)
    block_prices: Dict[str, float] = field(default_factory=dict)
    mean_response_time: Optional[int] = None

    def __init__(self, name=None, *args, **kwargs):
        if not name:
            name = kwargs.get("display_name", "")
        super().__init__(name)
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def blocks(self):
        return self.public_blocks

    def __hash__(self):
        return hash(self.id)


@dataclass
class Vendor:
    principal: Union[Author, Institution]
    block_price: List[int]
    observed_blocks: Optional[List[Block]] = field(default_factory=list)


class AgentStatus(Enum):
    ACTIVE = "active"
    TERMINATED = "terminated"