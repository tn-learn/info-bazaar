from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, List, Union, Optional, Dict


if TYPE_CHECKING:
    from bazaar.simulator import BuyerAgent, VendorAgent


@dataclass
class Query:
    text: str
    max_budget: int
    created_at_time: int
    issued_by: "BuyerAgent"
    urgency: Optional[int] = None
    required_by_time: Optional[int] = None

    def __post_init__(self):
        if self.urgency is not None:
            assert self.required_by_time is None
            self.required_by_time = self.created_at_time + self.urgency

    def time_till_deadline(self, now):
        time_remaining = self.required_by_time - now
        return time_remaining

    def deadline_passed(self, now):
        return self.required_by_time < now


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
    most_similar_question_answered_in_block: str
    block_price: float
    created_at_time: int
    issued_by: "VendorAgent"
    answer_block: Optional[str] = None
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

    def bind_answer_block(self, answer_block: str, overwrite: bool = False) -> "Quote":
        if answer_block is None and overwrite:
            self.answer_block = answer_block
        else:
            raise ValueError("Cannot overwrite answer block.")
        return self


@dataclass
class BulletinBoard:
    queries: List[Query]

    def maintain(self, now: int) -> "BulletinBoard":
        # Remove queries that have passed their deadline
        self.queries = [q for q in self.queries if not q.deadline_passed(now)]
        return self


@dataclass
class Principal:
    name: str


@dataclass
class BuyerPrincipal(Principal):
    query: Query

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
    block_id: str
    content: str
    num_tokens: int
    embedding: List[float]
    nuggets: Optional[List[Nugget]]


@dataclass
class Institution(Principal):
    id: str
    display_name: str
    ror: str
    country_code: str
    type: str
    blocks: Dict[str, Block] = field(default_factory=dict)

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
    blocks: Dict[str, Block] = field(default_factory=dict)

    def __init__(self, name=None, *args, **kwargs):
        if not name:
            name = kwargs.get("display_name", "")
        super().__init__(name)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __hash__(self):
        return hash(self.id)


@dataclass
class Vendor:
    principal: Union[Author, Institution]
    block_price: List[int]
    observed_blocks: Optional[List[Block]] = field(default_factory=list)
    response_time_guarantee: int = 0


class AgentStatus(Enum):
    ACTIVE = "active"
    TERMINATED = "terminated"
