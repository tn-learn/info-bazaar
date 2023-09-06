from dataclasses import dataclass
from enum import auto, IntEnum
from typing import Optional, List, Tuple

from networkx import DiGraph

from bazaar.schema import Query, Answer, Quote


class AnswerStatus(IntEnum):
    UNANSWERED = auto()
    PENDING_FOLLOW_UP = auto()
    ANSWERED = auto()


@dataclass
class QANode:
    query: Query
    answer: Optional[Answer] = None
    status: AnswerStatus = AnswerStatus.UNANSWERED


class QueryManager:
    def __init__(self):
        self.question_graph = DiGraph()

    def add_query(self, query: Query, parent_query: Optional[Query] = None):
        pass

    def generate_follow_up_queries(
        self, quotes: List[Quote]
    ) -> List[Tuple[Query, Query]]:
        pass

    def answer_root(self, quotes: List[Quote]) -> Answer:
        """This generates the answer to the root question."""
        pass
