import uuid
from dataclasses import dataclass, field
from enum import auto, Enum
from typing import TYPE_CHECKING, Optional, List, Tuple, Union, Any, Dict

import networkx as nx
from networkx import DiGraph

from bazaar.lem_utils import (
    synthesize_answer,
    select_follow_up_question,
    refine_answer,
    get_closed_book_answer,
)
from bazaar.schema import Query, Answer, Quote

if TYPE_CHECKING:
    from bazaar.simulator import BuyerAgent


class AnswerStatus(Enum):
    FAILED = "failed"
    UNANSWERED = "unanswered"
    PENDING_FOLLOW_UP = "pending_follow_up"
    ANSWERED = "answered"
    REFINED = "refined"


@dataclass
class QANode:
    query: Query
    answer: Optional[Answer] = None
    status: AnswerStatus = AnswerStatus.UNANSWERED
    is_followed_up: bool = False
    pre_refinement_answer: Optional[Answer] = None
    uuid: str = field(default_factory=lambda: str(uuid.uuid4()))

    def bind_answer(self, answer: Answer) -> "QANode":
        if not answer.success:
            self.status = AnswerStatus.FAILED
        else:
            self.status = AnswerStatus.ANSWERED
        self.answer = answer
        return self

    def bind_refined_answer(self, answer: Answer) -> "QANode":
        if answer.success:
            self.pre_refinement_answer = self.answer
            self.answer = answer
            self.status = AnswerStatus.REFINED
        return self

    def mark_as_failed(self) -> "QANode":
        self.status = AnswerStatus.FAILED
        return self

    def mark_as_answered(self) -> "QANode":
        self.status = AnswerStatus.ANSWERED
        return self

    def mark_as_pending_follow_up(self) -> "QANode":
        self.status = AnswerStatus.PENDING_FOLLOW_UP
        return self

    def mark_as_refined(self) -> "QANode":
        self.status = AnswerStatus.REFINED
        return self

    def mark_as_followed_up(self, is_followed_up: bool = True) -> "QANode":
        self.is_followed_up = True
        return self

    def evaluation_summary(self) -> Dict[str, Any]:
        return dict(
            query=self.query.evaluation_summary(),
            answer=(
                self.answer.evaluation_summary() if self.answer is not None else None
            ),
            pre_refinement_answer=(
                self.pre_refinement_answer.evaluation_summary()
                if self.pre_refinement_answer is not None
                else None
            ),
            status=str(self.status),
        )

    def get_content_prehash(self):
        return self.uuid

    def __hash__(self):
        return hash(self.get_content_prehash())

    def __eq__(self, other):
        if not isinstance(other, QANode):
            return False
        return self.get_content_prehash() == other.get_content_prehash()


class QueryManager:
    def __init__(
        self,
        agent: "BuyerAgent",
        answer_synthesis_model_name: str,
        follow_up_question_synthesis_model_name: str,
        max_query_depth: int = 3,
    ):
        # Private
        self._agent = agent
        self._allow_closed_book_answers = False
        # Public
        self.answer_synthesis_model_name = answer_synthesis_model_name
        self.follow_up_question_synthesis_model_name = (
            follow_up_question_synthesis_model_name
        )
        self.max_query_depth = max_query_depth
        self.question_graph = DiGraph()

    def find_node(self, query: Query) -> QANode:
        for node in self.question_graph.nodes:
            if node.query.compare_content(query):
                return node
        raise ValueError(f"Could not find node for query {query}")

    def mark_all_predecessors_with_status(
        self,
        node: QANode,
        status: AnswerStatus,
        include_self: bool = True,
    ) -> "QueryManager":
        if include_self:
            node.status = status

        for pred in self.question_graph.predecessors(node):
            self.mark_all_predecessors_with_status(pred, status)

        return self

    def mark_all_successors_with_status(
        self,
        node: QANode,
        status: AnswerStatus,
        include_self: bool = True,
    ) -> "QueryManager":
        if include_self:
            node.status = status

        for succ in self.question_graph.successors(node):
            self.mark_all_successors_with_status(succ, status)

        return self

    def get_node_depth(self, node: QANode) -> int:
        # Assumes the root node is the only node with in-degree of 0
        root_node = [n for n, d in self.question_graph.in_degree() if d == 0][0]
        lengths = nx.single_source_shortest_path_length(self.question_graph, root_node)
        return lengths[node]

    def add_query(
        self,
        query: Query,
        parent_query: Optional[Query] = None,
        if_depth_exceeds: str = "raise",
    ) -> "QueryManager":
        new_node = QANode(query=query)

        if parent_query is None:
            self.question_graph.add_node(new_node)
        else:
            parent_node = self.find_node(parent_query)
            potential_depth = self.get_node_depth(parent_node) + 1

            if potential_depth > self.max_query_depth:
                if if_depth_exceeds == "raise":
                    raise ValueError(
                        f"Adding the query {query} would exceed the maximum allowed depth of {self.max_query_depth}."
                    )
                elif if_depth_exceeds == "ignore":
                    return self
                else:
                    raise ValueError(
                        f"Invalid value for if_depth_exceeds: {if_depth_exceeds}"
                    )

            self.question_graph.add_edge(parent_node, new_node)

            # We mark all upstream queries in the graph as pending follow up
            self.mark_all_predecessors_with_status(
                parent_node, AnswerStatus.PENDING_FOLLOW_UP
            )

            # Make sure the graph is acyclic
            assert nx.is_directed_acyclic_graph(self.question_graph), (
                f"Graph is not acyclic after adding query {query}. "
                f"Parent query: {parent_query}."
            )
        return self

    def generate_follow_up_queries(
        self,
        quotes: List[Quote],
        commit: bool = True,
    ) -> List[Tuple[Query, Query]]:
        leaf_nodes = [
            node
            for node in self.question_graph.nodes
            if self.question_graph.out_degree(node) == 0  # noqa
        ]
        follow_up_queries = []
        for node in leaf_nodes:
            node: QANode
            # Skip if the node has been followed up already
            if node.is_followed_up:
                continue
            # If node has no answer, we try to generate one.
            if node.status in [AnswerStatus.UNANSWERED, AnswerStatus.FAILED]:
                self.synthesize_answer_for_query(node, quotes, commit=True)
            # If the depth of the node exceeds the maximum allowed depth, we
            # skip it.
            if self.get_node_depth(node) + 1 > self.max_query_depth:
                continue
            # Check if node has an answer. If it does, then we can generate a
            # follow-up question.
            if node.status == AnswerStatus.ANSWERED:
                follow_ups = self.synthesize_follow_up_queries(node)
                for follow_up_query in follow_ups:
                    follow_up_queries.append((follow_up_query, node.query))
                if commit:
                    node.mark_as_followed_up()
        if commit:
            for follow_up_query, parent_query in follow_up_queries:
                self.add_query(follow_up_query, parent_query)
        return follow_up_queries

    def synthesize_answer_for_query(
        self,
        node_or_query: Union[QANode, Query],
        quotes: List[Quote],
        commit: bool = True,
    ) -> Optional[Answer]:

        if isinstance(node_or_query, QANode):
            node = node_or_query
            query = node.query
        elif isinstance(node_or_query, Query):
            node = self.find_node(node_or_query)
            query = node_or_query
        else:
            raise TypeError(f"Expected a Query or QANode, got {type(node_or_query)}")

        # This method finds the quotes that are relevant to the query and then
        # synthesizes an answer from them.
        quotes_for_query = [
            quote for quote in quotes if quote.query.compare_content(query)
        ]
        if len(quotes_for_query) > 0:
            answer_text = synthesize_answer(
                query=query,
                quotes=quotes_for_query,
                model_name=self.answer_synthesis_model_name,
            )
            answer = Answer(
                success=True,
                text=answer_text,
                blocks=[
                    block for quote in quotes_for_query for block in quote.answer_blocks
                ],
                relevance_scores=[
                    score
                    for quote in quotes_for_query
                    for score in quote.relevance_scores
                ],
            )
            if commit:
                node.bind_answer(answer)
        elif self._allow_closed_book_answers:
            answer_text = get_closed_book_answer(
                question=query.text,
                model_name=self.answer_synthesis_model_name,
            )
            answer = Answer(
                success=True,
                text=answer_text,
                blocks=[],
                relevance_scores=[],
            )
            if commit:
                node.bind_answer(answer)
        else:
            answer = None
            if commit:
                node.mark_as_failed()
        return answer

    def refine_answer_for_query(
        self, query_or_node: Union[Query, QANode], quotes: List[Quote]
    ) -> Optional[Answer]:
        if isinstance(query_or_node, QANode):
            node = query_or_node
        elif isinstance(query_or_node, Query):
            node = self.find_node(query_or_node)
        else:
            raise TypeError(f"Expected a Query or QANode, got {type(query_or_node)}")

        # List to store the successor nodes with valid answers
        # (answered, refined, or synthesized)
        valid_successors = []

        # Check successors of the node
        for succ in self.question_graph.successors(node):
            succ: QANode

            # If successor is pending a follow-up, recursively refine
            if succ.status == AnswerStatus.PENDING_FOLLOW_UP:
                self.refine_answer_for_query(succ, quotes)

            # If successor is unanswered, try synthesizing an answer.
            # Note that we also try if it has failed, because we might be able to
            # synthesize an answer from the new quotes that might have come in the meantime.
            if succ.status in [AnswerStatus.UNANSWERED, AnswerStatus.FAILED]:
                answer = self.synthesize_answer_for_query(succ, quotes, commit=True)
                if answer is not None:
                    valid_successors.append(succ)
            elif succ.status in [AnswerStatus.ANSWERED, AnswerStatus.REFINED]:
                # If successor is answered or refined, add it to the list
                valid_successors.append(succ)

        # Now, use the apply_refinement function to update the answer of
        # the current node
        refined_answer = self.apply_refinement(node, valid_successors, commit=True)
        return refined_answer

    def answer_root(self, quotes: List[Quote]) -> Optional[Answer]:
        """This generates the answer to the root question."""
        # Find the root node and refine
        root_nodes = [
            node
            for node in self.question_graph.nodes
            if self.question_graph.in_degree(node) == 0  # noqa
        ]
        assert len(root_nodes) == 1, "Only a single root node supported for now."
        answer = self.refine_answer_for_query(root_nodes[0], quotes)
        return answer

    def synthesize_follow_up_queries(self, node: QANode) -> List[Query]:
        # Select the follow-up questions
        follow_up_questions = select_follow_up_question(
            question=node.query.text,
            current_answer=node.answer.text,
            model_name=self.follow_up_question_synthesis_model_name,
        )

        # Get the root query from which we inherit some query properties
        root_node = [
            node
            for node in self.question_graph.nodes
            if self.question_graph.in_degree(node) == 0
        ]
        assert len(root_node) == 1, "Only a single root node supported for now."
        root_query = root_node[0].query

        # Make a query out of the follow up questions
        follow_up_queries = []
        for follow_up_question in follow_up_questions:
            query = Query(
                text=follow_up_question,
                max_budget=self._agent.credit,
                created_at_time=self._agent.now,
                issued_by=self._agent,
                required_by_time=root_query.required_by_time,
                processor_model=root_query.processor_model,
                embedding_model=root_query.embedding_model,
            )
            follow_up_queries.append(query)

        return follow_up_queries

    def apply_refinement(
        self,
        node: QANode,
        successors: List[QANode],
        commit: bool = True,
    ) -> Optional[Answer]:
        successors = [
            node
            for node in successors
            if node.status in [AnswerStatus.ANSWERED, AnswerStatus.REFINED]
        ]
        if len(successors) == 0:
            # If we're here, then we tried to apply refinement but failed.
            # In this case, we should mark all successors as failed, and change
            # the status back to answered if possible
            if commit:
                if node.status in [
                    AnswerStatus.PENDING_FOLLOW_UP,
                    AnswerStatus.ANSWERED,
                ]:
                    # Change the status back to answered, so the existing answers can be used by
                    # upstream questions
                    node.mark_as_answered()
                    # Mark all successors as failed
                    self.mark_all_successors_with_status(
                        node, AnswerStatus.FAILED, include_self=False
                    )
                elif node.status == AnswerStatus.FAILED:
                    # If we're here, then we tried to apply refinement to a node that failed
                    # to have an answer. Obviously we won't try to refine it.
                    pass
                else:
                    raise ValueError(
                        f"Something went wrong, was expecting node status to be "
                        f"PENDING_FOLLOW_UP, ANSWERED, or FAILED, got {node.status}"
                    )
            return node.answer
        # There's some refinement to do
        refined_answer = refine_answer(
            question=node.query.text,
            original_answer=node.answer.text,
            follow_up_questions=[succ.query.text for succ in successors],
            answers_to_follow_up_questions=[succ.answer.text for succ in successors],
            model_name=self.answer_synthesis_model_name,
        )
        # Construct a new answer object
        answer = Answer(
            success=True,
            text=refined_answer,
            blocks=[block for succ in successors for block in succ.answer.blocks],
            relevance_scores=[
                score for succ in successors for score in succ.answer.relevance_scores
            ],
        )

        if answer is not None and commit:
            node.bind_refined_answer(answer)

        return answer

    def evaluation_summary(self) -> Dict[str, Any]:
        node_summaries = [
            node.evaluation_summary() for node in self.question_graph.nodes
        ]
        try:
            # Get the edges of the graph
            adjacency_matrix = nx.adjacency_matrix(self.question_graph).todense().tolist()
        except Exception:
            print("Oh Donkey")
            adjacency_matrix = []
        return dict(
            node_summaries=node_summaries,
            adjacency_matrix=adjacency_matrix,
        )
