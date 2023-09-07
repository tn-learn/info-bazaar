from dataclasses import dataclass
from enum import auto, IntEnum
from typing import TYPE_CHECKING, Optional, List, Tuple, Union, Any, Dict

from networkx import DiGraph

from bazaar.lem_utils import synthesize_answer, select_follow_up_question
from bazaar.schema import Query, Answer, Quote

if TYPE_CHECKING:
    from bazaar.simulator import BuyerAgent


class AnswerStatus(IntEnum):
    FAILED = auto()
    UNANSWERED = auto()
    PENDING_FOLLOW_UP = auto()
    ANSWERED = auto()
    REFINED = auto()


@dataclass
class QANode:
    query: Query
    answer: Optional[Answer] = None
    status: AnswerStatus = AnswerStatus.UNANSWERED

    def bind_answer(self, answer: Answer) -> "QANode":
        if not answer.success:
            self.status = AnswerStatus.FAILED
        else:
            self.status = AnswerStatus.ANSWERED
        self.answer = answer
        return self

    def mark_as_pending_follow_up(self) -> "QANode":
        self.status = AnswerStatus.PENDING_FOLLOW_UP
        return self

    def mark_as_refined(self) -> "QANode":
        self.status = AnswerStatus.REFINED
        return self


class QueryManager:
    def __init__(self, agent: "BuyerAgent", answer_synthesis_model_name: str):
        # Private
        self._agent = agent
        # Public
        self.answer_synthesis_model_name = answer_synthesis_model_name
        self.question_graph = DiGraph()

    def find_node(self, query: Query) -> QANode:
        for node in self.question_graph.nodes:
            if node.query.compare_content(query):
                return node
        raise ValueError(f"Could not find node for query {query}")

    def mark_all_predecessors_with_status(
        self, node: QANode, status: AnswerStatus
    ) -> "QueryManager":
        node.status = status

        for pred in self.question_graph.predecessors(node):
            self.mark_all_predecessors_with_status(pred, status)

        return self

    def add_query(
        self, query: Query, parent_query: Optional[Query] = None
    ) -> "QueryManager":
        if parent_query is None:
            self.question_graph.add_node(QANode(query=query))
        else:
            self.question_graph.add_edge(
                QANode(query=parent_query), QANode(query=query)
            )
            # We mark all upstream queries in the graph as pending follow up
            node = self.find_node(parent_query)
            self.mark_all_predecessors_with_status(node, AnswerStatus.PENDING_FOLLOW_UP)
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
            # If node has no answer, we try to generate one.
            if node.status == AnswerStatus.UNANSWERED:
                self.synthesize_answer_for_query(node, quotes, commit=True)
            # Check if node has an answer. If it does, then we can generate a
            # follow-up question.
            if node.status == AnswerStatus.ANSWERED:
                follow_ups = self.synthesize_follow_up_queries(node)
                for follow_up_query in follow_ups:
                    follow_up_queries.append((follow_up_query, node.query))
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
                blocks=[block for quote in quotes for block in quote.answer_blocks],
                relevance_scores=[
                    score for quote in quotes for score in quote.relevance_scores
                ],
            )
            if commit:
                node.bind_answer(answer)
        else:
            answer = None
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

            # If successor is unanswered, try synthesizing an answer
            if succ.status == AnswerStatus.UNANSWERED:
                answer = self.synthesize_answer_for_query(succ, quotes, commit=True)
                if answer is not None:
                    valid_successors.append(succ)
            elif succ.status in [AnswerStatus.ANSWERED, AnswerStatus.REFINED]:
                # If successor is answered or refined, add it to the list
                valid_successors.append(succ)

        # Now, use the apply_refinement function to update the answer of
        # the current node
        refined_answer = self.apply_refinement(node, valid_successors)

        if refined_answer:
            node.bind_answer(refined_answer).mark_as_refined()

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
            model_name="gpt-3.5-turbo",
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
        self, node: QANode, successors: List[QANode]
    ) -> Optional[Answer]:
        successors = [
            node
            for node in successors
            if node.status in [AnswerStatus.ANSWERED, AnswerStatus.REFINED]
        ]
        if len(successors) == 0:
            return None
        # There's some refinement to do
        # TODO: Implement this
        pass

    def evaluation_summary(self) -> Dict[str, Any]:
        # TODO: Implement this
        pass
