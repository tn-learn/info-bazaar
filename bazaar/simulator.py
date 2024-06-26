from collections import defaultdict
from contextlib import contextmanager
from typing import Optional, List, Dict, Union, Any, Callable
import traceback
import mesa
import numpy as np

from bazaar.database import build_retriever
from bazaar.lem_utils import (
    select_quotes_with_debate,
    rerank_quotes,
    default_llm_name,
    default_reranker_name, select_quotes_with_bm25_heuristic,
)
from bazaar.qa_engine import QueryManager
from bazaar.schema import (
    Principal,
    BulletinBoard,
    AgentStatus,
    BuyerPrincipal,
    Query,
    Quote,
    Institution,
    Author,
    QuoteStatus,
    Answer,
)


class BazaarAgentContext:
    BAZAAR_SIMULATOR: Optional["BazaarSimulator"] = None
    UNIQUE_ID_COUNTER: Optional[int] = None
    PRINTER = print

    @classmethod
    @contextmanager
    def activate(cls, bazaar_simulator: "BazaarSimulator"):
        cls.BAZAAR_SIMULATOR = bazaar_simulator
        cls.UNIQUE_ID_COUNTER = 0
        yield
        cls.BAZAAR_SIMULATOR = None
        cls.UNIQUE_ID_COUNTER = None

    @classmethod
    def get_unique_id(cls, increment: bool = True) -> int:
        counter = cls.UNIQUE_ID_COUNTER
        if increment:
            cls.UNIQUE_ID_COUNTER += 1
        return counter

    @classmethod
    def get_model(cls) -> "BazaarSimulator":
        return cls.BAZAAR_SIMULATOR

    @classmethod
    def get_mesa_agent_kwargs(cls) -> Dict[str, Union[int, "BazaarSimulator"]]:
        return {
            "unique_id": cls.get_unique_id(increment=True),
            "model": cls.get_model(),
        }

    @classmethod
    def bind_printer(cls, printer: Callable[[str], Any]):
        cls.PRINTER = printer

    @classmethod
    def print(cls, *args, **kwargs):
        cls.PRINTER(*args, **kwargs)


class BazaarAgent(mesa.Agent):
    def __init__(self, principal: Principal):
        super().__init__(**BazaarAgentContext.get_mesa_agent_kwargs())
        # Privates
        self._credit = 0
        self._exception_str = None
        # Publics
        self.principal = principal
        self.agent_status = AgentStatus.ACTIVE
        self.called_count = 0

    @staticmethod
    def get_llm_name(model_name: Optional[str]) -> str:
        return model_name or default_llm_name()

    @staticmethod
    def get_reranker_name(model_name: Optional[str]) -> str:
        return model_name or default_reranker_name()

    @property
    def now(self):
        self.model: "BazaarSimulator"
        return self.model.now

    @property
    def credit(self):
        return self._credit

    def credit_to_account(self, amount: int) -> "BazaarAgent":
        self._credit += amount
        return self

    def deduct_from_account(self, amount: int) -> "BazaarAgent":
        self._credit -= amount
        return self

    def transfer_to_agent(self, amount: int, agent: "BazaarAgent") -> "BazaarAgent":
        self.deduct_from_account(amount)
        agent.credit_to_account(amount)
        return self

    def prepare(self):
        """This function is called once before the first forward call."""
        pass

    @property
    def bulletin_board(self) -> BulletinBoard:
        self.model: "BazaarSimulator"
        return self.model.bulletin_board

    def terminate_agent(self) -> "BazaarAgent":
        self.agent_status = AgentStatus.TERMINATED
        return self

    def forward(self, *args, **kwargs) -> None:
        """This function is called every step."""
        raise NotImplementedError

    def register_exception_str(self, exception_str: str) -> "BazaarAgent":
        # Get the stack trace of the exception
        self._exception_str = exception_str
        return self

    def step(self) -> None:
        if self.agent_status == AgentStatus.ACTIVE:
            if self.called_count == 0:
                self.prepare()
            try:
                self.forward()
                self.called_count += 1
            except Exception as e:
                self.register_exception_str(traceback.format_exc())
                self.print(f"Exception encountered: {str(e)}")
                self.print(f"Exception stack trace: {self._exception_str}")
                self.terminate_agent()
                raise

    def print(self, *messages):
        message = " ".join([str(m) for m in messages])
        agent_info = f"{self.__class__.__name__}(id={self.unique_id})"
        BazaarAgentContext.print(f"[{agent_info}] {message}")

    def evaluation_summary(self) -> Dict[str, Any]:
        summary = dict(
            agent_status=str(self.agent_status),
            credit=float(self.credit),
            called_count=self.called_count,
            principal=self.principal.evaluation_summary(),
            exception_str=self._exception_str,
        )
        return summary


class VendorAgent(BazaarAgent):
    def __init__(
        self,
        principal: Union[Institution, Author],
        retriever_config: dict = None,
    ):
        super().__init__(principal)
        # Privates
        self._outstanding_quotes: List[Quote] = []
        self._bulletin_board_last_checked_at: int = -1
        self._retriever = build_retriever(**(retriever_config or {}))

    def check_bulletin_board_and_issue_quotes(self):
        """
        This function checks if there are queries in the bulletin board that can be accepted.
        If yes, issue quotes.
        """
        self.model: "BazaarSimulator"
        self.principal: Union[Institution, Author]
        queries_in_bulletin_board = self.model.bulletin_board.new_queries_since(
            self._bulletin_board_last_checked_at
        )
        self._bulletin_board_last_checked_at = self.now

        # Filter out the queries for which there is already a quote issued
        queries_in_bulletin_board = [
            query
            for query in queries_in_bulletin_board
            if not any(
                [
                    quote.query.compare_content(query)
                    for quote in self._outstanding_quotes
                ]
            )
        ]

        if len(queries_in_bulletin_board) == 0:
            return
        
        all_retrieved_outputs = self._retriever(
            queries=queries_in_bulletin_board,
            blocks=list(self.principal.public_blocks.values()),
        )
        
        # Issue the quotes
        for retrieved in all_retrieved_outputs:
            for retrieved_block, retrieval_score in zip(
                retrieved.blocks, retrieved.scores
            ):
                budget = retrieved.query.max_budget
                price = self.principal.block_prices[retrieved_block.block_id]
                if price > budget:
                    # Don't issue quotes for blocks that are too expensive
                    continue
                quote = Quote(
                    query=retrieved.query,
                    price=price,
                    answer_blocks=[retrieved_block],
                    created_at_time=self.now,
                    issued_by=self,
                    relevance_scores=[retrieval_score],
                )
                retrieved.query.issued_by.receive_quote(quote.issue())
                self._outstanding_quotes.append(quote)

    def forward(self) -> None:
        # Step 1: Check the bulletin board for new queries
        self.check_bulletin_board_and_issue_quotes()


class BuyerAgent(BazaarAgent):
    def __init__(
        self,
        principal: BuyerPrincipal,
        quote_review_top_k: Optional[int] = None,
        quote_review_use_block_metadata: bool = False,
        quote_review_use_metadata_only: bool = False,
        quote_review_num_tries: int = 1,
        num_quote_gathering_steps: int = 0,
        max_query_depth: int = 2,
        max_num_follow_up_questions_per_question: Optional[int] = None,
        stay_faithful_to_quotes_when_sythesizing_answers: bool = False,
        use_reranker: bool = False,
        reranker_max_num_quotes: Optional[int] = None,
        quote_selection_model_name: Optional[str] = None,
        answer_synthesis_model_name: Optional[str] = None,
        follow_up_question_synthesis_model_name: Optional[str] = None,
        reranking_model_name: Optional[str] = None,
        quote_selection_function_name: str = "select_quotes_with_debate",
        disable_answer_synthesis: bool = False,
    ):
        super().__init__(principal)
        # Privates
        self._query_queue: List[Query] = []
        self._submitted_queries: List[Query] = []
        self._quote_inbox: List[Quote] = []
        self._accepted_quotes: List[Quote] = []
        self._rejected_quotes: List[Quote] = []
        self._query_manager: QueryManager = QueryManager(
            agent=self,
            max_query_depth=max_query_depth,
            answer_synthesis_model_name=self.get_llm_name(answer_synthesis_model_name),
            follow_up_question_synthesis_model_name=self.get_llm_name(
                follow_up_question_synthesis_model_name
            ),
            max_num_follow_up_questions_per_question=max_num_follow_up_questions_per_question,
            stay_faithful_to_quotes_when_sythesizing_answers=stay_faithful_to_quotes_when_sythesizing_answers,
        )
        # Publics
        self.quote_review_top_k = quote_review_top_k
        self.quote_review_use_block_metadata = quote_review_use_block_metadata
        self.quote_review_use_metadata_only = quote_review_use_metadata_only
        self.quote_review_num_tries = quote_review_num_tries
        self.num_quote_gathering_steps = num_quote_gathering_steps
        self.use_reranker = use_reranker
        self.reranker_max_num_quotes = reranker_max_num_quotes
        self.quote_selection_model_name = self.get_llm_name(quote_selection_model_name)
        self.reranking_model_name = self.get_reranker_name(reranking_model_name)
        self.quote_selection_function_name = quote_selection_function_name
        self.disable_answer_synthesis = disable_answer_synthesis

    def prepare(self):
        """
        Initialize the agent's query queue from the principal's query.
        """
        self.principal: BuyerPrincipal
        self._query_queue.append(self.principal.query.ensure_issued_by(self))
        self._query_manager.add_query(self.principal.query)
        self.credit_to_account(self.principal.budget)

    def receive_quote(self, quote: Quote):
        self._quote_inbox.append(quote)

    def submit_final_response(self, response: Optional[Answer]) -> "BuyerAgent":
        self.principal: "BuyerPrincipal"
        self.print(f"Buyer {self.principal.name} submitted final response.")
        if response is None:
            response = Answer(success=False)
        self.principal.submit_final_response(answer=response)
        # Cancel all outstanding quotes
        for quote in list(self._quote_inbox):
            self.reject_quote(quote)

        # Terminate the agent
        return self.terminate_agent()  # F

    @property
    def response_submission_due_now(self):
        self.principal: "BuyerPrincipal"
        return self.principal.time_left(self.now) == 0

    def post_queries_to_bulletin_board(self):
        """
        This function submits the queries in the query queue to the bulletin board.
        """
        self.model: "BazaarSimulator"
        for query in list(self._query_queue):
            if query.created_at_time <= self.now:
                query.ensure_issued_by(self)
                self.model.bulletin_board.post(query)
                self._query_queue.remove(query)
                self._submitted_queries.append(query)
                self.print(f"Posted query to bulletin board: {query.text}")

    def accept_or_claim_quote(self, quote: Quote) -> "BuyerAgent":
        # If the block content in the quote is already available, we don't need to buy it again.
        for accepted_quote in self._accepted_quotes:
            if (
                accepted_quote.compare_block_content(quote)
                and accepted_quote.issued_by.unique_id == quote.issued_by.unique_id
            ):
                # Move to accepted quotes and remove from inbox, but don't pay for it.
                quote.claim_quote()
                self._accepted_quotes.append(quote)
                if quote in self._quote_inbox:
                    self._quote_inbox.remove(quote)
                break
        else:
            quote.accept_quote()
            self.transfer_to_agent(quote.price, quote.issued_by)
            self._accepted_quotes.append(quote)
            if quote in self._quote_inbox:
                self._quote_inbox.remove(quote)
        return self

    def reject_quote(self, quote: Quote) -> "BuyerAgent":
        quote.reject_quote()
        self._rejected_quotes.append(quote)
        if quote in self._quote_inbox:
            self._quote_inbox.remove(quote)
        return self

    def reject_all_quotes_for_query(self, query: Query) -> "BuyerAgent":
        for quote in list(self._quote_inbox):
            if quote.query.compare_content(query):
                self.reject_quote(quote)
        return self

    def process_quotes(self):
        """
        This function checks the quotes in the inbox and determines if they should be
        accepted, rejected, or waited on.
         -> If a quote should be accepted, the quote is added to the accepted quotes list, and the credit is deducted.
         -> If a quote is to be rejected, the quote is removed from the inbox.
         -> If a quote is to be waited on, the quote is left in the inbox.
        """

        pending_quotes = [
            quote
            for quote in self._quote_inbox
            if quote.quote_status == QuoteStatus.PENDING
        ]
        valid_pending_quotes = [
            quote
            for quote in pending_quotes
            if (self.now - quote.query.created_at_time)
            >= min(quote.query.urgency, self.num_quote_gathering_steps)
        ]
        # If we have pending quotes and the wait time is up, we need to process them
        if (
            len(valid_pending_quotes) > 0
            and pending_quotes[0].created_at_time + self.num_quote_gathering_steps
            <= self.now
        ):
            # We need to separate quotes by queries and process them individually.
            # This is how the infra in lem_utils is set up.
            valid_pending_quotes_by_query = defaultdict(list)
            for quote in valid_pending_quotes:
                valid_pending_quotes_by_query[quote.query].append(quote)
            for query, quotes in valid_pending_quotes_by_query.items():
                # Select the quotes to accept
                quotes_to_accept = self.select_quote(quotes)
                for quote_to_accept in quotes_to_accept:
                    self.accept_or_claim_quote(quote_to_accept)
                # Reject the rest of the quotes for this query
                self.reject_all_quotes_for_query(query)
                # Remove from bulletin board
                self.bulletin_board.mark_query_as_processed(query)

    def synthesize_final_response(self) -> Answer:
        if self.disable_answer_synthesis:
            if len(self._accepted_quotes) > 0:
                answer = Answer(
                    success=True,
                    text="\n---\n".join([quote.answer_blocks[0].content for quote in self._accepted_quotes]),
                    blocks=[quote.answer_blocks[0] for quote in self._accepted_quotes],
                    relevance_scores=[max(quote.relevance_scores) for quote in self._accepted_quotes],
                )
            else:
                answer = Answer(success=False)
        else:
            answer = self._query_manager.answer_root(self._accepted_quotes)
        return answer

    def enqueue_follow_up_queries(self):
        follow_up_queries = self._query_manager.generate_follow_up_queries(
            self._accepted_quotes, commit=True
        )
        for follow_up_query, _ in follow_up_queries:
            self._query_queue.append(follow_up_query)

    def finalize_step(self):
        """
        This is where we decide how to conclude the step. There are two possibilities:
            1. The time is up, and we need to synthesize the final response.
            2. There is time to dig deeper.

        # If the time is up, we synthesize the response.
        """
        if self.response_submission_due_now:
            # If this happens, we need to submit a final response.
            final_response = self.synthesize_final_response()
            self.submit_final_response(final_response)
        elif self._query_manager.max_query_depth > 0:
            # If there are accepted quotes, we synthesize an answer with them.
            self.enqueue_follow_up_queries()

    def remove_duplicate_quotes(self, quotes: List[Quote]) -> List[Quote]:
        # If there are two quotes with the same blocks and the same query, we keep
        # the one that is cheaper.
        # No-op codepath
        if len(quotes) == 0:
            return []
        # Real codepath
        quotes_by_query = defaultdict(lambda: defaultdict(list))
        for quote in quotes:
            quotes_by_query[quote.query][quote.get_block_content_hash()].append(quote)
        quotes = []
        for query, quotes_this_query in quotes_by_query.items():
            for _, quotes_this_query_by_block_content in quotes_this_query.items():
                quotes.append(
                    min(quotes_this_query_by_block_content, key=lambda q: q.price)
                )
        return quotes

    def select_quote(self, candidate_quotes: List[Quote]) -> List[Quote]:
        # The condition for calling this function is that all candidate_quotes have
        # the same query
        assert (
            len(set([quote.query for quote in candidate_quotes])) == 1
        ), "All candidate quotes must have the same query."
        # First we filter out all the quotes that are too expensive
        candidate_quotes = [
            quote for quote in candidate_quotes if quote.price <= self.credit
        ]
        # First filter: remove duplicate quotes
        candidate_quotes = [
            quote.progress_quote()
            for quote in self.remove_duplicate_quotes(candidate_quotes)
        ]
        if len(candidate_quotes) > 0:
            self.print(
                f"Received {len(candidate_quotes)} quotes for query: {candidate_quotes[0].query.text}"
            )
        else:
            self.print(f"There were no quotes to review.")
        # Apply reranker if required
        if len(candidate_quotes) > 0 and self.use_reranker:
            # Before applying the reranker, we want to make sure that we don't have
            # too many quotes. We only want to rerank the top k quotes as specified
            # by self.reranker_max_num_quotes.
            if (
                self.reranker_max_num_quotes is not None
                and len(candidate_quotes) > self.reranker_max_num_quotes
            ):
                threshold_quantile = 1 - (
                    self.reranker_max_num_quotes / len(candidate_quotes)
                )
                threshold_score = np.quantile(
                    [max(quote.relevance_scores) for quote in candidate_quotes],
                    threshold_quantile,
                )
                candidate_quotes = [
                    quote.progress_quote()
                    for quote in candidate_quotes
                    if max(quote.relevance_scores) >= threshold_score
                ]
            scores = rerank_quotes(
                candidate_quotes, model_name=self.reranking_model_name
            )
        else:
            scores = [max(quote.relevance_scores) for quote in candidate_quotes]
        if len(candidate_quotes) > 0 and self.quote_review_top_k is not None:
            # Keep the top_k quotes as ranked by their scores
            num_candidate_quotes = self.quote_review_top_k * self.quote_review_num_tries
            candidate_quotes = [
                quote
                for _, quote in sorted(
                    zip(scores, candidate_quotes),
                    key=lambda pair: pair[0],
                    reverse=True,
                )
            ][:num_candidate_quotes]
            candidate_quotes = [quote.progress_quote() for quote in candidate_quotes]
        # Select the quotes
        selected_quotes = []
        for try_count in range(self.quote_review_num_tries):
            if len(selected_quotes) > 1:
                break
            # Figure out which slice to select over this try
            index_start = try_count * self.quote_review_top_k
            index_end = (try_count + 1) * self.quote_review_top_k
            candidate_quotes_this_try = candidate_quotes[index_start:index_end]
            # Get the selector function
            if self.quote_selection_function_name == "select_quotes_with_debate":
                # Select the quotes
                selected_quotes.extend(
                    list(
                        select_quotes_with_debate(
                            candidate_quotes_this_try,
                            budget=self.credit,
                            model_name=self.quote_selection_model_name,
                            use_block_content_metadata=self.quote_review_use_block_metadata,
                            use_block_metadata_only=self.quote_review_use_metadata_only,
                        )
                    )
                )
            elif self.quote_selection_function_name == "select_quotes_with_bm25_heuristic":
                selected_quotes.extend(
                    list(
                        select_quotes_with_bm25_heuristic(
                            candidate_quotes_this_try,
                            budget=self.credit,
                        )
                    )
                )
            else:
                raise NotImplementedError(
                    f"Quote selection function {self.quote_selection_function_name} not implemented."
                )

        selected_quotes = [quote.progress_quote() for quote in selected_quotes]
        if len(candidate_quotes) > 0:
            self.print(
                f"Selected {len(selected_quotes)} quotes for query: {candidate_quotes[0].query.text}"
            )
        else:
            self.print(f"There were no quotes to select.")
        return selected_quotes

    def forward(self) -> None:
        # Step 1: Check if there are quotes in the inbox that need to be processed.
        self.process_quotes()
        self.print(f"Processed quotes at t = {self.now}.")
        # Step 2: Check if there are queries that need to be submitted to the bulletin board
        self.post_queries_to_bulletin_board()
        self.print(f"Posted queries to bulletin board at t = {self.now}.")
        # Step 3: Finalize the step (this decides what happens in the next step)
        self.finalize_step()
        self.print(f"Finalized step at t = {self.now}.")

    def evaluation_summary(self) -> Dict[str, Any]:
        summary = super().evaluation_summary()
        summary.update(
            dict(
                accepted_quotes=[
                    quote.evaluation_summary() for quote in self._accepted_quotes
                ],
                rejected_quotes=[
                    quote.evaluation_summary() for quote in self._rejected_quotes
                ],
                submitted_queries=[
                    query.evaluation_summary() for query in self._submitted_queries
                ],
                quote_inbox=[quote.evaluation_summary() for quote in self._quote_inbox],
                query_manager=self._query_manager.evaluation_summary(),
            )
        )
        return summary


class BazaarSimulator(mesa.Model):
    def __init__(
        self,
        bulletin_board: BulletinBoard,
        buyer_principals: List[BuyerPrincipal],
        vendor_principals: List[Union[Institution, Author]],
        seed: Optional[int] = None,
        buyer_agent_kwargs: Optional[Dict[str, Any]] = None,
        vendor_agent_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(seed=seed)
        self.schedule = mesa.time.RandomActivation(self)
        with BazaarAgentContext.activate(self):
            self.bulletin_board = bulletin_board
            self.buyer_agents = [
                BuyerAgent(principal, **(buyer_agent_kwargs or {}))
                for principal in buyer_principals
            ]
            self.vendor_agents = [
                VendorAgent(principal, **(vendor_agent_kwargs or {}))
                for principal in vendor_principals
            ]

        for agent in self.buyer_agents:
            self.schedule.add(agent)
        for agent in self.vendor_agents:
            self.schedule.add(agent)

    @property
    def now(self):
        return self.schedule.time

    @property
    def all_buyer_agents_terminated(self) -> bool:
        return all(
            [
                agent.agent_status == AgentStatus.TERMINATED
                for agent in self.buyer_agents
            ]
        )

    def step(self):
        self.schedule.step()
        self.bulletin_board.maintain(self.now)

    def run(
        self,
        max_num_steps: Optional[int] = None,
        print_callback: Optional[Callable[[str], Any]] = None,
        step_callback: Optional[Callable[[int], Any]] = None,
    ):
        if print_callback is None:
            print_callback = lambda *args, **kwargs: None
        else:
            BazaarAgentContext.bind_printer(print_callback)

        if step_callback is None:
            step_callback = lambda *args, **kwargs: None

        if max_num_steps is None:
            while not self.all_buyer_agents_terminated:
                print_callback(f"Simulating t = {self.now}.")
                self.step()
                step_callback(self.now)
        else:
            for _ in range(max_num_steps):
                if self.all_buyer_agents_terminated:
                    break
                print_callback(f"Simulating t = {self.now}.")
                self.step()
                step_callback(self.now)
        print_callback(f"Simulation complete at t = {self.now}.")

    def evaluation_summary(self) -> Dict[str, Any]:
        self.buyer_agents: List[BuyerAgent]
        self.vendor_agents: List[VendorAgent]
        return dict(
            buyer_agents=[agent.evaluation_summary() for agent in self.buyer_agents],
            vendor_agents=[agent.evaluation_summary() for agent in self.vendor_agents],
        )
