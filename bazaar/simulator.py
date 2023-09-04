from contextlib import contextmanager
from typing import Optional, List, Dict, Union, Any, Callable
import mesa

from bazaar.database import build_retriever
from bazaar.lem_utils import (
    select_quotes_with_debate,
    synthesize_answer,
    rerank_quotes,
    default_llm_name,
    default_reranker_name,
)
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

    def print(cls, *args, **kwargs):
        cls.PRINTER(*args, **kwargs)


class BazaarAgent(mesa.Agent):
    def __init__(self, principal: Principal):
        super().__init__(**BazaarAgentContext.get_mesa_agent_kwargs())
        # Privates
        self._credit = 0
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

    def step(self) -> None:
        if self.agent_status == AgentStatus.ACTIVE:
            if self.called_count == 0:
                self.prepare()
            self.forward()
            self.called_count += 1

    @staticmethod
    def print(*args, **kwargs):
        BazaarAgentContext.print(*args, **kwargs)

    def evaluation_summary(self) -> Dict[str, Any]:
        summary = dict(
            agent_status=str(self.agent_status),
            credit=float(self.credit),
            called_count=self.called_count,
            principal=self.principal.evaluation_summary(),
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
        num_quote_gathering_steps: int = 0,
        use_reranker: bool = False,
        quote_selection_model_name: Optional[str] = None,
        answer_synthesis_model_name: Optional[str] = None,
        reranking_model_name: Optional[str] = None,
    ):
        super().__init__(principal)
        # Privates
        self._query_queue: List[Query] = []
        self._submitted_queries: List[Query] = []
        self._quote_inbox: List[Quote] = []
        self._accepted_quotes: List[Quote] = []
        self._rejected_quotes: List[Quote] = []
        self._final_response: Optional[str] = None
        # Publics
        self.quote_review_top_k = quote_review_top_k
        self.num_quote_gathering_steps = num_quote_gathering_steps
        self.use_reranker = use_reranker
        self.quote_selection_model_name = self.get_llm_name(quote_selection_model_name)
        self.answer_synthesis_model_name = self.get_llm_name(
            answer_synthesis_model_name
        )
        self.reranking_model_name = self.get_reranker_name(reranking_model_name)

    def prepare(self):
        """
        Initialize the agent's query queue from the principal's query.
        """
        self.principal: BuyerPrincipal
        self._query_queue.append(self.principal.query)
        self.credit_to_account(self.principal.budget)

    def receive_quote(self, quote: Quote):
        self._quote_inbox.append(quote)

    @property
    def final_response(self) -> str:
        return self._final_response

    @property
    def final_response_available(self):
        return self._final_response is not None

    def submit_final_response(self, response: Optional[str]) -> "BuyerAgent":
        self.principal: "BuyerPrincipal"
        self._final_response = response
        self.print(f"Buyer {self.principal.name} submitted final response: {type(response)}")

        self.principal.submit_final_response(
            answer=response,
            blocks=[
                block
                for quote in self._accepted_quotes
                for block in quote.answer_blocks
            ],
            relevance_scores=[
                relevance_score
                for quote in self._accepted_quotes
                for relevance_score in quote.relevance_scores
            ],
        )
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
            if query.created_at_time == self.now:
                query.issued_by = self
                self.model.bulletin_board.post(query)
                self._query_queue.remove(query)
                self._submitted_queries.append(query)

    def accept_quote(self, quote: Quote) -> "BuyerAgent":
        quote.accept_quote()
        self.transfer_to_agent(quote.price, quote.issued_by)
        self._accepted_quotes.append(quote)
        self._quote_inbox.remove(quote)
        return self

    def reject_quote(self, quote: Quote) -> "BuyerAgent":
        quote.reject_quote()
        self._rejected_quotes.append(quote)
        self._quote_inbox.remove(quote)
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
        if (
            len(valid_pending_quotes) > 0
            and pending_quotes[0].created_at_time + self.num_quote_gathering_steps
            <= self.now
        ):
            quotes_to_accept = self.select_quote(valid_pending_quotes)
            for quote_to_accept in quotes_to_accept:
                self.accept_quote(quote_to_accept)
            for quote in list(self._quote_inbox):
                if (
                    len(quotes_to_accept) == 0
                    or quote.query == quotes_to_accept[0].query
                ):
                    self.reject_quote(quote)

    def gathered_quotes_are_good_enough(self) -> bool:
        # TODO: LEM call - with the  quotes and query
        pass

    def generate_follow_up_query(self) -> Query:
        pass

    def needs_to_generate_follow_up_query(self) -> bool:
        # TODO: implement this
        return False

    def finalize_step(self):
        """
        This is where the big decisions happen.
        Based on the accepted quotes, the agent decides if:
          1. It should submit a follow-up query to the bulletin board to gather more quotes. This is done
               by adding the query to the query queue.
          2. It should submit a final response and terminate. This happens when the agent has enough quotes
               to generate a response, or it's out of time. If it is to terminate, all outstanding quotes
               are rejected.
          3. It should do nothing and wait for quotes to come in.
        """

        # TODO: implement multi-hop queries to acquire compound information; requires splitting a query into multiple simpler queries and posting these.
        # TODO: fix this, it's a bleeding sharp edge if we have multi-query buyers
        query = self.principal.query
        if self.response_submission_due_now:
            response = synthesize_answer(
                query, self._accepted_quotes, model_name=self.answer_synthesis_model_name
            )
            self.submit_final_response(response)
            for quote in list(self._quote_inbox):
                self.reject_quote(quote)

        elif self.needs_to_generate_follow_up_query():
            # Submit follow-up query
            self._query_queue.append(self.generate_follow_up_query())

    def select_quote(self, candidate_quotes: List[Quote]) -> List[Quote]:
        # First we filter out all the quotes that are too expensive
        candidate_quotes = [
            quote for quote in candidate_quotes if quote.price <= self.credit
        ]
        # Apply reranker if required
        if len(candidate_quotes) > 0 and self.use_reranker:
            scores = rerank_quotes(
                candidate_quotes, model_name=self.reranking_model_name
            )
        else:
            scores = [max(quote.relevance_scores) for quote in candidate_quotes]
        if len(candidate_quotes) > 0 and self.quote_review_top_k is not None:
            # Keep the top_k quotes as ranked by their scores
            candidate_quotes = [
                quote
                for _, quote in sorted(
                    zip(scores, candidate_quotes),
                    key=lambda pair: pair[0],
                    reverse=True,
                )
            ][: self.quote_review_top_k]
        # Select the quotes
        return list(
            select_quotes_with_debate(
                candidate_quotes,
                budget=self.credit,
                model_name=self.quote_selection_model_name,
            )
        )

    def forward(self) -> None:
        # Step 1: Check if there are quotes in the inbox that need to be processed.
        self.process_quotes()
        # Step 2: Check if there are queries that need to be submitted to the bulletin board
        self.post_queries_to_bulletin_board()
        # Step 3: Finalize the step (this decides what happens in the next step)
        self.finalize_step()

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
    ):
        if print_callback is None:
            print_callback = lambda *args, **kwargs: None
        else:
            BazaarAgentContext.bind_printer(print_callback)

        if max_num_steps is None:
            while not self.all_buyer_agents_terminated:
                print_callback(f"Simulating t = {self.now}.")
                self.step()
        else:
            for _ in range(max_num_steps):
                if self.all_buyer_agents_terminated:
                    break
                print_callback(f"Simulating t = {self.now}.")
                self.step()
        print_callback(f"Simulation complete at t = {self.now}.")

    def evaluation_summary(self) -> Dict[str, Any]:
        self.buyer_agents: List[BuyerAgent]
        self.vendor_agents: List[VendorAgent]
        return dict(
            buyer_agents=[agent.evaluation_summary() for agent in self.buyer_agents],
            vendor_agents=[agent.evaluation_summary() for agent in self.vendor_agents],
        )
