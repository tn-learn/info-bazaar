import uuid
from contextlib import contextmanager
from typing import Optional, List, Dict, Union
import guidance
import mesa

from bazaar.database import retrieve_blocks
from bazaar.lem_utils import clean_program_string
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


class BazaarAgent(mesa.Agent):
    def __init__(self, principal: Principal):
        super().__init__(**BazaarAgentContext.get_mesa_agent_kwargs())
        # Privates
        self._credit = 0
        # Publics
        self.principal = principal
        self.agent_status = AgentStatus.ACTIVE
        self.called_count = 0

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


class VendorAgent(BazaarAgent):
    def __init__(
        self,
        principal: Union[Institution, Author],
        bulletin_board_retrieval_top_k: Optional[int] = None,
        bulletin_board_retrieval_score_threshold: Optional[float] = None,
    ):
        super().__init__(principal)
        # Privates
        self._outstanding_quotes: List[Quote] = []
        self._bulletin_board_last_checked_at: int = -1
        self._bulletin_board_retrieval_top_k = bulletin_board_retrieval_top_k
        self._bulletin_board_retrieval_score_threshold = (
            bulletin_board_retrieval_score_threshold
        )

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
        all_retrieved_outputs = retrieve_blocks(
            queries=queries_in_bulletin_board,
            blocks=self.principal.blocks,
            use_hyde=True,
            top_k=self._bulletin_board_retrieval_top_k,
            score_threshold=self._bulletin_board_retrieval_score_threshold,
        )
        self._bulletin_board_last_checked_at = self.now
        # Issue the quotes
        for retrieved in all_retrieved_outputs:
            if len(retrieved.blocks) == 0:
                continue
            price = 0
            answer_blocks = []
            for retrieved_block in retrieved.blocks:
                price += self.principal.block_prices[retrieved_block.block_id]
                answer_blocks.append(retrieved_block)
            quote = Quote(
                query=retrieved.query,
                price=price,
                answer_blocks=answer_blocks,
                created_at_time=self.now,
                issued_by=self,
            )
            retrieved.query.issued_by.receive_quote(quote)
            self._outstanding_quotes.append(quote)

    def forward(self) -> None:
        # Step 1: Check the bulletin board for new queries
        self.check_bulletin_board_and_issue_quotes()


class BuyerAgent(BazaarAgent):
    def __init__(self, principal: BuyerPrincipal):
        super().__init__(principal)
        self._credit: int = 0
        self._query_queue: List[Query] = []
        self._quote_inbox: List[Quote] = []
        self._accepted_quotes: List[Quote] = []
        self._final_response: Optional[str] = None

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

    def submit_final_response(self, response: str) -> "BuyerAgent":
        self._final_response = response
        return self.terminate_agent()  # F

    @property
    def response_submission_due_now(self):
        self.principal: "BuyerPrincipal"
        return self.principal.time_left == 0

    def submit_query_queue_to_bulletin_board(self):
        """
        This function submits the queries in the query queue to the bulletin board.
        """
        # TODO
        pass

    def process_quotes(self):
        """
        This function checks the quotes in the inbox and determines if they should be
        accepted, rejected, or waited on.
         -> If a quote should be accepted, the quote is added to the accepted quotes list, and the credit is deducted.
         -> If a quote is to be rejected, the quote is removed from the inbox.
         -> If a quote is to be waited on, the quote is left in the inbox.
        """
        for quote_idx in range(len(self._quote_inbox)):
            quote = self._quote_inbox[quote_idx]
            if quote.quote_status == QuoteStatus.ACCEPTED:
                continue
            if quote.quote_status == QuoteStatus.REJECTED:
                del self._quote_inbox[quote_idx]
            elif quote.quote_status == QuoteStatus.WAITING:
                pass


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
        # TODO
        pass

    def select_quote(self, candidate_quotes: List[Quote]):
        program_string = """
        {{#system~}}
        You are a Question Answering Agent operating inside an information market. You will be given a question, and a bunch of passages that might have an answer to that question in them. But beware that each passage has a cost. You want to minimize the amount you spend, while maximizing the quality of your answer. You will now be presented with several options; each has a price (in USD) and some text. You have the choice to buy no passage, one passage, or multiple passages.
        {{~/system}}

        {{#user~}}
        Your balance is ${{credit}}. Your question is: {{question}}?   

        Here are your options: 
        ---{{#each candidate_quotes}}
        Option {{@index}} for ${{this.price}}: {{this.block}}
        {{/each}}
        ---
        Please discuss each option very briefly (one line for pros, one for cons).
        {{~/user}}
        
        {{#assistant~}}
        {{gen 'procons' stop="\\n\\n" temperature=0.0}}
        {{~/assistant}}
        
        {{#user~}}
        Which of the passages would you like to purchase? Reply with {{#each candidate_quotes}}
        OPTION {{@index}} - Buy or Pass
        {{/each~}}
        {{~/user}}
        
        {{#assistant~}}
        {{gen 'buyorpass' stop="\\n\\n" temperature=0.0}}
        {{~/assistant}}
        """
        program_string = clean_program_string(program_string)
        program = guidance(program_string)
        # TODO Custom parsing functions

    def forward(self) -> None:
        # Step 1: Check if there are quotes in the inbox that need to be processed.
        self.process_quotes()
        # Step 2: Check if there are queries that need to be submitted to the bulletin board
        self.submit_query_queue_to_bulletin_board()
        # Step 3: Finalize the step (this decides what happens in the next step)
        self.finalize_step()


class BazaarSimulator(mesa.Model):
    def __init__(
        self,
        buyer_principals: List[BuyerPrincipal],
        vendor_principals: List[Union[Institution, Author]],
    ):
        super().__init__()
        self.schedule = mesa.time.RandomActivation(self)
        with BazaarAgentContext.activate(self):
            self.buyer_agents = [
                BuyerAgent(principal) for principal in buyer_principals
            ]
            self.vendor_agents = [
                VendorAgent(principal) for principal in vendor_principals
            ]
        self.bulletin_board = BulletinBoard()

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

    def run(self, max_num_steps: Optional[int] = None):
        if max_num_steps is None:
            while not self.all_buyer_agents_terminated:
                self.step()
        else:
            for _ in range(max_num_steps):
                if self.all_buyer_agents_terminated:
                    break
                self.step()
