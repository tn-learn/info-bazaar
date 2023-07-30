import uuid
from contextlib import contextmanager
from typing import Optional, List, Dict, Union

import mesa

from bazaar.schema import (
    Principal,
    BulletinBoard,
    AgentStatus,
    BuyerPrincipal,
    Query,
    Quote,
    Institution,
    Author,
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
    def __init__(self, principal: Union[Institution, Author]):
        super().__init__(principal)
        self._credit: int = 0
        self._query_queue: List[Query] = []
        self._pending_quotes: List[Quote] = []
        self._authorized_quotes: List[Quote] = []

    def issue_quote(self, query: Query):
        """
        For the query, this function issues a quote and posts it to the bulletin board.
        """
        # TODO
        pass

    def issue_quotes(self):
        """
        Issue all quotes in query queue and drain the queue.
        """
        pass

    def accept_query(self, query: Query) -> bool:
        """
        This function decides if a query is to be accepted. If it is, then it is added to the queue.
        """
        pass

    def check_bulletin_board(self):
        """
        This function checks if there are queries in the bulletin board that can be accepted.
        If yes, put them in the query queue list.
        """
        # TODO
        pass

    def process_quotes(self):
        """
        This function checks if
            1. There are quotes that were pending in the previous step, but now have been authorized.
                 These quotes are moved from pending_quotes to authorized_quotes.
            2. There are quotes that were previously authorized, but now have been accepted.
                 The credit is increased by the price amount, and the quote is removed from authorized_quotes.
        """
        # TODO
        pass

    def forward(self) -> None:
        # Step 1: Check if there are quotes that need to be processed.
        self.process_quotes()
        # Step 2: Check the bulletin board for new queries
        self.check_bulletin_board()
        # Step 3: Issue quotes for the accepted queries
        self.issue_quotes()


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
        # TODO
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
        buyer_agents: List[BuyerAgent],
        vendor_agents: List[VendorAgent],
        bulletin_board: BulletinBoard,
    ):
        super().__init__()
        self.schedule = mesa.time.RandomActivation(self)
        self.buyer_agents = buyer_agents
        self.vendor_agents = vendor_agents
        self.bulletin_board = bulletin_board
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
