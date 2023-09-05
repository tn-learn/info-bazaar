import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List, Tuple

from bazaar.lem_utils import (
    get_open_book_answer,
    get_closed_book_answer,
    evaluate_answer_with_debate,
)
from bazaar.py_utils import load_dict, dump_dict, root_dir_slash


@dataclass
class AnswerQuality:
    comprehensiveness: int
    correctness: int
    simplicity: int
    relevance: int

    def summary(self):
        return dict(
            comprehensiveness=self.comprehensiveness,
            correctness=self.correctness,
            simplicity=self.simplicity,
            relevance=self.relevance,
        )


@dataclass
class BlockWareSpec:
    block_id: str
    vendor_index: int
    vendor_name: str
    block_type: str
    block_index: int
    block_price: Optional[float] = None

    def summary(self):
        return dict(
            block_id=self.block_id,
            vendor_index=self.vendor_index,
            vendor_name=self.vendor_name,
            block_type=self.block_type,
            block_price=self.block_price,
            block_index=self.block_index,
        )


@dataclass
class BuyerAgentEvaluationResult:
    name: str
    question_text: str
    max_budget: float
    credit_left: float
    successfully_answered: bool
    answer_text: Optional[str]
    gold_block_ware_specs: List[BlockWareSpec]
    gold_block_content: str
    gold_block_rejected: bool
    retrieved_answer_quality: Optional[AnswerQuality] = None
    open_book_answer_quality: Optional[AnswerQuality] = None
    closed_book_answer_quality: Optional[AnswerQuality] = None
    open_book_answer: Optional[str] = None
    closed_book_answer: Optional[str] = None

    def summary(self):
        return dict(
            name=self.name,
            question_text=self.question_text,
            max_budget=self.max_budget,
            credit_left=self.credit_left,
            successfully_answered=self.successfully_answered,
            answer_text=self.answer_text,
            gold_block_ware_specs=[
                block_ware_spec.summary()
                for block_ware_spec in self.gold_block_ware_specs
            ],
            gold_block_content=self.gold_block_content,
            gold_block_rejected=self.gold_block_rejected,
            retrieved_answer_quality=(
                self.retrieved_answer_quality.summary()
                if self.retrieved_answer_quality is not None
                else None
            ),
            open_book_answer_quality=(
                self.open_book_answer_quality.summary()
                if self.open_book_answer_quality is not None
                else None
            ),
            closed_book_answer_quality=(
                self.closed_book_answer_quality.summary()
                if self.closed_book_answer_quality is not None
                else None
            ),
            open_book_answer=self.open_book_answer,
            closed_book_answer=self.closed_book_answer,
        )


@dataclass
class EvaluationResult:
    buyer_agents: List[BuyerAgentEvaluationResult]

    def summary(self):
        return dict(
            buyer_agents=[buyer_agent.summary() for buyer_agent in self.buyer_agents]
        )


class Evaluator:
    def __init__(
        self,
        evaluation_summary: Optional[dict] = None,
        run_directory: Optional[str] = None,
    ):
        # Privates
        self._block_index = defaultdict(list)
        # Publics
        if evaluation_summary is None:
            assert run_directory is not None
            evaluation_summary = load_dict(
                str(Path(run_directory) / "bazaar_summary.json")
            )
        self.evaluation_summary = evaluation_summary
        self.run_directory = run_directory

    def find_block(self, block_id: str) -> List[BlockWareSpec]:
        if block_id in self._block_index:
            return self._block_index[block_id]
        # Find the block in the summary
        for vendor_agent_idx, vendor_agent_summary in enumerate(
            self.evaluation_summary["vendor_agents"]
        ):
            # Loop over all blocks of this vendor, both public and private
            for block_type in ["public", "private"]:
                for block_index, block_summary in enumerate(
                    vendor_agent_summary["principal"][f"{block_type}_blocks"]
                ):
                    if block_summary["block_id"] == block_id:
                        self._block_index[block_id].append(
                            BlockWareSpec(
                                block_id=block_id,
                                vendor_index=vendor_agent_idx,
                                vendor_name=vendor_agent_summary["principal"]["name"],
                                block_type=block_type,
                                block_price=vendor_agent_summary["principal"][
                                    "block_prices"
                                ][block_id],
                                block_index=block_index,
                            )
                        )
        
                        
        # for buyer_agent_idx, buyer_agent_summary in enumerate(self.evaluation_summary["buyer_agents"]):
        #     block_summary = buyer_agent_summary["principal"]["query"]["gold_block"]
        #     if block_summary["block_id"] == block_id:
        #         self._block_index[block_id].append(
        #             BlockWareSpec(
        #                 block_id=block_id,
        #                 vendor_index=buyer_agent_idx,
        #                 vendor_name=buyer_agent_summary["principal"]["name"],
        #                 block_type="gold",
        #                 block_index=block_index,
        #             )
        #         )

        if block_id not in self._block_index:
            breakpoint()
            raise ValueError(f"Block {block_id} not found.")
        return self._block_index[block_id]

    def get_block_content(self, block_ware_spec: BlockWareSpec) -> str:
        if block_ware_spec.block_type == "gold":
            return self.evaluation_summary["buyer_agents"][block_ware_spec.vendor_index][
                "principal"
            ]["query"]["gold_block"]["content"]
        else:
            return self.evaluation_summary["vendor_agents"][block_ware_spec.vendor_index][
                "principal"
            ][f"{block_ware_spec.block_type}_blocks"][block_ware_spec.block_index][
                "content"
            ]


    @staticmethod
    def evaluate_answer_quality(
        question_text: str, answer_text: Optional[str], gold_block_content: str
    ) -> Tuple[Dict[str, AnswerQuality], Dict[str, str]]:
        # Get the closed and open book answers
        open_book_answer = get_open_book_answer(
            question=question_text,
            gold_passage=gold_block_content,
            model_name="gpt-3.5-turbo",
        )
        closed_book_answer = get_closed_book_answer(
            question=question_text, model_name="gpt-3.5-turbo"
        )
        # Evaluate the answer qualities
        evaluated_answers = evaluate_answer_with_debate(
            question=question_text,
            gold_passage=gold_block_content,
            retrieved_answer=answer_text,
            open_book_answer=open_book_answer,
            closed_book_answer=closed_book_answer,
            model_name="gpt-4",
        )

        answer_qualities = dict(
            retrieved=AnswerQuality(**evaluated_answers["retrieved"]),
            closed_book=AnswerQuality(**evaluated_answers["closed_book"]),
            open_book=AnswerQuality(**evaluated_answers["open_book"]),
        )
        answers = dict(
            retrieved=answer_text,
            closed_book=closed_book_answer,
            open_book=open_book_answer,
        )
        return answer_qualities, answers

    def evaluate_buyer_agents(self) -> EvaluationResult:
        evaluation = []
        for buyer_agent_summary in self.evaluation_summary["buyer_agents"]:
            # Find the gold block and retrieve its content
            gold_block_id = buyer_agent_summary["principal"]["query"]["gold_block_id"]
            gold_block_ware_specs = self.find_block(gold_block_id)
            if len(gold_block_ware_specs) == 0:
                raise ValueError(f"Gold block {gold_block_id} not found.")
            gold_block_content = self.get_block_content(gold_block_ware_specs[0])
            # Evaluate the buyer agent
            # TODO: debug this - answer should never be empty
            try:
                buyer_agent_summary["principal"]["answer"]["success"]
            except Exception:
                breakpoint()
                continue

            if buyer_agent_summary["principal"]["answer"]["success"]:
                answer_text = buyer_agent_summary["principal"]["answer"]["text"]
            else:
                answer_text = None
            answer_qualities, answers = self.evaluate_answer_quality(
                question_text=buyer_agent_summary["principal"]["query"]["text"],
                answer_text=answer_text,
                gold_block_content=gold_block_content,
            )
            # Check if gold block was rejected as a quote
            gold_block_rejected = any(
                [
                    answer_block["block_id"] == gold_block_id
                    for rejected_quote_summary in buyer_agent_summary["rejected_quotes"]
                    for answer_block in rejected_quote_summary["answer_blocks"]
                ]
            )
            # Create the evaluation result and store it
            result = BuyerAgentEvaluationResult(
                name=buyer_agent_summary["principal"]["name"],
                question_text=buyer_agent_summary["principal"]["query"]["text"],
                max_budget=buyer_agent_summary["principal"]["query"]["max_budget"],
                credit_left=buyer_agent_summary["credit"],
                successfully_answered=buyer_agent_summary["principal"]["answer"][
                    "success"
                ],
                answer_text=buyer_agent_summary["principal"]["answer"]["text"],
                gold_block_ware_specs=gold_block_ware_specs,
                gold_block_content=gold_block_content,
                gold_block_rejected=gold_block_rejected,
                retrieved_answer_quality=answer_qualities.get("retrieved"),
                open_book_answer_quality=answer_qualities.get("open_book"),
                closed_book_answer_quality=answer_qualities.get("closed_book"),
                open_book_answer=answers.get("open_book"),
                closed_book_answer=answers.get("closed_book"),
            )
            evaluation.append(result)
            print(result)
            print("-----")
        return EvaluationResult(buyer_agents=evaluation)


def main(args: Optional[argparse.Namespace] = None):
    if args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--run_directory",
            type=str,
            default=None,
            help="path to run directory",
        )
        args = parser.parse_args()
    # Evaluate the buyer agents
    evaluator = Evaluator(run_directory=args.run_directory)
    print("Evaluating buyer agents...")
    evaluation = evaluator.evaluate_buyer_agents()
    # Done
    dump_dict(evaluation.summary(), Path(args.run_directory) / "evaluation.json")
    print("Done.")


if __name__ == "__main__":
    main()
