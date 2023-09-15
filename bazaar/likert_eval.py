import glob
import os
import argparse
import ast
import random
import yaml
from tqdm import tqdm
from typing import Optional, Dict, List, Any

import numpy as np
import pandas as pd

from bazaar.py_utils import dump_dict, load_dict
from bazaar.lem_utils import evaluate_answer_with_likert


def try_literal_eval(x):
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return x


def create_html_text(question, gold_block_content, answer, llm_name, answer_type):
    html_dict = {
        "Question": f'<span data-toggle="tooltip" data-html="true" data-placement="bottom" title="Context: Question"><b>QUESTION</b>: {question}</span><span><b>GOLD PASSAGE</b>: {gold_block_content}</span>',
        "Answer 1": f'<span data-toggle="tooltip" data-html="true" data-placement="bottom" title="Context: {answer_type}  {llm_name}">{answer}</span>',
    }
    return html_dict


class LikertEvaluator:
    """
    The purpose of this class is to produce a CSV with cols:
        "question": This is the question.
        "question_type": This is the question type (open, closed).
        "budget": This is the amount that the agent had to spend on this question.
        "credit_spent": This is the amount spent on retrieval.
        "gold_block": This is the gold passage that goes with the question.
        "answer": This is the retrieved / closed book / (open book) answer.
        "answer_type": Is this a retrieved answer, closed book answer, or open book answer?
        "model": This is the id of the model used to generate the answer.
        "evaluator_model": This is the id of the model used to evaluate the answer.
        ""
        "potato_text": HTML blurb that renders question, gold_passage and answer to HTML.
    """

    def __init__(
        self,
        experiment_root: str,
        experiment_name: str,
        evaluator_model: str,
        save_key: Optional[str] = None,
    ):
        self.experiment_root = experiment_root
        self.experiment_name = experiment_name
        self.evaluator_model = evaluator_model
        self.save_key = save_key

    @property
    def result_output_path(self):
        if self.save_key in ["", None]:
            return os.path.join(
                self.experiment_root, f"likert_eval_{self.evaluator_model}.csv"
            )
        else:
            return os.path.join(
                self.experiment_root,
                f"likert_eval_{self.evaluator_model}_{self.save_key}.csv",
            )

    def read_bazaar_summaries(self) -> List[Dict[str, str]]:
        rows = []
        for experiment_path in glob.glob(
            os.path.join(self.experiment_root, self.experiment_name + "*/")
        ):
            summary_path = os.path.join(experiment_path, "Logs", "bazaar_summary.json")
            config_path = os.path.join(
                experiment_path, "Configurations", "train_config.yml"
            )
            if not os.path.exists(summary_path) or not os.path.exists(config_path):
                continue
            config = yaml.safe_load(open(config_path, "r"))
            summary = load_dict(summary_path)
            for buyer_agent_summary in summary["buyer_agents"]:
                to_append = {}
                to_append["experiment_name"] = experiment_path
                to_append["seed"] = config["rng_seed"]
                to_append["agent_model"] = config["llm_name"]
                to_append["question"] = buyer_agent_summary["principal"]["query"][
                    "text"
                ]
                to_append["question_type"] = buyer_agent_summary["principal"][
                    "query"
                ].get("type", "general")
                credit_left = buyer_agent_summary["credit"]
                budget = buyer_agent_summary["principal"]["query"]["max_budget"]
                to_append["credit_spent"] = budget - credit_left
                to_append["budget"] = budget
                to_append["gold_block"] = buyer_agent_summary["principal"]["query"][
                    "gold_block"
                ]["content"]
                to_append["answer"] = buyer_agent_summary["principal"]["answer"]["text"]
                to_append["answer_type"] = config.get("run_type", "retrieve")
                to_append["num_answer_blocks"] = len(
                    buyer_agent_summary["principal"]["answer"]["blocks"]
                )
                to_append["potato_text"] = create_html_text(
                    question=to_append["question"],
                    gold_block_content=to_append["gold_block"],
                    answer=to_append["answer"],
                    llm_name=to_append["agent_model"],
                    answer_type=to_append["answer_type"],
                )
                rows.append(to_append)
        return rows

    def evaluate_likert_score_for_row(
        self, row: Dict[str, Any], inplace: bool = False
    ) -> Dict[str, str]:
        evaluated_answers = evaluate_answer_with_likert(
            question=row["question"],
            gold_block=row["gold_block"],
            answer=row["answer"],
            model_name=self.evaluator_model,
        )
        allowed_keys = {
            "comprehensiveness",
            "correctness",
            "simplicity",
            "relevance",
            "overall_quality",
        }
        answer_filtered = {
            f"likert_{k}": v for k, v in evaluated_answers.items() if k in allowed_keys
        }
        if inplace:
            row.update(answer_filtered)
            row.update({"evaluator_model": self.evaluator_model})
        return answer_filtered

    def run(self):
        # We do all the things here
        bazaar_summaries = self.read_bazaar_summaries()
        for row in tqdm(bazaar_summaries):
            self.evaluate_likert_score_for_row(row, inplace=True)
        df = pd.DataFrame.from_dict(bazaar_summaries)
        df.to_csv(self.result_output_path)


def main(args: Optional[argparse.Namespace] = None):
    if args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--experiment_root", type=str)
        parser.add_argument("--experiment_name", type=str)
        parser.add_argument("--evaluator_model", type=str)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--save_key", type=str, default="")
        args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("Loading evaluator...")
    evaluator = LikertEvaluator(
        experiment_root=args.experiment_root,
        experiment_name=args.experiment_name,
        evaluator_model=args.evaluator_model,
    )
    print("Running evaluation...")
    evaluator.run()
    # Done
    print("Done.")


if __name__ == "__main__":
    main()
