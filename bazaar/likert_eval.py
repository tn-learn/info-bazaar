import glob
import os
import argparse
import ast
import random
import traceback

import yaml
from tqdm import tqdm
from typing import Optional, Dict, List, Any
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from bazaar.py_utils import dump_dict, load_dict
from bazaar.lem_utils import (
    evaluate_answer_with_likert,
    evaluate_answer_with_likert_and_debate,
)


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
        evaluator_function_key: str = "likert_eval",
        auto_glob: bool = True,
        save_in_root: bool = True,
        num_threads: Optional[int] = None,
    ):
        assert experiment_root is not None, "experiment_root must be specified."
        assert experiment_name is not None, "experiment_name must be specified."
        assert evaluator_model is not None, "evaluator_model must be specified."
        self.experiment_root = experiment_root
        self.experiment_name = experiment_name
        self.evaluator_model = evaluator_model
        self.evaluator_function_key = evaluator_function_key
        self.auto_glob = auto_glob
        self.save_in_root = save_in_root
        self.num_threads = num_threads

    def get_result_output_path(self, mkdir: bool = True) -> str:
        if self.auto_glob and "*" not in self.experiment_name:
            experiment_name = f"{self.experiment_name}*"
        else:
            experiment_name = self.experiment_name
        experiment_name = experiment_name.replace("*", "STAR")

        if not self.save_in_root:
            path = os.path.join(
                self.experiment_root,
                experiment_name,
                "Logs",
                f"{self.evaluator_function_key}_{self.evaluator_model}.csv",
            )
        else:
            path = os.path.join(
                self.experiment_root,
                f"{self.evaluator_function_key}_{self.evaluator_model}_{experiment_name}.csv",
            )
        # Create the directory if it doesn't exist
        if mkdir:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def get_experiment_paths(self) -> List[str]:
        # Figure out which experiments we're dealing with
        if self.auto_glob and "*" not in self.experiment_name:
            # Glob all experiments
            experiment_paths = glob.glob(
                os.path.join(self.experiment_root, f"{self.experiment_name}*")
            )
        elif not self.auto_glob and "*" not in self.experiment_name:
            # Single experiment
            experiment_paths = [
                os.path.join(self.experiment_root, self.experiment_name)
            ]
        elif "*" in self.experiment_name:
            # Glob all experiments
            experiment_paths = glob.glob(
                os.path.join(self.experiment_root, self.experiment_name)
            )
        else:
            raise ValueError(f"Invalid experiment name: {self.experiment_name}")
        return experiment_paths

    def read_bazaar_summaries(self) -> List[Dict[str, str]]:
        rows = []
        for experiment_path in tqdm(self.get_experiment_paths()):
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
                ].get("question_type", "general")
                credit_left = buyer_agent_summary["credit"]
                budget = buyer_agent_summary["principal"]["query"]["max_budget"]
                to_append["credit_spent"] = budget - credit_left
                to_append["budget"] = budget
                to_append["gold_block"] = buyer_agent_summary["principal"]["query"][
                    "gold_block"
                ]["content"]
                if buyer_agent_summary["principal"]["answer"] is None:
                    buyer_agent_summary["principal"]["answer"] = {"text": None, "blocks": []}
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

    @staticmethod
    def _safety_wrap(fn, *args, **kwargs) -> Dict[str, Any]:
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            print("-----")
            print("Ignoring exception:")
            exception_string = traceback.format_exc()
            print(exception_string)
            print("-----")
            return {}

    def evaluate_likert_score_for_row(
        self, row: Dict[str, Any], inplace: bool = False
    ) -> Dict[str, str]:
        if self.evaluator_function_key == "likert_eval":
            evaluated_answers = self._safety_wrap(
                evaluate_answer_with_likert,
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
        elif self.evaluator_function_key == "likert_debate_eval":
            evaluated_answers = self._safety_wrap(
                evaluate_answer_with_likert_and_debate,
                question=row["question"],
                gold_block=row["gold_block"],
                answer=row["answer"],
                bulletize_gold_block=True,
                model_name=self.evaluator_model,
            )
            allowed_keys = {
                "fluency",
                "relevance",
                "correctness",
            }
        else:
            raise NotImplementedError
        answer_filtered = {
            f"likert_{k}": v for k, v in evaluated_answers.items() if k in allowed_keys
        }
        for key in allowed_keys:
            if f"likert_{key}" not in answer_filtered:
                answer_filtered[f"likert_{key}"] = np.nan
        if inplace:
            row.update(answer_filtered)
            row.update({"evaluator_model": self.evaluator_model})
        else:
            row = {**row, **answer_filtered, "evaluator_model": self.evaluator_model}
        return row

    def run(self):
        # We do all the things here
        bazaar_summaries = self.read_bazaar_summaries()
        if self.num_threads in [None, 0]:
            for row in tqdm(bazaar_summaries):
                self.evaluate_likert_score_for_row(row, inplace=True)
        else:
            # Use executor map to run the evaluate_likert_score_for_row
            # function in parallel
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                bazaar_summaries = list(
                    tqdm(
                        executor.map(
                            self.evaluate_likert_score_for_row, bazaar_summaries
                        ),
                        total=len(bazaar_summaries),
                    )
                )
        df = pd.DataFrame.from_dict(bazaar_summaries)
        df.to_csv(self.get_result_output_path())

    def dry_run(self):
        print("Experiment paths that would have been processed:")
        for experiment_path in self.get_experiment_paths():
            print(experiment_path)
        print("-" * 80)
        print("Would have dumped results to:", self.get_result_output_path(False))


def main(args: Optional[argparse.Namespace] = None):
    if args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--experiment_root", type=str)
        parser.add_argument("--experiment_name", type=str)
        parser.add_argument("--evaluator_model", type=str)
        parser.add_argument("--evaluator_function_key", type=str, default="likert_eval")
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--no_auto_glob", action="store_true", default=False)
        parser.add_argument("--no_save_in_root", action="store_true", default=False)
        parser.add_argument("--dry_run", action="store_true", default=False)
        parser.add_argument("--num_threads", type=int, default=None)
        args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("Loading evaluator...")
    evaluator = LikertEvaluator(
        experiment_root=args.experiment_root,
        experiment_name=args.experiment_name,
        evaluator_model=args.evaluator_model,
        evaluator_function_key=args.evaluator_function_key,
        auto_glob=(not args.no_auto_glob),
        save_in_root=(not args.no_save_in_root),
        num_threads=args.num_threads,
    )

    print("Running evaluation...")
    if args.dry_run:
        evaluator.dry_run()
    else:
        evaluator.run()
    # Done
    print("Done.")


if __name__ == "__main__":
    main()
