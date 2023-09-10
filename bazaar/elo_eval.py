import itertools
import json
import os
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool

from bazaar.lem_utils import (
    evaluate_answer_with_debate_retrieved_closed,
    get_closed_book_answer,
)
from bazaar.py_utils import load_dict, dump_dict, root_dir_slash
from bazaar.eval import (
    AnswerQuality,
    BlockWareSpec,
    EvaluationResult,
    BuyerAgentEvaluationResult,
)


def evaluate_answer_quality_binary(
    question_text: str,
    answer1: str,
    answer2: str,
    gold_block_content: str,
    model_name: str,
) -> Tuple[Dict[str, AnswerQuality], Dict[str, str]]:
    # Evaluate the answer qualities
    evaluated_answers = evaluate_answer_with_debate_retrieved_closed(
        question=question_text,
        gold_passage=gold_block_content,
        answer1=answer1,
        answer2=answer2,
        model_name=model_name,
    )

    answer_qualities = dict(
        answer1=AnswerQuality(**evaluated_answers["answer1"]),
        answer2=AnswerQuality(**evaluated_answers["answer2"]),
    )
    return answer_qualities


def process_b(
    b: Dict, experiment_name: str, seed: str, config: Dict, num_blocks: Dict[str, int]
):
    closed_book_answer = get_closed_book_answer(
        question=b["principal"]["query"]["text"], model_name=config["llm_name"],
    )
    flattened = {
        "experiment_name": experiment_name,
        "seed": seed,
        "closed_book_answer": closed_book_answer,
        "model_name": config["llm_name"],
        "budget_used": b["max_budget"] - b["credit_left"],
    }

    for k, v in b.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                flattened[f"{k}/{sub_k}"] = sub_v
        else:
            flattened[k] = v
    flattened["num_block"] = num_blocks.get(b["name"], None)
    return flattened


class EloEvaluator:
    def __init__(self, exp_root: str, n_jobs: int = 10, cache: bool = True):
        # Publics
        self.n_jobs = n_jobs
        self.exp_root = exp_root
        self.output_path = f"{exp_root}/elo_evaluation.csv"
        if os.path.exists(self.output_path) and cache:
            print(f"Loading from {self.output_path}")
            self.df = pd.read_csv(self.output_path)
            return
        else:
            print(f"Creating new dataframe")

        print(f"Loading experiments: {os.listdir(exp_root)} from {exp_root}")
        all_data = []
        for experiment_name in os.listdir(exp_root):
            exp_base_dir = os.path.join(exp_root, experiment_name)
            seed_dirs = [
                d
                for d in os.listdir(exp_base_dir)
                if os.path.isdir(os.path.join(exp_base_dir, d))
            ]

            for seed in tqdm(seed_dirs):
                seed_dir = os.path.join(exp_base_dir, seed)
                summary_path = os.path.join(seed_dir, "bazaar_summary.json")
                config_path = os.path.join(seed_dir, "config.json")
                try:
                    with open(summary_path, "r") as f:
                        summary = json.load(f)
                    with open(config_path, "r") as f:
                        config = json.load(f)
                except FileNotFoundError:
                    continue

                num_blocks = {
                    b["principal"]["name"]: len(b["principal"]["answer"]["blocks"])
                    for b in summary["buyer_agents"]
                    if b["principal"]["answer"] is not None
                }

                with Pool(processes=self.n_jobs) as pool:
                    all_data = pool.map(
                        process_b,
                        summary["buyer_agents"],
                        itertools.repeat(experiment_name),
                        itertools.repeat(seed),
                        itertools.repeat(config),
                        itertools.repeat(num_blocks),
                    )
        self.df = pd.DataFrame(all_data)
        self.df.to_csv(self.output_path)
        print(f"Saved to {self.output_path}")

    def run(self, eval_with_model: str) -> EvaluationResult:
        def filter_group(group):
            if group["successfully_answered"].sum() >= 2:
                return group[group["successfully_answered"]]

        filtered_df = pd.concat(
            [
                filter_group(group)
                for _, group in self.df.groupby("question_text")
                if filter_group(group) is not None
            ]
        )

        pairs = []
        for cols, group in tqdm(filtered_df.groupby("question_text")):
            for pair in itertools.combinations(group.iterrows(), 2):
                pairs.append(pair)
        breakpoint()
        all_answer_qualities = []
        for pair in tqdm(pairs):
            question_text = pair[0]["question_text"]
            answer1 = pair[0]["answer"]
            answer2 = pair[1]["answer"]
            gold_block_content = pair[0]["gold_block_content"]
            answer_qualities = evaluate_answer_quality_binary(
                question_text=question_text,
                answer1=answer1,
                answer2=answer2,
                gold_block_content=gold_block_content,
                model_name=eval_with_model,
            )
            all_answer_qualities.append(answer_qualities)


def main(args: Optional[argparse.Namespace] = None):
    if args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--exp_root", type=str, default=None, help="path to run directory",
        )
        parser.add_argument("--evaluator_model", type=str, default="llama-70b")
        parser.add_argument("--cache", action="store_true")
        parser.add_argument("--n_jobs", type=int, default=10)
        args = parser.parse_args()

    print("Loading evaluator...")
    evaluator = EloEvaluator(
        exp_root=args.exp_root, n_jobs=args.n_jobs, cache=args.cache
    )
    print("Running evaluation...")
    evaluation = evaluator.run(eval_with_model=args.evaluator_model)
    # Done
    dump_dict(evaluation.summary(), Path(args.run_directory) / "elo_evaluation.json")
    print("Done.")


if __name__ == "__main__":
    main()
