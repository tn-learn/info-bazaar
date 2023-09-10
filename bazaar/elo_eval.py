import itertools
import json
import os
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from bazaar.lem_utils import (
    evaluate_answer_with_debate_retrieved_closed,
    get_closed_book_answer,
)
from bazaar.py_utils import dump_dict
from bazaar.eval import (
    AnswerQuality,
    EvaluationResult,
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

def process_b(args_tuple):
    b, experiment_name, seed, llm_name = args_tuple

    closed_book_answer = get_closed_book_answer(
        question=b["principal"]["query"]["text"], model_name=llm_name,
    )
    flattened = {
        "experiment_name": experiment_name,
        "seed": seed,
        "closed_book_answer": closed_book_answer,
        "llm_name": llm_name,
        "budget_used": b["max_budget"] - b["credit_left"],
        "num_blocks": len(b["principal"]["answer"]["blocks"]),
    }

    for k, v in b.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                flattened[f"{k}/{sub_k}"] = sub_v
        else:
            flattened[k] = v
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
        rows = defaultdict(list)
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
                for b in summary["buyer_agents"]:
                    model_name = config["llm_name"]
                    question = b["principal"]["query"]["text"]
                    try:
                        if b["principal"]["answer"]["success"]:
                            del b['rejected_quotes']
                            rows[question].append((b, experiment_name, seed, model_name))
                    except (KeyError, TypeError):
                        continue
        flattened_rows = [item for sublist in rows.values() for item in sublist]
        breakpoint()
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            results = list(tqdm(executor.map(process_b, flattened_rows), total=len(flattened_rows)))
        self.df = pd.DataFrame(results)
        self.df.to_csv(self.output_path)
        print(f"Saved to {self.output_path}")

    def run(self, eval_with_model: str) -> EvaluationResult:
        pairs = []
        for cols, group in tqdm(self.df.groupby("question_text")):
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
        dump_dict(all_answer_qualities, f"{self.exp_root}/evaluation.json")
        return all_answer_qualities

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
    print("Done.")


if __name__ == "__main__":
    main()
