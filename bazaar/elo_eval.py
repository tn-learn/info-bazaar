import itertools
import json
import os
import argparse
from collections import defaultdict
import ast
import random
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import yaml
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


def try_literal_eval(x):
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return x


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
    allowed_keys = {"comprehensiveness", "correctness", "simplicity", "relevance"}
    answer1_filtered = {
        k: v for k, v in evaluated_answers["answer1"].items() if k in allowed_keys
    }
    answer2_filtered = {
        k: v for k, v in evaluated_answers["answer2"].items() if k in allowed_keys
    }

    answer_qualities = dict(
        answer1=AnswerQuality(**answer1_filtered),
        answer2=AnswerQuality(**answer2_filtered),
    )
    return answer_qualities


def process_b(args_tuple):
    b, experiment_name, seed, llm_name = args_tuple

    closed_book_answer = get_closed_book_answer(
        question=b["principal"]["query"]["text"], model_name=llm_name,
    )
    flattened = {
        "seed": seed,
        "experiment_name": experiment_name,
        "closed_book_answer": closed_book_answer,
        "llm_name": llm_name,
        "budget_used": b.get("max_budget", 0) - b.get("credit_left", 0),
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
    def __init__(
        self, exp_root: str, evaluator_model: str, n_jobs: int = 10, cache: bool = True
    ):
        # Publics
        self.n_jobs = n_jobs
        self.exp_root = exp_root
        self.evaluator_model = evaluator_model
        self.output_path = f"{exp_root}/elo_evaluation.csv"
        self.pairs_path = f"{self.exp_root}/all_pairs.csv"
        if os.path.exists(self.output_path) and os.path.exists(self.pairs_path) and cache:
            print(f"Loading from {self.output_path}")
            self.df = pd.read_csv(self.output_path)
            self.all_pairs_df = pd.read_csv(self.pairs_path).applymap(try_literal_eval)
            return
        else:
            print(f"Creating new dataframe")

        print(f"Loading experiments: {os.listdir(exp_root)} from {exp_root}")
        rows = defaultdict(list)
        experiments = [
            f for f in os.listdir(exp_root) if os.path.isdir(os.path.join(exp_root, f))
        ]
        for experiment_name in tqdm(experiments):
            exp_base_dir = os.path.join(exp_root, experiment_name)
            summary_path = os.path.join(exp_base_dir, "Logs", "baazar_summary.json")
            config_path = os.path.join(
                exp_base_dir, "Configurations", "train_config.yml"
            )
            try:
                with open(summary_path, "r") as f:
                    summary = json.load(f)
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
            except FileNotFoundError:
                continue
            for b in summary["buyer_agents"]:
                model_name = config["llm_name"]
                question = b["principal"]["query"]["text"]
                try:
                    if b["principal"]["answer"]["success"]:
                        del b["rejected_quotes"]
                        rows[question].append(
                            (b, experiment_name, config["rng_seed"], model_name)
                        )
                except (KeyError, TypeError):
                    continue
        flattened_rows = [item for sublist in rows.values() for item in sublist]
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            results = list(
                tqdm(executor.map(process_b, flattened_rows), total=len(flattened_rows))
            )

        # Initial DataFrame setup and transformations
        self.df = pd.DataFrame(results).applymap(try_literal_eval)
        self.df["question_text"] = self.df["principal/query"].apply(
            lambda x: x.get("text")
        )
        self.df["answer_text"] = self.df["principal/answer"].apply(
            lambda x: x.get("text")
        )
        self.df["answer_success"] = self.df["principal/answer"].apply(
            lambda x: x.get("success")
        )

        # Melt DataFrame
        self.df = pd.melt(
            self.df,
            id_vars=self.df.columns.difference(["closed_book_answer", "answer_text"]),
            value_vars=["closed_book_answer", "answer_text"],
            var_name="answer_type",
            value_name="answer",
        )

        # Generate pairs and concatenate into a new DataFrame
        pair_data = []
        for _, group in tqdm(self.df.groupby("question_text")):
            for answer1, answer2 in itertools.combinations(group.iterrows(), 2):
                pair_data.append(
                    {
                        **{"answer1_" + k: v for k, v in answer1[1].items()},
                        **{"answer2_" + k: v for k, v in answer2[1].items()},
                    }
                )

        self.all_pairs_df = pd.DataFrame(pair_data).applymap(try_literal_eval)
        self.all_pairs_df.to_csv(f"{self.exp_root}/all_pairs.csv")
        self.df.to_csv(self.output_path)
        print(f"Saved to {self.output_path}")

    def sub_sample_by_question(self, max_num_questions: int):
        questions = self.all_pairs_df["answer1_question_text"].unique()
        sampled_questions = np.random.choice(
            questions, max_num_questions, replace=False
        )
        orig_num_pairs = len(self.all_pairs_df)
        self.all_pairs_df = self.all_pairs_df[
            self.all_pairs_df["answer1_question_text"].isin(sampled_questions)
        ]
        print(
            f"Subsampled to {max_num_questions} questions. Reduced from {orig_num_pairs} to {len(self.all_pairs_df)} pairs"
        )

    def run(self, max_num_questions=-1) -> EvaluationResult:
        if max_num_questions > 0:
            self.sub_sample_by_question(max_num_questions)
        all_answer_qualities = []
        for idx, pair in tqdm(self.all_pairs_df.iterrows()):
            question_text = pair["answer1_question_text"]
            answer1 = pair["answer1_answer"]
            answer2 = pair["answer2_answer"]
            gold_block_content = pair["answer1_principal/query"]["gold_block"][
                "content"
            ]
            try:
                answer_qualities = evaluate_answer_quality_binary(
                    question_text=question_text,
                    answer1=answer1,
                    answer2=answer2,
                    gold_block_content=gold_block_content,
                    model_name=self.evaluator_model,
                )
                result = {
                    "answer_pair": dict(pair),
                    "answer1_quality": answer_qualities["answer1"].summary(),
                    "answer2_quality": answer_qualities["answer2"].summary(),
                    "evaluator_model": self.evaluator_model,
                }
                all_answer_qualities.append(result)
            except Exception:
                continue
        dump_dict(
            all_answer_qualities,
            f"{self.exp_root}/evaluation_{self.evaluator_model}.json",
        )
        return all_answer_qualities


def main(args: Optional[argparse.Namespace] = None):
    if args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--exp_root", type=str)
        parser.add_argument("--evaluator_model", type=str)
        parser.add_argument("--cache", action="store_true")
        parser.add_argument("--n_jobs", type=int, default=10)
        parser.add_argument("--max_num_questions", type=int, default=-1)
        args = parser.parse_args()
    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    print("Loading evaluator...")
    evaluator = EloEvaluator(
        exp_root=args.exp_root,
        n_jobs=args.n_jobs,
        cache=args.cache,
        evaluator_model=args.evaluator_model,
    )
    print("Running evaluation...")
    evaluation = evaluator.run(max_num_questions=args.max_num_questions)
    # Done
    print("Done.")


if __name__ == "__main__":
    main()
