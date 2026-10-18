import os
import shutil
import random
import argparse
import re
import guidance
from tqdm import tqdm
import hashlib
import itertools
from bazaar.lem_utils import get_guidance_cache_directory, get_llm
import json
import backoff
import pandas as pd
from collections import defaultdict
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bazaar.lem_utils import rephrase_passage, select_quotes_cot, select_quotes_direct

from bazaar.lem_utils import OAI_EXCEPTIONS
from typing import List, SupportsFloat, Optional
from bazaar.lem_utils import clean_program_string, select_quotes_with_debate, ask_for_guidance
from bazaar.schema import Quote
from bazaar.py_utils import dataclass_from_dict
from bazaar.schema import Query
from bazaar.schema import Block
import copy

class IssuedBy:
    def __init__(self, unique_id):
        self.unique_id = unique_id

def generate_hash(exp_params):
    serialized_params = json.dumps(exp_params, sort_keys=True)
    return hashlib.sha256(serialized_params.encode()).hexdigest()

def main(output_dir: str, base_price: int, higher_price: int, num_questions: int, rephrase_both: bool, name1: str, name2: str, quote_fns: List[str], rephraser_model: str):
    summary = json.load(open("/network/scratch/w/weissmar/tn/info-bazaar/experiments/fup-stackexchange-specific-Llama-2-7b-100q/Logs/bazaar_summary.json", "r"))
    # summary = json.load(open("/network/scratch/w/weissmar/tn/info-bazaar/experiments/fup-specific-gpt-4-4.605-retrieve/Logs/bazaar_summary.json", "r"))
    # summary = json.load(open("/network/scratch/w/weissmar/tn/info-bazaar/experiments/fup-general-gpt-4-4.605-retrieve/Logs/bazaar_summary.json", "r"))
    dataset = json.load(open("/network/scratch/w/weissmar/tn/info-bazaar/data/final_dataset_with_metadata.json", "r"))

    all_candidates = []
    for buyer_idx in range(len(summary['buyer_agents'])):
        buyer = summary['buyer_agents'][buyer_idx]
        candidate_quotes = []
        for quote in buyer['accepted_quotes']:
            if quote['query']['text'] != buyer['principal']['query']['text']:
                continue
            if quote['answer_blocks'][0]['block_id'] == quote['query']['gold_block']['block_id']:
                continue
            quote = copy.deepcopy(quote)
            quote['query']['required_by_time'] = None
            quote['query']['issued_by'] = IssuedBy(quote['query']['issued_by'])
            quote['issued_by'] = IssuedBy(quote['issued_by'])
            block_dict = quote['query']['gold_block']
            block_id = block_dict['block_id']
            document_id, section_title, token_start, token_end = block_id.split("/")
            document_title = dataset[document_id]['metadata']['title']
            publication_date = dataset[document_id]['metadata']['publication_date']
            block_dict['document_id'] = document_id
            block_dict['section_title'] = section_title
            block_dict['token_start'] = token_start
            block_dict['token_end'] = token_end
            block_dict['document_title'] = document_title
            block_dict['publication_date'] = publication_date
            quote['query']['_gold_block'] = dataclass_from_dict(Block, block_dict)
            quote['query'] = dataclass_from_dict(Query, quote['query'])
            block_dict = quote['answer_blocks'][0]
            block_id = block_dict['block_id']
            document_id, section_title, token_start, token_end = block_id.split("/")
            document_title = dataset[document_id]['metadata']['title']
            publication_date = dataset[document_id]['metadata']['publication_date']
            block_dict['document_id'] = document_id
            block_dict['section_title'] = section_title
            block_dict['token_start'] = token_start
            block_dict['token_end'] = token_end
            block_dict['document_title'] = document_title
            block_dict['publication_date'] = publication_date
            quote['answer_blocks'][0] = dataclass_from_dict(Block, block_dict)
            quote = dataclass_from_dict(Quote, quote)
            candidate_quotes.append(quote)
    
        for quote in buyer['rejected_quotes']:
            if quote['query']['text'] != buyer['principal']['query']['text']:
                continue
            if quote['answer_blocks'][0]['block_id'] == quote['query']['gold_block']['block_id']:
                continue
    
            if quote['quote_progression'] == 3:
                quote = copy.deepcopy(quote)
                quote['query']['required_by_time'] = None
                quote['query']['issued_by'] = IssuedBy(quote['query']['issued_by'])
                quote['issued_by'] = IssuedBy(quote['issued_by'])
                block_dict = quote['query']['gold_block']
                block_id = block_dict['block_id']
                document_id, section_title, token_start, token_end = block_id.split("/")
                document_title = dataset[document_id]['metadata']['title']
                publication_date = dataset[document_id]['metadata']['publication_date']
                block_dict['document_id'] = document_id
                block_dict['section_title'] = section_title
                block_dict['token_start'] = token_start
                block_dict['token_end'] = token_end
                block_dict['document_title'] = document_title
                block_dict['publication_date'] = publication_date
                quote['query']['_gold_block'] = dataclass_from_dict(Block, block_dict)
                quote['query'] = dataclass_from_dict(Query, quote['query'])
                block_dict = quote['answer_blocks'][0]
                block_id = block_dict['block_id']
                document_id, section_title, token_start, token_end = block_id.split("/")
                document_title = dataset[document_id]['metadata']['title']
                publication_date = dataset[document_id]['metadata']['publication_date']
                block_dict['document_id'] = document_id
                block_dict['section_title'] = section_title
                block_dict['token_start'] = token_start
                block_dict['token_end'] = token_end
                block_dict['document_title'] = document_title
                block_dict['publication_date'] = publication_date
                quote['answer_blocks'][0] = dataclass_from_dict(Block, block_dict)
                quote = dataclass_from_dict(Quote, quote)
                candidate_quotes.append(quote)
    
        random.shuffle(candidate_quotes)
        if len(candidate_quotes) >= 3:
            all_candidates.append(candidate_quotes[:3])
    all_candidates = all_candidates[:num_questions]


    rational_choice_results = []    
    models = ["Llama-2-70b-chat-hf"]#, "gpt-3.5-turbo"]#, "gpt-4"] # "Llama-2-7b-chat-hf",
    exp_types = ["different_price"]#, "same_price"]
    all_fns = {"debate": select_quotes_with_debate, "direct": select_quotes_direct, "cot": select_quotes_cot}
    select_quotes_fns = {k: v for k, v in all_fns.items() if k in quote_fns}
    reversed_vals = [False] #[False, True]
    all_combinations = list(itertools.product(exp_types, select_quotes_fns.items(), models, reversed_vals))
    
    for exp_type, (select_quotes_name, select_quotes_fn), model_name, is_reversed in tqdm(all_combinations):
        for candidate_quotes in all_candidates:
            quote1 = copy.deepcopy(candidate_quotes[0])
            quote2 = copy.deepcopy(candidate_quotes[0])
            quote1.price = base_price
            quote2.price = base_price
    
            exp_result = {"exp_type": exp_type, "selection_mechanism": select_quotes_name, "model": model_name,
                          "query_text": quote1.query.text, "is_reversed": is_reversed}
            print(f"running: {exp_result}")
    
            block1 = copy.deepcopy(quote1.query._gold_block)
            block2 = copy.deepcopy(quote1.query._gold_block)
            try:
                if rephrase_both:
                    answer1 = rephrase_passage(quote1.query._gold_block.content, model=rephraser_model, caching=False)
                    block1.content = answer1
                else:
                    pass # block1.content is gold content in this path
                answer2 = rephrase_passage(quote2.query._gold_block.content, model=rephraser_model, caching=False)
                block2.content = answer2
    
                quote1.answer_blocks = [block1]
                quote2.answer_blocks = [block2]
        
                if exp_type == "different_price":
                    quote1.price = higher_price
        
                if is_reversed:
                    quote_perm = [quote1, quote2]
                else:
                    quote_perm = [quote2, quote1]
                result, program_output = select_quotes_fn(quotes=quote_perm, budget=100.0,
                                          model_name=model_name, use_block_content_metadata=False,
                                          use_block_metadata_only=False, return_program_output=True, caching=False)
            except Exception as e:
                print(e)
                continue
            exp_result["program_output"] = program_output['answer']
    
            if len(result) == 2:
                exp_result["choice"] = "both"
            elif len(result) == 0:
                exp_result["choice"] = "neither"
            elif (result[0] == quote_perm[0] or result[0] == quote_perm[1]) and exp_type == "same_price":
                exp_result["choice"] = "equivalent_one"
            elif result[0].price == base_price and exp_type == "different_price":
                exp_result["choice"] = "cheaper_one"
            elif result[0].price == higher_price and exp_type == "different_price":
                exp_result["choice"] = "pricey_one"
            rational_choice_results.append(exp_result)
            print(rational_choice_results)
        rational_df = pd.DataFrame(rational_choice_results)
        print(rational_df)
        filename = f"{name1}-{name2}-{base_price}-{higher_price}-rephrase_both-{rephrase_both}-rephraser-{rephraser_model}-reversed_vals-{reversed_vals}-nq-{num_questions}.csv"
        path = os.path.join(output_dir, filename)
        print(f"saving to {path}")
        rational_df.to_csv(path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Experiment Settings')
    parser.add_argument('--output_dir', type=str, default="rational_choice_results", help='Output CSV filename')
    parser.add_argument('--base_price', type=float, default=10.0, help='Base price for the quotes')
    parser.add_argument('--higher_price', type=float, default=15.0, help='Higher price for the quotes')
    parser.add_argument('--num_questions', type=int, default=10, help='Number of questions to run')
    parser.add_argument('--name1', type=str, default="Michael", help='Who is debating')
    parser.add_argument('--name2', type=str, default="Bobby", help='Who is debating')
    parser.add_argument('--rephrase_both', type=bool, default=True, help='True -> Rephrase Both; False -> one is a gold block and the other is rephrased.')
    parser.add_argument('--rephraser_model', type=str, default="gpt-3.5-turbo", help='What model is doing the rephrasing.')
    parser.add_argument('--quote_fns', type=str, nargs='+', help='What method are we using to select between quotes')
    

    args = parser.parse_args()
    
    main(output_dir=args.output_dir, base_price=args.base_price, higher_price=args.higher_price, num_questions=args.num_questions, rephrase_both=args.rephrase_both, name1=args.name1, name2=args.name2, quote_fns=args.quote_fns, rephraser_model=args.rephraser_model)
