import os
import shutil
import random
import re
import guidance
from tqdm import tqdm
import hashlib
import argparse
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
os.environ["OPENAI_API_KEY"] = "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"

class IssuedBy:
    def __init__(self, unique_id):
        self.unique_id = unique_id
summary = json.load(open("/network/scratch/w/weissmar/tn/info-bazaar/experiments/fup-stackexchange-specific-Llama-2-7b-100q/Logs/bazaar_summary.json", "r"))
# summary = json.load(open("/Users/martinweiss/PycharmProjects/tn-learn/info-bazaar/experiments/fup-specific-gpt-4-4.605-retrieve/Logs/bazaar_summary.json", "r"))
dataset = json.load(open("/network/scratch/w/weissmar/tn/info-bazaar/data/final_dataset_with_metadata.json", "r"))

def get_or_rephrase_answer(quote, rephrased_answers):
    gold_block_content = quote.query._gold_block.content
    if gold_block_content not in rephrased_answers:
        rephrased_answer = rephrase_passage(gold_block_content, model="gpt-4")
        rephrased_answers[gold_block_content] = rephrased_answer
    return rephrased_answers[gold_block_content]

def prepare_quotes_gold_and_rephrased(quote, rephrased_answer=None):
    # Deep copy original quote for gold and rephrased versions
    gold_quote = copy.deepcopy(quote)
    
    # Prepare gold quote
    gold_block = copy.deepcopy(quote.query._gold_block)
    gold_quote.answer_blocks = [gold_block]
    
    # Prepare rephrased quote
    if rephrased_answer is not None:
        rephrased_quote = copy.deepcopy(quote)
        rephrased_block = copy.deepcopy(gold_block)
        rephrased_block.content = rephrased_answer
        rephrased_quote.answer_blocks = [rephrased_block]
        return gold_quote, rephrased_quote
    else:
        return gold_quote

def generate_hash(exp_params):
    serialized_params = json.dumps(exp_params, sort_keys=True)
    return hashlib.sha256(serialized_params.encode()).hexdigest()

def main(output_dir: str, prices: List[float], num_questions: int):
    printable_model_names = {"Llama-2-70b-chat-hf": 'Llama 2 (70B)', "Llama-2-7b-chat-hf": 'Llama 2 (7B)', "gpt-3.5-turbo": 'GPT-3.5', "gpt-4": "GPT-4" }

    all_candidates = []
    for buyer_idx in range(0, 30):
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
    
    models = ["Llama-2-70b-chat-hf", "gpt-3.5-turbo",  "gpt-4"] #, "Llama-2-7b-chat-hf"]
    metadata_options = [False, True]
    single_good_demand_curve = []
    all_combinations = list(itertools.product(prices, models, all_candidates[:num_questions], metadata_options))
    for price, model_name, candidate_quotes, use_block_metadata_only in all_combinations:
        single_good_demand_curve.append({"price": price, "model": model_name, "printable_model_name": printable_model_names[model_name], "query_text": candidate_quotes[0].query.text, "use_block_metadata_only": use_block_metadata_only, "quote": candidate_quotes[0], "choice": None, "debate": None})
    single_good_demand_curve_df = pd.DataFrame(single_good_demand_curve)
    for idx, row in tqdm(single_good_demand_curve_df.iterrows(), total=len(single_good_demand_curve_df)):
        if row.choice != None:
            continue
    
        quote = copy.deepcopy(row.quote)
        quote.price = row.price
        block = copy.deepcopy(quote.query._gold_block)
        quote.answer_blocks = [block]
    
        try:
            result_choice, result_debate = select_quotes_with_debate(quotes=[quote], budget=100, model_name=row.model, use_block_content_metadata=False, use_block_metadata_only=row.use_block_metadata_only, return_program_output=True)
            
            print(f"{idx}-row.use_block_metadata_only-{row.use_block_metadata_only}-{result_choice}-{result_debate}")
            print("----------")
        except Exception as e:
            print(e)
            result = [None]
    
        if len(result_choice) == 1:
            choice = "buy"
        elif len(result_choice) == 0:
            choice = "pass"
        single_good_demand_curve_df.loc[idx, 'choice'] = choice
        single_good_demand_curve_df.loc[idx, 'debate'] = result_debate['answer']
        print(single_good_demand_curve_df)
        filename = f"prices-{'-'.join([str(x) for x in prices])}-qs-{num_questions}.csv"
        path = os.path.join(output_dir, filename)
        print(f"saving to {path}")
        single_good_demand_curve_df.to_csv(path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Experiment Settings')
    parser.add_argument('--output_dir', type=str, default="rational_choice_results", help='Output CSV filename')
    parser.add_argument('--num_questions', type=int, default=10, help='Number of questions to run')
    parser.add_argument('--prices', type=str, nargs='+', help='What prices to use')
    args = parser.parse_args()
    prices = [float(x) for x in args.prices]

    main(output_dir=args.output_dir, prices=prices, num_questions=args.num_questions)
