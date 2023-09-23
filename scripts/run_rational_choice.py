import os
import shutil
import random
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
os.environ["OPENAI_API_KEY"] = "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"
summary = json.load(open("/network/scratch/w/weissmar/tn/info-bazaar/experiments/fup-stackexchange-specific-Llama-2-7b-100q/Logs/bazaar_summary.json", "r"))
# summary = json.load(open("/Users/martinweiss/PycharmProjects/tn-learn/info-bazaar/experiments/fup-specific-gpt-4-4.605-retrieve/Logs/bazaar_summary.json", "r"))
dataset = json.load(open("/network/scratch/w/weissmar/tn/info-bazaar/data/final_dataset_with_metadata.json", "r"))

class IssuedBy:
    def __init__(self, unique_id):
        self.unique_id = unique_id

@backoff.on_exception(backoff.expo, OAI_EXCEPTIONS, max_tries=5)
def select_quotes_with_debate(
        quotes: List["Quote"],
        budget: Optional[SupportsFloat] = None,
        fraction_of_max_budget: Optional[float] = None,
        model_name: Optional[str] = None,
        use_block_content_metadata: bool = False,
        use_block_metadata_only: bool = False,
) -> List["Quote"]:
    if len(quotes) == 0:
        return []
    assert all(
        [quotes[0].query.compare_content(quote.query) for quote in quotes[1:]]
    ), "All quotes must have the same query."
    # Get the budget
    if budget is None:
        budget = quotes[0].query.max_budget
    else:
        budget = float(budget)
    if fraction_of_max_budget is not None:
        budget = round(fraction_of_max_budget * quotes[0].query.max_budget, 1)

    # We need to scale the prices. For this, we can assume that the scaled budget
    # will always be $100. The prices must be scaled accordingly.
    scale_factor = 100 / budget

    # Get the question
    question = quotes[0].query.text

    # Build the content extractor
    def content_extractor(block: "Block") -> str:
        if use_block_content_metadata:
            return block.content_with_metadata
        elif use_block_metadata_only:
            return block.metadata
        else:
            return block.content

    # Get the options
    options = [
        {
            "answer_block": " [...] ".join(
                [content_extractor(block) for block in quote.answer_blocks]
            ),
            "price": max(int(round(quote.price * scale_factor)), 1),
        }
        for quote in quotes
    ]

    program_string = """
    {{#system~}}
    Bobby William and Michael Burry are employed by a company that specializes in acquiring information. They are trying to answer a question by purchasing information from an information market. In this market, vendors sell pieces of information at a price. 

    Bobby wants to do a really good job at answering the question. This entails knowing as much as possible.

    Michael, on the other hand, is financially responsible. Michael wants to make sure ensures that they don't waste money buying unnecessary information. For instance, if two pieces of information offer the same insight, then Michael would go for the cheaper one.  
    {{~/system}}

    {{#user~}}
    The question is "{{question}}?"

    Here are your options.
    ---{{#each options}}
    Option {{add @index 1}}: {{this.answer_block}}
    {{/each}}---

    {{#each options~}}
    Option {{add @index 1}} costs ${{this.price}}
    {{/each}}
    Together, Bobby and Michael must decide which options to buy and which ones to not buy with their budget of ${{balance}}. Simulate a constructive argument between Bobby and Michael, where they debate about the usefulness of the information provided in each option towards answering the question, and whether their price is worth paying. 

    Note that Bobby and Michael may choose to buy any number of options, or none at all. At the end of the argument, they must arrive at a verdict. This verdict must be printed as: 

    VERDICT:

    {{#each options~}}
    Option {{add @index 1}}: <Buy or Pass>
    {{/each}}
    {{~/user}}

    {{#assistant~}}
    {{gen "answer" temperature=0.0 max_tokens=2048}}
    {{~/assistant}}
    """
    program_string = clean_program_string(program_string)
    print(question)

    print(options)
    # Run the program
    program_output = ask_for_guidance(
        program_string=program_string,
        llm=get_llm(model_name=model_name),
        silent=True,
        inputs=dict(question=question, options=options, balance=100, ),
        output_keys=["answer"],
    )
    answer = program_output["answer"]

    # Now parse the answer
    def extract_verdicts(s: str) -> List[bool]:
        # Split the text into sections based on "VERDICT:"
        sections = re.split(r"\bVERDICT\b\s*:\s*", s, flags=re.IGNORECASE)
        if len(sections) < 2:
            return []

        # Dictionary to store the verdicts of each option
        option_verdicts = {}
        for section in sections[1:]:
            # Extract options and their verdicts in a case-insensitive manner
            options = re.findall(
                r"Option (\d+): (Buy|Pass)", section, flags=re.IGNORECASE
            )

            for option_num, verdict in options:
                option_num = int(option_num)
                is_buy = verdict.lower() == "buy"

                # Check if this option was seen before
                if option_num in option_verdicts:
                    # If the verdict is inconsistent, raise an exception
                    if option_verdicts[option_num] != is_buy:
                        raise ValueError(
                            f"Inconsistent verdict for Option {option_num}."
                        )
                else:
                    option_verdicts[option_num] = is_buy

        # Convert the verdicts dictionary to a sorted list based on option numbers
        return [option_verdicts[num] for num in sorted(option_verdicts.keys())]

    # Parse the verdicts, select the quotes and return
    verdicts = extract_verdicts(answer)
    selected_quotes = [quote for quote, verdict in zip(quotes, verdicts) if verdict]
    return selected_quotes


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
rephrased_answers = {}
rational_choice_results = []

models = ["gpt-3.5-turbo", "gpt-4", "Llama-2-70b-chat-hf"]  # "Llama-2-7b-chat-hf",
exp_types = ["same_price", "different_price"]
select_quotes_fns = {"debate": select_quotes_with_debate, "direct": select_quotes_direct, "cot": select_quotes_cot}
reversed_vals = [True, False]

all_combinations = list(itertools.product(exp_types, select_quotes_fns.items(), models, reversed_vals))
for exp_type, (select_quotes_name, select_quotes_fn), model_name, is_reversed in tqdm(all_combinations):
    for candidate_quotes in all_candidates:
        quote = copy.deepcopy(candidate_quotes[0])

        exp_result = {"exp_type": exp_type, "selection_mechanism": select_quotes_name, "model": model_name,
                      "query_text": quote.query.text}
        print(f"running: {exp_result}")
        exp_hash = generate_hash(exp_result)

        exp_result["sample_hash"] = exp_hash
        found = False
        for sample in rational_choice_results:
            if sample.get("sample_hash") == exp_hash:
                found = True
        if found:
            continue

        rephrased_answer = get_or_rephrase_answer(quote, rephrased_answers)
        gold_quote, rephrased_quote = prepare_quotes_gold_and_rephrased(quote, rephrased_answer)
        if exp_type == "different_price":
            rephrased_quote.price = rephrased_quote.price * 1.5

        if is_reversed:
            gold_and_rephrased_quotes = [gold_quote, rephrased_quote]
        else:
            gold_and_rephrased_quotes = [rephrased_quote, gold_quote]

        result = select_quotes_fn(quotes=gold_and_rephrased_quotes, budget=quote.query.max_budget,
                                  model_name=model_name, use_block_content_metadata=False,
                                  use_block_metadata_only=False)
        guidance.llms.Transformers.cache.clear()
        guidance.llms.OpenAI.cache.clear()

        if len(result) == 2:
            exp_result["choice"] = "both"
        elif len(result) == 0:
            exp_result["choice"] = "neither"
        elif (result[0] == gold_quote or result[0] == rephrased_quote) and exp_type == "same_price":
            exp_result["choice"] = "equivalent_one"
        elif result[0] == gold_quote and exp_type == "different_price":
            exp_result["choice"] = "cheaper_one"
        elif result[0] == rephrased_quote and exp_type == "different_price":
            exp_result["choice"] = "pricey_one"

        rational_choice_results.append(exp_result)
    rational_df = pd.DataFrame(rational_choice_results)
    rational_df.to_csv("rational_choice_results.csv")

# Filter data for 'same_price' and 'different_price' experiments
same_price_df = df[df['exp_type'] == 'same_price']
different_price_df = df[df['exp_type'] == 'different_price']

# Calculate proportions for 'same_price' experiments
same_price_grouped = same_price_df.groupby(['model', 'selection_mechanism', 'choice']).size().reset_index(name='count')
same_price_pivot = same_price_grouped.pivot_table(index=['model', 'selection_mechanism'], columns='choice', values='count', fill_value=0)
same_price_pivot['total'] = same_price_pivot.sum(axis=1)
same_price_pivot['proportion_rational'] = (same_price_pivot['equivalent_one'] + same_price_pivot['neither']) / same_price_pivot['total']
same_price_proportions = same_price_pivot[['proportion_rational']].reset_index()

# Calculate proportions for 'different_price' experiments
different_price_grouped = different_price_df.groupby(['model', 'selection_mechanism', 'choice']).size().reset_index(name='count')
different_price_pivot = different_price_grouped.pivot_table(index=['model', 'selection_mechanism'], columns='choice', values='count', fill_value=0)
different_price_pivot['total'] = different_price_pivot.sum(axis=1)
different_price_pivot['proportion_rational'] = (different_price_pivot['cheaper_one'] + different_price_pivot['neither'] )/ different_price_pivot['total']
different_price_proportions = different_price_pivot[['proportion_rational']].reset_index()

# Concatenate the two tables for easier comparison
same_price_proportions['exp_type'] = 'same_price'
different_price_proportions['exp_type'] = 'different_price'
final_table = pd.concat([same_price_proportions, different_price_proportions], ignore_index=True)
final_table.to_csv("rational_proportions.csv")
