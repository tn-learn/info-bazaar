import os
import re
import json
import time
import openai
import tiktoken
from tqdm import tqdm
from collections import defaultdict

data_root = "/Users/martinweiss/PycharmProjects/tn-learn/info-bazaar/data"
dataset = json.load(open(os.path.join(data_root, "dataset.json"), "r"))
openai.api_key = os.environ.get("OPENAI_API_KEY")


def remove_invalid_escapes(input_string):
    # Define a regular expression pattern to match invalid escape sequences
    invalid_escape_pattern = r'\\(?!\\|/|[bfnrtvN])'

    # Use re.sub to replace invalid escape sequences with an empty string
    cleaned_string = re.sub(invalid_escape_pattern, '', input_string)

    return cleaned_string



def extract_nuggets(paper):
    functions = [
        {
            "name": "return_qa_pairs",
            "description": "Returns a list of question and answer pairs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "qa_pairs": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "question": {
                                    "type": "string",
                                    "description": "A plain-text question (no latex or markdown)",
                                },
                                "answer": {
                                    "type": "string",
                                    "description": "A plain-text answer (no latex or markdown)",
                                },
                            },
                            "required": ["question", "answer"],
                        },
                    },
                },
                "required": ["qa_pairs"],
            },
        }
    ]
    nugget_prompt = """
    You are Question-Answer-GPT, and you read scientific texts to extract questions and answers. Each answer must be a factual and based on the text's content. Each question should be answered by its corresponding factual statement. Factual statements about the world are objective assertions that describe reality and can be empirically verified or supported by evidence. They can and often should be multiple sentences long to provide sufficient context.  Answers should read like an excerpt from wikipedia.

    Your function call must follow these rules:
    1. Only write about factual statements.
    2. Do not refer to the provided text - your questions and answers should contain no phrases like "in this paper", or "findings are studied". Your questions and answers should be answerable in a world where this paper was never published.
    3. Do not write any citations.
    4. Do not refer to any published works.
    5. Do not include LaTeX - only plain text.    
    """
    intro = paper.split("\section{Introduction}")[1].split("\section")[0]
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    num_tokens = len(encoding.encode(intro))
    while num_tokens > 1500:
        intro = intro[:int(len(intro) * 0.8)]
        num_tokens = len(encoding.encode(intro))

    messages = [{"role": "system", "content": nugget_prompt}, {"role": "user", "content": intro}]
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.0,
            max_tokens=2048,
            functions=functions,
            function_call={"name": "return_qa_pairs"},
        )
        # content = response.choices[0]["message"]["content"]
        content = remove_invalid_escapes(response['choices'][0]['message']['function_call']['arguments'])
        nuggets = json.loads(content)['qa_pairs']
        breakpoint()
    except Exception as e:
        print(e)
        nuggets = []
    return nuggets

def embed_nuggets(nuggets):
    EMBEDDING_MODEL = "text-embedding-ada-002"  # OpenAI's best embeddings as of Apr 2023
    BATCH_SIZE = 1000  # you can submit up to 2048 embedding inputs per request
    nugget_questions = [nugget['question'] for nugget in nuggets]
    nugget_answers = [nugget['answer'] for nugget in nuggets]
    embeddings = []
    for batch_start in range(0, len(nuggets), BATCH_SIZE):
        batch_end = batch_start + BATCH_SIZE
        batch = nugget_answers[batch_start:batch_end]
        print(f"Batch {batch_start} to {batch_end - 1}")
        response = openai.Embedding.create(model=EMBEDDING_MODEL, input=batch)
        for i, be in enumerate(response["data"]):
            assert i == be["index"]  # double check embeddings are in same order as input
        batch_embeddings = [e["embedding"] for e in response["data"]]
        embeddings.extend(batch_embeddings)
    embeddings = [list(embedding) for embedding in embeddings]
    return {"nugget_questions": nugget_questions, "nugget_answers": nugget_answers, "embedding": embeddings}

def vendor_dataset_mapping():
    vendors = defaultdict(list)
    id_to_vendor = {}
    for arxiv_id, paper_info in dataset.items():
        authors = paper_info["authorships"]
        for author in authors:
            for affiliation in author["institutions"]:
                affiliation_id = affiliation["id"].replace("https://openalex.org/", "")
                affiliation_name = affiliation["display_name"]
                vendors[affiliation_id].append(arxiv_id)
                if affiliation_id not in id_to_vendor:
                    id_to_vendor[affiliation_id] = affiliation_name
    vendors = {
        vendor_id: dict(papers=papers, display_name=id_to_vendor[vendor_id])
        for vendor_id, papers in vendors.items()
    }
    return vendors


if __name__ == "__main__":
    paper_samples = json.load(
        open(
            os.path.join(data_root, "paper_samples_concept_0.4_n_100_weighting_50_inst_50_conc.json"),
            "r",
        )
    )[0]

    dataset_w_nuggets = {}
    for blob in tqdm(dataset.values()):
        paper = blob["paper"]

        if "\section{Introduction}" not in paper:
            continue
        if blob['arxiv_id'] in paper_samples:
            nuggets = extract_nuggets(paper)
            if len(nuggets) == 0:
                continue
            embedding_dict = embed_nuggets(nuggets)
            blob["embedding_df"] = embedding_dict
            blob["vendor_id"] = blob["authorships"][0]["institutions"][0]["id"].replace(
                "https://openalex.org/", ""
            )
            dataset_w_nuggets[blob["arxiv_id"]] = blob
            json.dump(dataset_w_nuggets, open(os.path.join(data_root, "dataset_with_nuggets.json"), "w"))
            time.sleep(10)
            if len(dataset_w_nuggets) == 100:
                break

