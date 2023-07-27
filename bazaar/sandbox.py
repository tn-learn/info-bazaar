import os
import json
from collections import defaultdict
import pandas as pd
import openai
from tqdm import tqdm
from thefuzz import fuzz

data_path = "/Users/nrahaman/Python/info-bazaar/data/dataset.json"
dataset = json.load(open(data_path, "r"))

# Set the API key
openai.api_key = os.environ.get("OPENAI_API_KEY")


def extract_nuggets(paper):
    nugget_prompt = """
    You will be given some scientific text in LaTeX. Your task is to extract from the text a list of valuable nuggets of information about Nature. For example, "A dynamical property of dark energy is the decay of large-scale gravitational potentials" is a valuable nugget. A bad nugget is "In this paper, we test these claims using yet another type of BOSS DR12 void catalogue". The reason the latter is not a valuable nugget is that it is primarily describing a method of study, not a fact about Nature. Each nugget should be self-contained, i.e. there should be no demonstrative pronouns, like "these" and "those". Any demonstrative pronoun should be replaced with a noun phrase, such that it is self-contained. If a nugget is written in such a way that it references other nuggets, then it should be rewritten as a standalone nugget. In other words, nuggets should not refer to each other. Do not emit any escape characters in the string that could cause the JSON to fail to parse.
    
    It is very important for you to return these nuggets as a JSON. For example:
    ```
    [
      <nugget 1>,
      <nugget 2>,
      ... <more>
    ]
    ```
    """
    intro = paper.split("\section{Introduction}")[1].split("\section")[0]
    messages = [{"role": "system", "content": nugget_prompt}, {"role": "user", "content": intro}]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.0,
        max_tokens=2048,
    )
    content = response.choices[0]["message"]["content"]
    try:
        nuggets = json.loads(content)
    except Exception as e:
        print(e)
        nuggets = []
    return nuggets

def embed_nuggets(nuggets):
    EMBEDDING_MODEL = "text-embedding-ada-002"  # OpenAI's best embeddings as of Apr 2023
    BATCH_SIZE = 1000  # you can submit up to 2048 embedding inputs per request

    embeddings = []
    for batch_start in range(0, len(nuggets), BATCH_SIZE):
        batch_end = batch_start + BATCH_SIZE
        batch = nuggets[batch_start:batch_end]
        print(f"Batch {batch_start} to {batch_end - 1}")
        response = openai.Embedding.create(model=EMBEDDING_MODEL, input=batch)
        for i, be in enumerate(response["data"]):
            assert i == be["index"]  # double check embeddings are in same order as input
        batch_embeddings = [e["embedding"] for e in response["data"]]
        embeddings.extend(batch_embeddings)

    df = pd.DataFrame({"nugget": nuggets, "embedding": embeddings})
    return df

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
            "../data/paper_samples_concept_0.4_n_100_weighting_50_inst_50_conc.json",
            "r",
        )
    )[0]

    for blob in tqdm(dataset.values()):
        paper = blob["paper"]

        if "\section{Introduction}" not in paper:
            continue
        if blob['arxiv_id'] in paper_samples:
            nuggets = extract_nuggets(paper)
            blob["nuggets"] = nuggets
            embedding_df = embed_nuggets(blob['nuggets'])
