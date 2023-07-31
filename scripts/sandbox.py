import os
import re
import json
import time
from pathlib import Path
from TexSoup import TexSoup
from pylatexenc.latex2text import LatexNodes2Text


import openai
import tiktoken
from tqdm import tqdm
from collections import defaultdict

if Path("~").expanduser().name == "nrahaman":
    data_root = "/Users/nrahaman/Python/info-bazaar/data"
else:
    data_root = "/Users/martinweiss/PycharmProjects/tn-learn/info-bazaar/data"
    openai.api_key = os.environ.get("OPENAI_API_KEY")

dataset = json.load(open(os.path.join(data_root, "dataset.json"), "r"))


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
    breakpoint()
    for arxiv_id in paper_samples:
        nuggets = extract_nuggets(paper)
        if len(nuggets) == 0:
            continue
        embedding_dict = embed_nuggets(nuggets)
        blob["embedding_dict"] = embedding_dict
        blob["vendor_id"] = blob["authorships"][0]["institutions"][0]["id"].replace(
            "https://openalex.org/", ""
        )
        dataset_w_nuggets[blob["arxiv_id"]] = blob
        json.dump(
            dataset_w_nuggets,
            open(os.path.join(data_root, "dataset_with_nuggets.json"), "w"),
        )
        time.sleep(10)
        if len(dataset_w_nuggets) == 100:
            break
