import concurrent
import copy
from datetime import datetime
import requests
import pickle
from itertools import islice
from concurrent.futures import ThreadPoolExecutor
import gzip
import tarfile
import os
import re
import json
import time
from pathlib import Path
from pylatexenc.latex2text import LatexNodes2Text
import openai
from typing import List
import tiktoken
from tqdm import tqdm
from bazaar.lem_utils import (
    clean_content,
    extract_questions,
    split_to_paragraphs,
)
from bazaar.schema import Block

EMBEDDING_MODEL = "text-embedding-ada-002"

if Path("~").expanduser().name == "nrahaman":
    data_root = "/Users/nrahaman/Python/info-bazaar/data"
else:
    # data_root = "/Users/martinweiss/PycharmProjects/tn-learn/info-bazaar/data"
    # data_root = "/home/mila/w/weissmar/scratch/tn/info-bazaar/data"
    data_root = "/Users/martinweiss/PycharmProjects/tn-learn/info-bazaar/data"
    openai.api_key = os.environ.get("OPENAI_API_KEY")


def chunks(data, size=50):
    it = iter(data)
    for i in range(0, len(data), size):
        yield {k: data[k] for k in islice(it, size)}


def fetch_works(chunk):
    dois = "|".join(
        metadata.get("doi") for metadata in chunk.values() if metadata.get("doi")
    )
    time.sleep(0.25)
    result = requests.get(
        f"https://api.openalex.org/works?mailto=weissmar@mila.quebec&filter=doi:{dois}"
    )
    return result


def download_pdf(arxiv_id, category):
    url = f"https://arxiv.org/e-print/{arxiv_id}"
    response = requests.get(url, stream=True)
    time.sleep(0.025)

    if response.status_code == 200:
        with open(f"data/{category}/papers/{arxiv_id}.zip", "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
    else:
        print(f"Error downloading file, status code: {response.status_code}")


def parse_latex(latex_string: str, title: str, publication_date: str):
    # Remove comments
    rm_comments = re.sub(r"(?<!\\)%.*$", "", latex_string, flags=re.MULTILINE)

    # Remove unnecessary information (here we remove the documentclass line as an example)
    rm_unnecessary = re.sub(r"\\documentclass[^\n]*\n", "", rm_comments)

    # Split by sections
    removed_labels = re.sub(r"\\label\{.*?\}", "", rm_unnecessary)
    content_emph = re.sub(r"\\emph\{(.*?)\}", r"\1", removed_labels)
    content_emph = re.sub(r"\{\\emph (.*?)\}", r"\1", content_emph)
    content_it = re.sub(r"\\textit\{(.*?)\}", r"\1", content_emph)
    content_bf = re.sub(r"\\textbf\{(.*?)\}", r"\1", content_it)
    content_ul = re.sub(r"\\underline\{(.*?)\}", r"\1", content_bf)

    split_by_sections = re.split(
        r"(\\section\{.*?\}|\\subsection\{.*?\}|\\subsubsection\{.*?\})", content_ul
    )

    all_blocks = []
    for i in range(1, len(split_by_sections), 2):
        section_title = re.search(
            r"(\\section|\\subsection|\\subsubsection)\{(.*?)(\\label\{.*?\})?\}",
            split_by_sections[i],
        ).group(2)
        section_title = LatexNodes2Text().latex_to_text(section_title.strip())
        content = split_by_sections[i + 1]

        # Separate figures and equations from the text
        content_without_figures_equations = re.sub(
            r"(\\begin\{figure\}.*?\\end\{figure\}|\\begin\{equation\}.*?\\end\{equation\}|\[.*?\])",
            "",
            content,
            flags=re.DOTALL,
        )

        # Remove tags
        # content_without_tags = re.sub(r"\\[^{]*\{.*?\}", "", content_without_figures_equations)
        content_no_lone_newlines = re.sub(
            r"(?<!\n)\n(?!\n)", r" ", content_without_figures_equations
        )

        content_cite = re.sub(r"\\citep\{(.*?)\}", r"[\1]", content_no_lone_newlines)
        content_cite = re.sub(r"\\citet\{(.*?)\}", r"[\1]", content_cite)
        content_cite = re.sub(r"\\cite\{(.*?)\}", r"[\1]", content_cite)
        content_linebreak = re.sub(r"\\\\", " ", content_cite)

        # Split the remaining text into paragraphs
        paras = re.split(r"\n\n\s*", content_linebreak)  # Split by blank lines
        paras = [
            paragraph.strip() for paragraph in paras if paragraph.strip()
        ]  # Remove leading and trailing whitespace and ignore empty paragraphs
        paras = [
            LatexNodes2Text().latex_to_text(paragraph).strip() for paragraph in paras
        ]
        token_start = 0
        for idx, para in enumerate(paras):
            token_end = token_start + Block.num_tokens_in_content(para)
            block = Block(
                document_id=arxiv_id,
                document_title=title,
                publication_date=publication_date,
                token_start=token_start,
                token_end=token_end,
                section_title=section_title,
                content=para,
            )
            token_start = token_end
            all_blocks.append(block)
    return all_blocks


def remove_invalid_escapes(input_string):
    # Define a regular expression pattern to match invalid escape sequences
    invalid_escape_pattern = r"\\(?!\\|/|[bfnrtvN])"

    # Use re.sub to replace invalid escape sequences with an empty string
    cleaned_string = re.sub(invalid_escape_pattern, "", input_string)

    return cleaned_string


def merge_small_blocks(blocks: List[dict], min_block_size: int = 50):
    tiktoken_enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    # copy the blocks so this is not destructive
    blocks = copy.deepcopy(blocks)
    smol_blocks = [
        block
        for block in blocks
        if len(tiktoken_enc.encode(block["content"])) < min_block_size
    ]
    while smol_blocks:
        for idx in range(len(smol_blocks)):
            block_idx = blocks.index(smol_blocks[idx])
            if block_idx > 0:  # Merge with previous block if not the first block
                blocks[block_idx - 1]["content"] += smol_blocks[idx]["content"]
                blocks[block_idx - 1]["num_tokens"] += smol_blocks[idx]["num_tokens"]
            elif (
                block_idx < len(blocks) - 1
            ):  # Merge with next block if not the last block
                blocks[block_idx + 1]["content"] += smol_blocks[idx]["content"]
                blocks[block_idx + 1]["num_tokens"] += smol_blocks[idx]["num_tokens"]
            blocks.pop(block_idx)  # Remove the current small block

        # Update the list of small blocks after merging
        smol_blocks = [
            block
            for block in blocks
            if len(tiktoken_enc.encode(block["content"])) < min_block_size
        ]
    return blocks


# PARSE ARXIV METADATA SNAPSHOT FOR CATEGORY
# -----------------------------------------------------------
category = "machine-learning"  # astrophysics
if category == "astrophysics":
    if os.path.exists("data/arxiv-meta-astro-ph-2017-2023.json"):
        print("Loading arxiv papers.")
        with open("data/arxiv-meta-astro-ph-2017-2023.json", "r") as f:
            selected_metadata = json.load(f)
    else:
        print("Parsing arxiv papers.")
        selected_metadata = {}
        with open("data/arxiv-metadata-oai-snapshot.json", "r") as f:
            for line in tqdm(f.readlines()):
                metadata = json.loads(line)
                parsed_date = datetime.strptime(
                    metadata["versions"][0]["created"], "%a, %d %b %Y %H:%M:%S %Z"
                )
                year = parsed_date.year
                if year > 2016:
                    print(metadata["categories"])
                    if "astro-ph" in metadata["categories"]:
                        selected_metadata[metadata["id"]] = metadata
elif category == "machine-learning":
    if os.path.exists("data/arxiv-meta-ml-2020-2023.json"):
        print("Loading arxiv papers.")
        with open("data/arxiv-meta-ml-2020-2023.json", "r") as f:
            selected_metadata = json.load(f)
    else:
        print("Parsing arxiv papers.")
        selected_metadata = {}
        with open("data/arxiv-metadata-oai-snapshot.json", "r") as f:
            for line in tqdm(f.readlines()):
                metadata = json.loads(line)
                parsed_date = datetime.strptime(
                    metadata["versions"][0]["created"], "%a, %d %b %Y %H:%M:%S %Z"
                )
                year = parsed_date.year
                if year > 2020:
                    if "cs.LG" in metadata["categories"]:
                        selected_metadata[metadata["id"]] = metadata
    with open("data/arxiv-meta-ml-2020-2023.json", "w") as f:
        json.dump(selected_metadata, f)
else:
    raise ValueError("invalid category")


# SCRAPE OPENALEX WORKS
# -----------------------------------------------------------
oa_works = {}
if os.path.exists(f"data/{category}/oa_works_w_arxiv.json"):
    pass
elif os.path.exists(f"data/{category}/oa_works.pkl"):
    print("Loading OA Works papers.")
    with open(f"data/{category}/oa_works.pkl", "rb") as f:
        oa_works = pickle.load(f)
else:
    print("Scraping OpenAlex for Works.")
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(fetch_works, chunk) for chunk in chunks(selected_metadata)
        ]
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            result = future.result()
            if result.status_code == 200:
                result = result.json()["results"]
            else:
                print(result.text)
                continue
            for oa_work in result:
                for arxiv_id, metadata in selected_metadata.items():
                    if metadata["doi"] == oa_work["doi"].replace(
                        "https://doi.org/", ""
                    ):
                        oa_works[arxiv_id] = oa_work

    with open(f"data/{category}/oa_works.pkl", "wb") as f:
        pickle.dump(oa_works, f)

# DOWNLOAD ARXIV PAPERS
# -----------------------------------------------------------
if os.path.isdir(f"data/{category}/papers"):
    print("We already have the Arxiv paper source files.")
    pass
else:
    print("Scraping the Arxiv source files.")
    os.makedirs(f"data/{category}/papers")
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for arxiv_id, oa_work in tqdm(oa_works.items()):
            futures.append(executor.submit(download_pdf, arxiv_id, category))
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            pass


# GET LATEX SOURCE FROM ARXIV ARCHIVE
# -----------------------------------------------------------
papers = {}
if os.path.exists(f"data/{category}/papers.json"):
    print("Loading the latex source.")
    with open(f"data/{category}/papers.json", "r") as f:
        papers = json.load(f)
else:
    print("Parsing the latex source.")
    for archive in tqdm(os.listdir(f"data/{category}/papers")):
        if archive.endswith(".zip"):
            try:
                with gzip.open(f"data/{category}/papers/" + archive, "rb") as gzip_file:
                    # Open the Tar archive from the Gzip file
                    with tarfile.open(fileobj=gzip_file, mode="r") as tar_archive:
                        # Get a list of all the members (files and directories) in the Tar archive
                        members = tar_archive.getmembers()

                        # Iterate through the members and print the names of files
                        for member in members:
                            if member.isfile():
                                if member.name.endswith(".tex"):
                                    try:
                                        contents = (
                                            tar_archive.extractfile(member)
                                            .read()
                                            .decode("utf-8")
                                        )
                                        if "\section{Introduction}" not in contents:
                                            print(
                                                f"Skipping {member.name} - no introduction found"
                                            )
                                            continue
                                        else:
                                            papers[archive.split(".zip")[0]] = contents
                                            break
                                    except Exception:
                                        pass
            except Exception as e:
                print(f"Error extracting {archive}: {e}")
                pass
    json.dump(papers, open(f"data/{category}/papers.json", "w"))


# FILTER OA_WORKS WITHOUT ARXIV PAPER
# -----------------------------------------------------------
oa_works_w_arxiv = {}
if os.path.exists(f"data/{category}/oa_works_w_arxiv.json"):
    print("Loading the filtered OA Work blobs w/ arxiv paper latex.")
    with open(f"data/{category}/oa_works_w_arxiv.json", "r") as f:
        oa_works_w_arxiv = json.load(f)
else:
    print("Filtering the OA Works for just those w/ arxiv paper latex.")
    for arxiv_id, paper in tqdm(papers.items()):
        for oa_arxiv_id, oa_work in oa_works.items():
            if oa_arxiv_id == arxiv_id:
                oa_work["paper"] = paper
                oa_works_w_arxiv[arxiv_id] = oa_work
                break
    json.dump(oa_works_w_arxiv, open(f"data/{category}/oa_works_w_arxiv.json", "w"))

if category == "astrophysics":
    paper_samples = json.load(
        open(
            os.path.join(
                data_root,
                "paper_samples_concept_0.4_n_100_weighting_50_inst_50_conc.json",
            ),
            "r",
        )
    )[0]
else:
    paper_samples = list(oa_works_w_arxiv.keys())[-100:]


# EMBED BLOCKS
# -----------------------------------------------------------
dataset_step_0 = {}
if os.path.exists(f"data/{category}/dataset_step_0.pkl"):
    print("Loading embedded blocks.")
    with open(f"data/{category}/dataset_step_0.pkl", "rb") as f:
        dataset_step_0 = pickle.load(f)
else:
    print("Embedding blocks.")
    for arxiv_id in tqdm(paper_samples):
        if arxiv_id not in oa_works_w_arxiv:
            continue
        try:
            data = oa_works_w_arxiv[arxiv_id]
            blocks = parse_latex(data["paper"], data["title"], data["publication_date"])
            blocks = blocks[:10]
            model_name = "RemoteLlama-2-70b-chat-hf"

            cleaned_blocks = []
            for block in tqdm(blocks):
                cleaned = clean_content(block.content, model_name=model_name)
                block.content = cleaned

            merged_blocks = merge_small_blocks(cleaned_blocks)
            final_blocks = []
            for block in merged_blocks:
                if block.num_tokens > 450:
                    new_blocks = split_to_paragraphs(block, model_name=model_name)
                    final_blocks.extend(new_blocks)
                else:
                    final_blocks.append(block)
            data["blocks"] = blocks
            dataset_step_0[arxiv_id] = data
        except Exception as e:
            print(f"Error embedding blocks for {arxiv_id}: {e}")
    pickle.dump(dataset_step_0, open(f"data/{category}/dataset_step_0.pkl", "wb"))


# EXTRACT QUESTIONS
# -----------------------------------------------------------
dataset_step_1 = {}

path = f"data/{category}/dataset_step_1.pkl"
if os.path.exists(path):
    with open(path, "rb") as f:
        dataset_step_1 = pickle.load(f)
else:
    for arxiv_id, data in tqdm(dataset_step_0.items()):
        for block_id, block in tqdm(enumerate(data["blocks"])):
            block.questions = extract_questions(
                block.content, model_name="RemoteLlama-2-70b-chat-hf"
            )
            break
        dataset_step_1[arxiv_id] = data
        breakpoint()
        if (len(dataset_step_1) % 10) == 0:
            pickle.dump(dataset_step_1, open(path, "wb"))
    pickle.dump(dataset_step_1, open(path, "wb"))
