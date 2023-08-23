from datetime import datetime
import pickle
from concurrent.futures import ThreadPoolExecutor
import gzip
import tarfile
import json
from typing import Callable, Optional

from tqdm import tqdm
import feedparser

from bazaar.lem_utils import extract_questions, split_to_paragraphs, clean_content
import concurrent
import copy
import requests
from itertools import islice
import os
import re
import time
from pylatexenc.latex2text import LatexNodes2Text
from typing import List, Dict
import tiktoken
from bazaar.schema import Block


def chunks(data: List[Dict], size: int = 50):
    it = iter(data)
    for i in range(0, len(data), size):
        yield {k: data[k] for k in islice(it, size)}


def load_or_parse_file(path: str, parse_func: Callable, args: tuple):
    if os.path.exists(path):
        print(f"Loading data from {path}.")
        with open(path, "r") as f:
            return json.load(f)
    else:
        print(f"Parsing data for {path}.")
        data = parse_func(*args)
        with open(path, "w") as f:
            json.dump(data, f)
        return data


# PARSE ARXIV METADATA SNAPSHOT FOR CATEGORY
# -----------------------------------------------------------
def parse_arxiv_dump(relevant_category: str, min_year: int, data_root: str):
    print("Parsing arxiv papers.")
    selected_metadata = {}
    with open(f"{data_root}/arxiv-metadata-oai-snapshot.json", "r") as f:
        for line in tqdm(f.readlines()):
            metadata = json.loads(line)
            parsed_date = datetime.strptime(
                metadata["versions"][0]["created"], "%a, %d %b %Y %H:%M:%S %Z"
            )
            year = parsed_date.year
            if year > min_year:
                if relevant_category in metadata["categories"]:
                    selected_metadata[metadata["id"]] = metadata
    return selected_metadata


def get_llm_metadata(data_root: str):
    with open(f"{data_root}/llm/awesome-llm-paper-list.md", "r") as f:
        text = f.read()

    arxiv_links = re.findall(r"https://arxiv\.org/pdf/\d+\.\d+\.pdf", text)
    arxiv_abs = re.findall(r"https://arxiv\.org/abs/\d+\.\d+", text)
    arxiv_links += [f"{link.replace('abs', 'pdf')}.pdf" for link in arxiv_abs]

    def fetch_arxiv_details(arxiv_link):
        arxiv_id = arxiv_link.split("/")[-1].split(".pdf")[0]
        paper_path = f"{data_root}/llm/papers/{arxiv_link.split('/')[-1]}"

        url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
        feed = feedparser.parse(url)
        entry = feed.entries[0]
        from urllib.parse import quote_plus

        encoded_title = quote_plus(entry.title)
        url = f"https://api.crossref.org/works?query.title={encoded_title}"
        x = requests.get(url)
        try:
            doi = x.json()['message']['items'][0]['DOI']
        except Exception as e:
            doi = None
        details = {
            "id": entry.id.split("/abs/")[-1],
            "submitter": getattr(entry, "author", None),
            "authors": ", ".join(author.name for author in entry.authors),
            "title": entry.title,
            "comments": getattr(entry, "summary", None),
            "journal-ref": getattr(entry, "journal_ref", None),
            "doi": doi,
            "report-no": getattr(entry, "report_no", None),
            "categories": ", ".join(tag["term"] for tag in entry.tags),
            "license": getattr(entry, "license", None),
            "abstract": entry.summary,
            "versions": [
                {
                    "version": entry.id.split("/")[-1].split("v")[-1],
                    "created": entry.published,
                }
            ],
            "update_date": entry.updated,
            "authors_parsed": [
                [author.name.split(" ")[-1], " ".join(author.name.split(" ")[:-1]), ""]
                for author in entry.authors
            ],
        }

        if not os.path.exists(paper_path):
            try:
                r = requests.get(arxiv_link, allow_redirects=True)
                r.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
                os.makedirs(f"{data_root}/llm/papers/", exist_ok=True)
                open(paper_path, "wb").write(r.content)
            except requests.exceptions.RequestException as e:
                # Handle exception, e.g., log the error, retry, or ignore
                print(f"An error occurred while fetching {arxiv_link}: {str(e)}")

        return arxiv_id, details
    metadata = {}
    for arxiv_link in tqdm(arxiv_links):
        arxiv_id, details = fetch_arxiv_details(arxiv_link)
        metadata[arxiv_id] = details

    with open(f"{data_root}/llm/arxiv-meta-llm.json", "w") as f:
        json.dump(metadata, f)
    return metadata


def load_or_parse_arxiv_data(category: str, data_root: str):
    if category == "astrophysics":
        path = f"{data_root}/astrophysics/arxiv-meta-astro-ph-2017-2023.json"
        min_year = 2016
        relevant_category = "astro-ph"
        metadata = load_or_parse_file(
            path, parse_arxiv_dump, [relevant_category, min_year, data_root]
        )
    elif category == "machine-learning":
        path = f"{data_root}/machine-learning/arxiv-meta-ml-2020-2023.json"
        min_year = 2020
        relevant_category = "cs.LG"
        metadata = load_or_parse_file(
            path, parse_arxiv_dump, [relevant_category, min_year, data_root]
        )
    elif category == "llm":
        path = f"{data_root}/llm/arxiv-meta-llm.json"
        metadata = load_or_parse_file(path, get_llm_metadata, [data_root,])
    else:
        raise ValueError("invalid category")
    return metadata


# SCRAPE OPENALEX WORKS
# -----------------------------------------------------------
def fetch_works(chunk):
    dois = "|".join(
        metadata.get("doi") for metadata in chunk.values() if metadata.get("doi")
    )
    time.sleep(0.25)
    result = requests.get(
        f"https://api.openalex.org/works?mailto=weissmar@mila.quebec&filter=doi:{dois}"
    )
    return result


def load_or_scrape_openalex_works(
    category: str, metadata: Dict[str, dict], data_root: str
):
    oa_works = {}
    pickle_path = f"{data_root}/{category}/oa_works.pkl"

    if os.path.exists(pickle_path):
        print("Loading OA Works papers.")
        with open(pickle_path, "rb") as f:
            oa_works = pickle.load(f)
    else:
        print("Scraping OpenAlex for Works.")
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(fetch_works, chunk) for chunk in chunks(metadata)
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
                    for arxiv_id, data in metadata.items():
                        if data["doi"] == oa_work["doi"].replace(
                            "https://doi.org/", ""
                        ):
                            oa_works[arxiv_id] = oa_work

        with open(pickle_path, "wb") as f:
            pickle.dump(oa_works, f)

    return oa_works


# DOWNLOAD ARXIV PAPERS
# -----------------------------------------------------------
def download_pdf(arxiv_id: str, category: str, data_root: str) -> None:
    url = f"https://arxiv.org/e-print/{arxiv_id}"
    response = requests.get(url, stream=True)
    time.sleep(0.025)

    if response.status_code == 200:
        with open(f"{data_root}/{category}/papers/{arxiv_id}.zip", "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
    else:
        print(f"Error downloading file, status code: {response.status_code}")


def download_arxiv_papers(category: str, oa_works: dict, data_root: str):
    papers_dir = f"{data_root}/{category}/papers"
    if os.path.isdir(papers_dir):
        print("We already have the Arxiv paper source files.")
    else:
        print("Scraping the Arxiv source files.")
        os.makedirs(papers_dir)
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for arxiv_id, oa_work in tqdm(oa_works.items()):
                futures.append(executor.submit(download_pdf, arxiv_id, category, data_root))
            for future in tqdm(
                concurrent.futures.as_completed(futures), total=len(futures)
            ):
                pass


# GET LATEX SOURCE FROM ARXIV ARCHIVE
# -----------------------------------------------------------


def parse_latex(arxiv_id: str, latex_string: str, title: str, publication_date: str):
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


def load_or_parse_latex_source(category: str, data_root: str) -> Dict[str, dict]:
    papers = {}
    papers_path = f"{data_root}/{category}/papers.json"
    archive_path = f"{data_root}/{category}/papers"

    if os.path.exists(papers_path):
        print("Loading the latex source.")
        with open(papers_path, "r") as f:
            papers = json.load(f)
    else:
        print("Parsing the latex source.")
        for archive in tqdm(os.listdir(archive_path)):
            if archive.endswith(".zip"):
                try:
                    with gzip.open(archive_path + "/" + archive, "rb") as gzip_file:
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
                                                papers[
                                                    archive.split(".zip")[0]
                                                ] = contents
                                                break
                                        except Exception:
                                            pass
                except Exception as e:
                    print(f"Error extracting {archive}: {e}")
                    pass

        json.dump(papers, open(papers_path, "w"))

    return papers


# FILTER OA_WORKS WITHOUT ARXIV PAPER
# -----------------------------------------------------------
def filter_and_load_oa_works(category, papers, oa_works, data_root=None):
    oa_works_w_arxiv = {}
    file_path = f"{data_root}/{category}/oa_works_w_arxiv.json"
    if os.path.exists(file_path):
        print("Loading the filtered OA Work blobs w/ arxiv paper latex.")
        with open(file_path, "r") as f:
            oa_works_w_arxiv = json.load(f)
    else:
        print("Filtering the OA Works for just those w/ arxiv paper latex.")
        for arxiv_id, paper in tqdm(papers.items()):
            for oa_arxiv_id, oa_work in oa_works.items():
                if oa_arxiv_id == arxiv_id:
                    oa_work["paper"] = paper
                    oa_works_w_arxiv[arxiv_id] = oa_work
                    break
        json.dump(oa_works_w_arxiv, open(file_path, "w"))

    if category == "astrophysics":
        paper_samples = json.load(
            open(
                os.path.join(
                    data_root,
                    "machine-learning",
                    "paper_samples_concept_0.4_n_100_weighting_50_inst_50_conc.json",
                ),
                "r",
            )
        )[0]
    else:
        paper_samples = list(oa_works_w_arxiv.keys())[-100:]

    return oa_works_w_arxiv, paper_samples


# EMBED BLOCKS
# -----------------------------------------------------------


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


def build_blocks(
    category: str,
    oa_works_w_arxiv: dict,
    data_root: str,
    paper_samples: Optional[dict] = None,
    model_name: str = "RemoteLlama-2-70b-chat-hf",
):
    dataset_step_0 = {}
    file_path = f"{data_root}/{category}/dataset_step_0.pkl"

    if os.path.exists(file_path):
        print("Loading embedded blocks.")
        with open(file_path, "rb") as f:
            dataset_step_0 = pickle.load(f)
    else:
        print("Embedding blocks.")
        if paper_samples is None:
            paper_samples = list(oa_works_w_arxiv.keys())[-100:]
        for arxiv_id in tqdm(paper_samples):
            if arxiv_id not in oa_works_w_arxiv:
                continue
            try:
                data = oa_works_w_arxiv[arxiv_id]
                blocks = parse_latex(
                    arxiv_id, data["paper"], data["title"], data["publication_date"]
                )

                cleaned_blocks = []
                for block in tqdm(blocks):
                    block.content = clean_content(block.content, model_name=model_name)

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
        pickle.dump(dataset_step_0, open(file_path, "wb"))

    return dataset_step_0


# EXTRACT QUESTIONS
# -----------------------------------------------------------
def extract_questions_from_blocks(category, dataset_step_0, model_name):
    dataset_step_1 = {}
    path = f"data/{category}/dataset_step_1.pkl"

    if os.path.exists(path):
        with open(path, "rb") as f:
            dataset_step_1 = pickle.load(f)
    else:
        for arxiv_id, data in tqdm(dataset_step_0.items()):
            for block_id, block in tqdm(enumerate(data["blocks"])):
                block.questions = extract_questions(
                    block.content, model_name=model_name
                )
            dataset_step_1[arxiv_id] = data
            if (len(dataset_step_1) % 10) == 0:
                pickle.dump(dataset_step_1, open(path, "wb"))
        pickle.dump(dataset_step_1, open(path, "wb"))

    return dataset_step_1
