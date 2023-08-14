import concurrent
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
import tiktoken
from tqdm import tqdm

EMBEDDING_MODEL = "text-embedding-ada-002"

if Path("~").expanduser().name == "nrahaman":
    data_root = "/Users/nrahaman/Python/info-bazaar/data"
else:
    # data_root = "/Users/martinweiss/PycharmProjects/tn-learn/info-bazaar/data"
    data_root = "/home/mila/w/weissmar/scratch/tn/info-bazaar/data"
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


def parse_latex(latex_string, model_name="gpt-3.5-turbo"):
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
        blocks = re.split(r"\n\n\s*", content_linebreak)  # Split by blank lines
        blocks = [
            paragraph.strip() for paragraph in blocks if paragraph.strip()
        ]  # Remove leading and trailing whitespace and ignore empty paragraphs
        blocks = [
            LatexNodes2Text().latex_to_text(paragraph).strip() for paragraph in blocks
        ]
        for idx, block in enumerate(blocks):
            num_tokens = len(tiktoken.encoding_for_model(model_name).encode(block))
            if num_tokens < 50 or num_tokens > 300:
                continue
            all_blocks.append(
                {
                    "block_id": f"{arxiv_id}/{section_title}/{idx}",
                    "content": block,
                    "num_tokens": num_tokens,
                }
            )
    return all_blocks


def remove_invalid_escapes(input_string):
    # Define a regular expression pattern to match invalid escape sequences
    invalid_escape_pattern = r"\\(?!\\|/|[bfnrtvN])"

    # Use re.sub to replace invalid escape sequences with an empty string
    cleaned_string = re.sub(invalid_escape_pattern, "", input_string)

    return cleaned_string

def extract_nuggets(block):
    system = """
Socrates and Plato sit under a tree, discussing the nature of truth and knowledge. They have a scroll in front of them containing scientific texts. Socrates believes in extracting questions and answers that are factual and based on the content of the text. Plato, on the other hand, emphasizes that these answers must be objective assertions that describe reality and are supported by evidence.
    
Socrates: "Knowledge, my dear Plato, must be empirical and verifiable. Our task is to extract questions and answers from this scroll that adhere to this principle."
    
Plato: "Agreed, Socrates. But each answer must be comprehensive, providing context and depth. They should be reminiscent of the great archives, like an excerpt from our Athenian repositories."
        
Now, my dear philosophers, you must propose questions and factual statements based on content provided by the user. You will simulate a long argument with each other about which is the best question and answer for this passage. At the end of the argument, arrive at a single verdict. This verdict must be printed as: 

---

VERDICT:
    
question: <answer>
answer: <answer>
"""

    content = f"""The scientific text is as follows: {block['content']}"""

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": content},
    ]
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.2,
            max_tokens=1536,
        )
        content = remove_invalid_escapes(response.choices[0]["message"]["content"])
        question_pattern = r'question: \"(.*?)\"'
        answer_pattern = r'answer: \"(.*?)\"'
        
        question_match = re.search(question_pattern, content)
        answer_match = re.search(answer_pattern, content)
        
        question = question_match.group(1) if question_match else None
        answer = answer_match.group(1) if answer_match else None

    except Exception as e:
        print(e)
        question = None
        answer = None
    return question, answer

def embed_nuggets(nuggets):
    BATCH_SIZE = 1000  # you can submit up to 2048 embedding inputs per request
    nugget_questions = [nugget["question"] for nugget in nuggets]
    nugget_answers = [nugget["answer"] for nugget in nuggets]
    embeddings = []
    for batch_start in range(0, len(nuggets), BATCH_SIZE):
        batch_end = batch_start + BATCH_SIZE
        batch = nugget_answers[batch_start:batch_end]
        print(f"Batch {batch_start} to {batch_end - 1}")
        response = openai.Embedding.create(model=EMBEDDING_MODEL, input=batch)
        batch_embeddings = [e["embedding"] for e in response["data"]]
        embeddings.extend(batch_embeddings)
    embeddings = [list(embedding) for embedding in embeddings]

    for idx, embedding in enumerate(embeddings):
        nuggets[idx]['embedding'] = embedding
    return nuggets

# PARSE ARXIV METADATA SNAPSHOT FOR CATEGORY
# -----------------------------------------------------------
category = "machine-learning" # astrophysics
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
                    if metadata['doi'] == oa_work['doi'].replace("https://doi.org/", ""):
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
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
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
                data_root, "paper_samples_concept_0.4_n_100_weighting_50_inst_50_conc.json",
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
            blocks = parse_latex(data["paper"])
            block_contents = [block['content'] for block in blocks]
            response = openai.Embedding.create(model=EMBEDDING_MODEL, input=block_contents)
            batch_embeddings = [e["embedding"] for e in response["data"]]
            for idx, block in enumerate(blocks):
                block['embedding'] = batch_embeddings[idx]
            data["blocks"] = blocks
            dataset_step_0[arxiv_id] = data
        except Exception as e:
            print(e)

    pickle.dump(dataset_step_0, open(f"data/{category}/dataset_step_0.pkl", "wb"))


# EXTRACT NUGGETS
# -----------------------------------------------------------
dataset_step_1 = {}

if os.path.exists(f"data/{category}/dataset_step_1.json"):
    with open(f"data/{category}/dataset_step_1.json", "r") as f:
        dataset_step_1 = json.load(f)
else:

    for arxiv_id, data in tqdm(dataset_step_0.items()):
        for block_id, block in tqdm(enumerate(data['blocks'])):
            try:
                question, answer = extract_nuggets(block)
                nuggets = [{"question": question, "answer": answer}]
                embedding_dict = embed_nuggets(nuggets)
                block['nuggets'] = embedding_dict
            except Exception as e:
                print(e)
                block['nuggets'] = {}
            break
        try:
            data["vendor_id"] = data["authorships"][0]["institutions"][0]["id"].replace(
                "https://openalex.org/", ""
            )
        except Exception as e:
            data["vendor_id"] = ""
        dataset_step_1[arxiv_id] = data
        if (len(dataset_step_1) % 10) == 0:
            json.dump(
                dataset_step_1,
                open(f"data/{category}/dataset_step_1.json", "w"),
            )
        time.sleep(2)
        if len(dataset_step_1) == 100:
            break
    json.dump(dataset_step_1, open(f"data/{category}/dataset_step_1.json", "w"))