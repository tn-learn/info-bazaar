import concurrent
import json
import os.path
from tqdm import tqdm
import time
from datetime import datetime
import requests
from itertools import islice
from concurrent.futures import ThreadPoolExecutor

def chunks(data, size=50):
    it = iter(data)
    for i in range(0, len(data), size):
        yield {k: data[k] for k in islice(it, size)}

def fetch_works(chunk):
    dois = "|".join(metadata.get('doi') for metadata in chunk.values() if metadata.get('doi'))
    time.sleep(0.25)
    result = requests.get(f"https://api.openalex.org/works?mailto=weissmar@mila.quebec&filter=doi:{dois}")
    return result

if os.path.exists("data/arxiv-meta-astro-ph-2017-2023.json"):
    with open("data/arxiv-meta-astro-ph-2017-2023.json", "r") as f:
        selected_metadata = json.load(f)
else:
    selected_metadata = {}
    with open("data/arxiv-metadata-oai-snapshot.json", "r") as f:
        for line in tqdm(f.readlines()):
            metadata = json.loads(line)
            parsed_date = datetime.strptime(metadata['versions'][0]['created'], '%a, %d %b %Y %H:%M:%S %Z')
            year = parsed_date.year
            if year > 2016:
                print(metadata["categories"])
                if "astro-ph" in metadata["categories"]:
                    selected_metadata[metadata['id']] = metadata

oa_works = {}
if os.path.exists("data/oa_works.json"):
    with open("data/oa_works.json", "r") as f:
        oa_works = json.load(f)

else:
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(fetch_works, chunk) for chunk in chunks(selected_metadata)]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            result = future.result()
            if result.status_code == 200:
                result = result.json()['results']
            else:
                print(result.text)
                continue
            for oa_work in result:
                oa_works[oa_work['doi']] = oa_work

    with open("data/oa_works.json", "w") as f:
        json.dump(oa_works, f)

if os.path.exists("data/oa_merge.json"):
    with open("data/oa_merge.json", "r") as f:
        oa_works = json.load(f)
else:
    for doi, oa_work in tqdm(oa_works.items()):
        for arxiv_id, metadata in selected_metadata.items():
            if f"https://doi.org/{metadata.get('doi')}" == doi:
                oa_work['arxiv_id'] = arxiv_id
                oa_work["metadata"] = metadata
                break
    json.dump(oa_works, open("data/oa_merge.json", "w"))

def download_pdf(arxiv_id):
    url = f"https://arxiv.org/e-print/{arxiv_id}"
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(f"data/papers/{arxiv_id}.zip", "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
    else:
        print(f"Error downloading file, status code: {response.status_code}")


with ThreadPoolExecutor() as executor:
    futures = []
    for doi, oa_work in tqdm(oa_works.items()):
        if not oa_work.get("arxiv_id"):
            continue
        futures.append(executor.submit(download_pdf, oa_work['arxiv_id']))
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        pass
pass
import os
import gzip
import tarfile
import json
from tqdm import tqdm

papers = {}
for archive in tqdm(os.listdir('data/papers')):
    if archive.endswith('.zip'):
        try:
            with gzip.open("data/papers/" + archive, 'rb') as gzip_file:
                # Open the Tar archive from the Gzip file
                with tarfile.open(fileobj=gzip_file, mode='r') as tar_archive:
                    # Get a list of all the members (files and directories) in the Tar archive
                    members = tar_archive.getmembers()

                    # Iterate through the members and print the names of files
                    for member in members:
                        if member.isfile():
                            if member.name.endswith(".tex"):
                                try:
                                    papers[archive.split(".zip")[0]] = tar_archive.extractfile(member).read().decode('utf-8')
                                except Exception:
                                    pass
        except Exception as e:
            print(f"Error extracting {archive}: {e}")
            pass

json.dump(papers, open("data/papers.json", "w"))

papers = json.load(open("data/papers.json", "r"))
oa_meta = json.load(open("data/oa_merge.json", "r"))
breakpoint()
dataset = {}
for arxiv_id, paper in tqdm(papers.items()):
    for doi, oa_work in oa_meta.items():
        if oa_work.get("arxiv_id") == arxiv_id:
            oa_work["paper"] = paper
            dataset[arxiv_id] = oa_work
            break
json.dump(dataset, open("data/dataset.json", "w"))
