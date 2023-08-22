import re
import os
import requests
from tqdm import tqdm

root = "/Users/martinweiss/PycharmProjects/tn-learn/info-bazaar"
with open(f'{root}/data/awesome-llm-paper-list.md', 'r') as f:
    text = f.read()

arxiv_links = re.findall(r'https://arxiv\.org/pdf/\d+\.\d+\.pdf', text)

for link in tqdm(arxiv_links):

    r = requests.get(link, allow_redirects=True)
    os.makedirs(f"{root}/data/llm", exist_ok=True)
    open(f"{root}/data/llm/{link.split('/')[-1]}", 'wb').write(r.content)
