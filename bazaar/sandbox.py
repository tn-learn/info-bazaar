import os
import json
import openai
from tqdm import tqdm

dataset = json.load(open("data/dataset.json", "r"))

# Set the API key
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Create a prompt
prompt = """
You will see some scientific text. Your task is return a list of factual statements extracted from the text. The elements of this list should be self-contained and not depend on other elements, but can contain multiple related sentences.

It is very important for you to return these facts as a JSON. For example:
```
[
  <statement 1>,
  <statement 2>,
  ... <more>
]
```
----------
[
"""
breakpoint()

for datum in tqdm(dataset.values()):
  paper = datum['paper']
  if "\section{Introduction}" in paper:
    intro = paper.split("\section{Introduction}")[1].split("\section")[0]
    prompt += intro
    response = openai.Completion.create(
      engine="text-davinci-003",
      prompt=prompt,
      max_tokens=1024
    )

    print(response.choices[0].text.strip())

breakpoint()
print(dataset['1702.04431'])
